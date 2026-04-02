# 🧠 Prompt Tuning with PEFT — Japanese GPT-2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/prompt_tuning_colab.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

軽量な日本語GPT-2モデル（`rinna/japanese-gpt2-small`）を使い、**プロンプト学習（Prompt Tuning）** を Google Colab 上で体験できるノートブックです。モデルの重みを一切変更せず、わずかな「ソフトトークン」だけを学習することで、効率的なタスク適応を実現します。

---

## 📋 目次

- [環境・要件](#-環境要件)
- [クイックスタート](#-クイックスタート)
- [プロンプト学習とは](#-プロンプト学習とは)
  - [ファインチューニングとの比較](#ファインチューニングとの比較)
  - [数学的定式化](#数学的定式化)
  - [ソフトプロンプトの学習](#ソフトプロンプトの学習)
  - [初期化戦略](#初期化戦略)
- [関連手法との比較](#-関連手法との比較)
- [ノートブック構成](#-ノートブック構成)
- [実験結果](#-実験結果)
- [カスタマイズ](#-カスタマイズ)
- [参考文献](#-参考文献)

---

## 🖥 環境・要件

| 項目 | 要件 |
|------|------|
| GPU | Google Colab T4（15GB VRAM）以上 |
| Python | 3.10+ |
| PyTorch | 2.x |
| 主要ライブラリ | `peft`, `transformers`, `datasets`, `sentencepiece` |

---

## 🚀 クイックスタート

1. 右上の **"Open in Colab"** バッジをクリック
2. ランタイム → ランタイムのタイプを変更 → **T4 GPU** を選択
3. 「すべてのセルを実行」で完結します（所要時間：約5〜10分）

---

## 📖 プロンプト学習とは

### 背景

大規模言語モデル（LLM）を特定のタスクに適応させる従来の手法は **ファインチューニング（Fine-tuning）** です。しかし、数十億パラメータを持つモデルを丸ごと再学習することは、計算コスト・ストレージコストともに非常に大きな負担となります。

**プロンプト学習（Prompt Learning）** は、この問題を解決するために提案された手法群の総称です。モデルの重みを凍結（freeze）したまま、入力の一部として付加する「学習可能なベクトル」だけを最適化します。

---

### ファインチューニングとの比較

```
【ファインチューニング】

  入力テキスト ──→ [モデル全体を更新] ──→ 出力
                      ↑
                  全パラメータを学習
                  （数億〜数十億個）

【プロンプト学習】

  [ソフトトークン] + 入力テキスト ──→ [モデルは固定] ──→ 出力
        ↑
    このベクトルだけを学習
   （数百〜数千個のパラメータ）
```

| 比較項目 | ファインチューニング | プロンプト学習 |
|----------|-------------------|--------------|
| 学習パラメータ数 | 全パラメータ（100%） | ソフトトークンのみ（< 0.1%） |
| 計算コスト | 高 | 非常に低 |
| ストレージ | モデルごとに全重みを保存 | アダプター数KBのみ保存 |
| 破滅的忘却 | 起こりうる | 起こらない（重みが固定） |
| 複数タスク対応 | モデルを複数用意 | アダプターを切り替えるだけ |

---

### 数学的定式化

#### 通常のプロンプト（ハードプロンプト）

通常の言語モデルは、入力トークン列 $x = [x_1, x_2, \ldots, x_n]$ に対して、次のトークンの確率を計算します。

$$P(y \mid x; \theta) = \prod_{t=1}^{T} P(y_t \mid x, y_{\lt t}; \theta)$$


ここで $\theta$ はモデルの全パラメータです。ハードプロンプト（通常のテキストプロンプト） $p$ を前置した場合：

$$P(y \mid p, x; \theta)$$

この $p$ は離散的なトークン列であり、勾配が伝播しないため、最適化が困難です。

---

#### ソフトプロンプト（Prompt Tuning）

Prompt Tuning（Lester et al., 2021）では、離散的なトークンの代わりに、**連続的な学習可能ベクトル** $P_\theta \in \mathbb{R}^{k \times d}$ を導入します。

$$P_\theta = [p_1, p_2, \ldots, p_k]$$

ここで：
- $k$：ソフトトークン数（ハイパーパラメータ）
- $d$：モデルの埋め込み次元（例：GPT-2 small では $d = 768$）
- 各 $p_i \in \mathbb{R}^d$：学習可能な連続ベクトル

入力の埋め込み $X_e \in \mathbb{R}^{n \times d}$ とソフトプロンプトを結合します：

$$\tilde{X} = \text{concat}(P_\theta, X_e) \in \mathbb{R}^{(k+n) \times d}$$

モデルへの入力は $\tilde{X}$ となり、出力確率は：

$$P(y \mid \tilde{X}; \theta_{\text{frozen}}) = \prod_{t=1}^{T} P(y_t \mid \tilde{X}, y_{\lt t}; \theta_{\text{frozen}})$$

学習は $P_\theta$ のみを更新し、 $\theta_{\text{frozen}}$ は固定です：

$$P_\theta^* = \arg\min_{P_\theta} \mathcal{L}(P_\theta; \theta_{\text{frozen}})$$

---

### ソフトプロンプトの学習

損失関数は通常の言語モデリング損失（負の対数尤度）を使用します：

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(y_t \mid \tilde{X}, y_{\lt t}; \theta_{\text{frozen}})$$

勾配は $P_\theta$ に対してのみ計算されます：

$$\nabla_{P_\theta} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial P_\theta}$$

これは通常の逆伝播で計算可能であり、モデルの残りの重みには勾配が蓄積されません。

**学習パラメータ数の比較：**

$$\text{ファインチューニング}: |\theta| \approx 117 \times 10^6 \quad (\text{GPT-2 small})$$

$$\text{プロンプト学習}: |P_\theta| = k \times d = 10 \times 768 = 7{,}680$$

削減率： $\dfrac{7{,}680}{117 \times 10^6} \approx 0.0066\%$

---

### 初期化戦略

ソフトトークンの初期化は学習の収束速度と最終性能に影響します。PEFTライブラリでは以下の2通りが選択できます。

#### 1. ランダム初期化（`RANDOM`）

```python
prompt_tuning_init=PromptTuningInit.RANDOM
```

各 $p_i$ を標準正規分布からサンプリング：

$$p_i \sim \mathcal{N}(0, \sigma^2)$$

学習は収束するが、初期値依存性が高く、不安定になる場合があります。

#### 2. テキストによる初期化（`TEXT`）

```python
prompt_tuning_init=PromptTuningInit.TEXT,
prompt_tuning_init_text="映画のレビューを書いてください:"
```

指定したテキストをトークナイズし、その埋め込みベクトルを初期値として使用します：

$$p_i \leftarrow \text{Embedding}(t_i)$$

ここで $t_i$ は初期化テキストの $i$ 番目のトークンです。意味的に関連するテキストで初期化することで、収束が速く、性能が安定します（論文でも推奨）。

---

## 🔬 関連手法との比較

プロンプト学習には複数のバリエーションが存在します。

### Prefix Tuning（Li & Liang, 2021）

Prompt Tuning がモデルの**入力層**にのみソフトトークンを付加するのに対し、Prefix Tuning は**全トランスフォーマー層**のKey・Valueに学習可能なベクトルを付加します。

各層 $l$ のアテンション計算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Prefix Tuning では $K$, $V$ にプレフィックスを結合：

$$K' = \text{concat}(P_K^{(l)}, K), \quad V' = \text{concat}(P_V^{(l)}, V)$$

表現力は高いが、学習パラメータが増加します。

### P-Tuning v2（Liu et al., 2022）

Prefix Tuning と同様に全層にソフトプロンプトを付加しつつ、分類ヘッドも学習対象とします。特に NLU（自然言語理解）タスクで効果的です。

### LoRA（Hu et al., 2021）

重み行列の更新を低ランク分解で近似します：

$$W' = W + \Delta W = W + BA$$

ここで $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$（ $r \ll d$ ）。プロンプト学習と異なりモデルの内部重みを（間接的に）変更しますが、学習パラメータは大幅に削減されます。

### 各手法のまとめ

```
                    入力層のみ  全層   重みを変更
                       ↓        ↓       ↓
Prompt Tuning:         ✅       ❌      ❌   ← 本ノートブック
Prefix Tuning:         ❌       ✅      ❌
P-Tuning v2:           ❌       ✅      一部
LoRA:                  ❌       ❌      ✅（低ランク近似）
ファインチューニング:  ❌       ❌      ✅（全体）
```

---

## 📁 ノートブック構成

```
prompt_tuning.ipynb
├── ① ライブラリのインストール
│     peft / transformers / datasets / sentencepiece / accelerate
├── ② GPU確認
│     CUDA利用可否・VRAM容量の表示
├── ③ モデルとトークナイザーの読み込み
│     rinna/japanese-gpt2-small（約117Mパラメータ）
├── ④ 学習前のベースライン生成
│     プロンプト学習前の素の生成を確認
├── ⑤ PEFTでプロンプトチューニングの設定
│     PromptTuningConfig / num_virtual_tokens=10
├── ⑥ 学習データの準備
│     日本語映画レビュー8件のカスタムDataset
├── ⑦ 学習ループ
│     AdamW / 10エポック / Loss表示
├── ⑧ 学習曲線の可視化
│     matplotlib（日本語フォント対応済み）
├── ⑨ 学習後のテキスト生成
│     学習前後の生成比較
├── ⑩ アダプターの保存・読み込み
│     save_pretrained / PeftModel.from_pretrained
└── ⑪ Google Driveへの保存（オプション）
```

---

## 📊 実験結果

| 設定 | 値 |
|------|-----|
| ベースモデル | `rinna/japanese-gpt2-small`（117M params） |
| ソフトトークン数 | 10 |
| 初期化 | テキスト（`映画のレビューを書いてください:`） |
| 学習データ | 映画レビュー 8件 |
| エポック数 | 10 |
| 学習率 | 3e-3 |
| 学習パラメータ数 | 7,680（全体の約0.007%） |
| アダプターサイズ | 約60KB |

---

## 🔧 カスタマイズ

### ソフトトークン数の変更

```python
peft_config = PromptTuningConfig(
    num_virtual_tokens=20,  # デフォルト: 10（増やすと表現力UP）
    ...
)
```

一般に `num_virtual_tokens` が大きいほど表現力が増しますが、学習が安定しにくくなります。5〜100の範囲で試すとよいでしょう。

### 別タスクへの応用

`train_texts` を変えるだけで様々なタスクに適用できます。

```python
# 例: ニュース記事要約スタイルの学習
train_texts = [
    "本日の主要ニュースをお伝えします。",
    "経済指標の発表により、市場は...",
    ...
]
```

### より大きなモデルへの拡張

T4 GPU（15GB VRAM）であれば以下のモデルも動作します：

```python
# 約3Bパラメータの日本語モデル
MODEL_NAME = "cyberagent/open-calm-3b"
```

---

## 📚 参考文献

- Lester, B., Al-Rfou, R., & Constant, N. (2021). **The Power of Scale for Parameter-Efficient Prompt Tuning.** EMNLP 2021. [arxiv:2104.08691](https://arxiv.org/abs/2104.08691)
- Li, X. L., & Liang, P. (2021). **Prefix-Tuning: Optimizing Continuous Prompts for Generation.** ACL 2021. [arxiv:2101.00190](https://arxiv.org/abs/2101.00190)
- Liu, X., et al. (2022). **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks.** ACL 2022. [arxiv:2110.07602](https://arxiv.org/abs/2110.07602)
- Hu, E. J., et al. (2021). **LoRA: Low-Rank Adaptation of Large Language Models.** ICLR 2022. [arxiv:2106.09685](https://arxiv.org/abs/2106.09685)
- HuggingFace PEFT ライブラリ: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- rinna 日本語 GPT-2: [https://huggingface.co/rinna/japanese-gpt2-small](https://huggingface.co/rinna/japanese-gpt2-small)

---

## 📄 ライセンス

MIT License
