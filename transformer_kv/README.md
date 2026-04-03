# Transformer Self-Attention — K・V 計算ステップ デモ

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo/blob/main/transformer_kv_demo.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Attention%20Is%20All%20You%20Need-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/1706.03762)

Scaled Dot-Product Attention の Q・K・V 計算を、小さな数値例（T=3 トークン、d=2 次元）で段階的に可視化したノートブックです。

```
Attention(Q, K, V) = softmax( QKᵀ / √dk ) · V
```

---

## 目次

1. [アーキテクチャの概要](#アーキテクチャの概要)
2. [LLM における Self-Attention の役割](#llm-における-self-attention-の役割)
3. [Q・K・V の数理](#qkv-の数理)
4. [計算ステップ](#計算ステップ)
5. [使い方](#使い方)

---

## アーキテクチャの概要

Self-Attention は「各トークンが他のすべてのトークンにどれだけ注目すべきか」を計算する機構です。入力系列 $X \in \mathbb{R}^{T \times d}$ から Q・K・V を線形射影し、注目度を確率として求めた上で Value の加重平均を出力します。

---

## LLM における Self-Attention の役割

### Transformer ブロックの中での位置づけ

GPT・BERT・LLaMA などの LLM は、Transformer ブロックを $N$ 層積み重ねた構造を持ちます。各ブロックは Self-Attention と Feed-Forward Network（FFN）の2つのサブレイヤーで構成されています。

```
入力トークン列
      │
      ▼
┌─────────────────────────┐
│  Transformer Block × N  │
│                         │
│  ┌───────────────────┐  │
│  │  Self-Attention   │  │  ← Q・K・V はここで計算される
│  └────────┬──────────┘  │
│           │ + Residual  │
│  ┌────────▼──────────┐  │
│  │  Feed-Forward Net │  │  ← 位置ごとの非線形変換
│  └────────┬──────────┘  │
│           │ + Residual  │
└───────────┼─────────────┘
            ▼
         次の層へ
```

Self-Attention は **「トークン間の関係を集約する」** 役割を担い、FFN は **「各トークンの表現を個別に変換する」** 役割を担います。両者が交互に積み重なることで、局所的な特徴と大域的な文脈が層を経るごとに精緻化されていきます。

### Self-Attention が解決する本質的な問題

LLM が言語を扱う上で避けられない問題は **長距離依存（long-range dependency）** です。たとえば次のような文では、"it" が何を指すかを理解するために文の先頭まで参照する必要があります。

```
"The animal didn't cross the street because it was too tired."
                                              ↑
                                     "animal" を指す（"street" ではない）
```

RNN はこの問題を隠れ状態の逐次伝播で解こうとしましたが、系列長が長くなると勾配消失によって遠い位置の情報が失われていました。Self-Attention は全トークン間の関係を **一度の行列演算で並列計算** するため、距離に依存せず任意のトークンペアを直接参照できます。

### 層ごとに異なる抽象度の文脈を学習

Self-Attention は単一ではなく、**Multi-Head Attention** として複数のヘッドを並列に実行します。

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_Q^i,\; K W_K^i,\; V W_V^i)
$$

各ヘッドは異なる部分空間に射影するため、同じ入力から複数の異なる「注目パターン」を並列に学習できます。研究によって、ヘッドごとに構文的な依存関係・照応関係・位置的な近接性など、異なる言語的特徴を捉えることが示されています。

さらに Transformer を深く積み重ねることで、浅い層では局所的な構文（品詞・係り受け）、深い層では意味的・談話的な抽象表現（照応・推論・常識知識）が形成されることが知られています。

### Self-Attention の計算量と LLM のスケーリング

Self-Attention のアテンション行列は $T \times T$ の密行列であり、系列長 $T$ に対して **時間・空間計算量が $O(T^2)$** となります。これが長文処理のボトルネックとなるため、大規模な LLM ではスパースアテンション・線形アテンション・スライディングウィンドウなど様々な近似手法が研究されています。

| モデル | パラメータ数 | Attention の工夫 |
|--------|-------------|-----------------|
| GPT-2 | 1.5B | 標準 Self-Attention |
| GPT-3 | 175B | 標準 Self-Attention |
| LLaMA 2 | 7B〜70B | Grouped Query Attention (GQA) |
| Mistral | 7B | Sliding Window Attention |

Self-Attention は LLM の性能の中核を担う機構であり、**パラメータ数・文脈長・ヘッド数のスケーリング**が言語理解能力の向上に直結しています。

---

## Q・K・V の数理

### 1. 線形射影

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

| 記号 | 形状 | 役割 |
|------|------|------|
| $X$ | $T \times d_{\text{model}}$ | 入力トークン列（埋め込み済み） |
| $W_Q, W_K$ | $d_{\text{model}} \times d_k$ | Query・Key の投影行列 |
| $W_V$ | $d_{\text{model}} \times d_v$ | Value の投影行列 |
| $Q, K$ | $T \times d_k$ | 「何を聞くか」「何を持つか」 |
| $V$ | $T \times d_v$ | 「何を渡すか」 |

$W_Q, W_K, W_V$ はモデルが学習するパラメータです。

### 2. スケーリングとスコア計算

$$
\text{scores} = \frac{Q K^\top}{\sqrt{d_k}}
$$

- $QK^\top \in \mathbb{R}^{T \times T}$ の $(i,j)$ 成分はトークン $i$ と $j$ の内積（類似度）
- $\sqrt{d_k}$ による除算は **スケーリング** と呼ばれる正規化で、$d_k$ が大きくなると内積の分散が $d_k$ 倍に増大するため、Softmax の勾配消失を防ぐ

### 3. Softmax による正規化

$$
A = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right), \quad A_{ij} = \frac{\exp(\text{scores}_{ij})}{\sum_{k} \exp(\text{scores}_{ik})}
$$

- 各行の合計が 1 となり、**注目の確率分布** として解釈できる
- $A_{ij}$ が大きいほど、トークン $i$ はトークン $j$ を強く参照する

### 4. Value の加重平均（最終出力）

$$
\text{Output} = A \cdot V, \quad \text{Output}_i = \sum_{j} A_{ij} \cdot V_j
$$

- トークン $i$ の出力は、全トークンの Value を注目度で重み付けして合算したもの
- **Query と Key が決める「どこを見るか」** と **Value が決める「何を受け取るか」** が分離されている点が本機構の核心

### 数式まとめ

$$
\boxed{\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V}
$$

---

## 計算ステップ

### Step 1 — 入力トークン列 X

3 トークン（"The" / "cat" / "sat"）を 2 次元埋め込みで表現した入力行列。

![Step 1: Input X](0.png)

---

### Step 2 — 重み行列 Wq・Wk・Wv

学習によって最適化される 3 つの投影行列。ここでは固定値を使用。

![Step 2: Weight matrices](1.png)

---

### Step 3 — Q・K・V の計算

$Q = X W_Q$、$K = X W_K$、$V = X W_V$ の行列積で射影。

![Step 3: Q, K, V projections](2.png)

---

### Step 4 — アテンションスコア

$QK^\top / \sqrt{d_k}$ によるスケーリング前後の比較。スケーリングによって値の分散が抑えられる。

![Step 4: Attention scores](3.png)

---

### Step 5 — アテンション重み（Softmax）

スコアに Softmax を適用して確率化。各行の合計が 1.0 になる。

> "cat" が他のトークンより高い注目を集めていることが可視化されている。

![Step 5: Attention weights](4.png)

---

### Step 6 — 最終出力

$\text{Output} = A \cdot V$。アテンション重みを使って Value を加重平均。

![Step 6: Output](5.png)

---

### Step 7 — 全ステップまとめ

6 つの行列演算を一枚に集約した俯瞰図。

![Step 7: All steps](6.png)

---

### ボーナス — パラメータ変更（T=5, d=4）

トークン数・次元数を変えて動作確認。

![Bonus: T=5, d=4](7.png)

---

## 使い方

### 必要環境

- Google Colab（追加インストール不要）
- `numpy`, `matplotlib`（Colab にデフォルト搭載）

### 実行手順

```
初回のみ    → セル 1（フォントセットアップ）を実行
毎回の起動  → セル 2（ステップ 0）から順に実行
```

> **注意**：カーネル再起動後はセル 1 をスキップし、**セル 2 から実行**してください。  
> フォント設定はセル 2 に統合されています。

### ファイル構成

```
.
├── transformer_kv_demo.ipynb   # メインノートブック
└── README.md
```

---

## 参考

- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- The Illustrated Transformer — Jay Alammar
