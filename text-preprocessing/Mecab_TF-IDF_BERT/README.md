# 日本語ベクトル化の比較：分かち書きあり vs BERT

日本語テキストを機械学習で扱う際の2つのアプローチ、**従来手法（MeCab + TF-IDF）** と **BERTベース手法** を比較するJupyter Notebookです。Google Colabですぐに実行できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/japanese_vectorization_comparison.ipynb)

---

## 目次

1. [背景知識：トークン化とは](#トークン化とは)
2. [背景知識：ベクトル化とは](#ベクトル化とは)
3. [比較する2つの手法](#比較する2つの手法)
4. [ノートブックの内容](#ノートブックの内容)
5. [実行方法](#実行方法)
6. [使用ライブラリ](#使用ライブラリ)

---

## トークン化とは

**トークン化（Tokenization）** とは、文章を機械が処理しやすい最小単位（トークン）に分割する処理です。

英語はスペースで単語が区切られているため、自然にトークン分割できます。

```
"I love sushi" → ["I", "love", "sushi"]
```

一方、**日本語はスペースがない**ため、そのままでは単語の境界がわかりません。

```
"私は寿司が好きです" → どこで区切る？
```

### 日本語のトークン化アプローチ

#### 1. 形態素解析（分かち書き）
MeCabなどのツールを使って、文法的な単位（形態素）に分割します。

```
"私は寿司が好きです"
→ ["私", "は", "寿司", "が", "好き", "です"]
```

辞書ベースのルールで動作するため、高速で動作が予測しやすい反面、辞書にない語（新語・固有名詞）には弱いという特徴があります。

#### 2. サブワード分割（BERTなど）
WordPieceやSentencePieceというアルゴリズムを使って、文字・単語・部分単語の単位で柔軟に分割します。

```
"私は寿司が好きです"
→ ["私", "は", "寿司", "が", "好き", "です"]  # 既知の単語はそのまま

"トランスフォーマー"（未知語の例）
→ ["トランス", "##フォーマー"]  # サブワードに分割
```

辞書にない語でもサブワードに分解して対応できるため、未知語に強いという特徴があります。

---

## ベクトル化とは

**ベクトル化（Vectorization / Embedding）** とは、テキストを数値の配列（ベクトル）に変換する処理です。機械学習モデルは文字列を直接扱えないため、数値への変換が必要です。

```
"寿司が好きです" → [0.12, -0.85, 0.33, 0.71, ...]  ← 数値の配列
```

ベクトル化によって、テキスト同士の**類似度計算**や**分類**が可能になります。意味が近い文章はベクトル空間上でも近い位置に配置されることが理想です。

```
"寿司が好きです"  →  [0.12, -0.85, 0.33, ...]
"ラーメンを食べた" →  [0.10, -0.79, 0.41, ...]  ← 近い（同じ料理カテゴリ）
"AIが進化している" →  [0.91,  0.23, -0.55, ...]  ← 遠い（別カテゴリ）
```

### ベクトル化の手法

#### 1. TF-IDF（Term Frequency - Inverse Document Frequency）
単語の出現頻度に基づいてベクトルを生成します。

- **TF（単語頻度）**：その文書内で単語が何回出現するか
- **IDF（逆文書頻度）**：全文書のうち何割の文書にその単語が出現するか（珍しい単語ほど重要と判断）

```
語彙: ["寿司", "ラーメン", "AI", "技術", ...]
文書: "寿司は美味しい"
→ [0.85, 0.00, 0.00, 0.00, ...]  ← 「寿司」だけ高いスコア
```

**特徴：**
- ベクトルの次元数 ＝ 語彙数（数千〜数万次元）になりやすい
- ほとんどの要素が0のスパース（疎）なベクトル
- 単語の意味や文脈は考慮されない（「好き」と「嫌い」は全く別の次元）

#### 2. BERTの埋め込み（Dense Embedding）
BERTはTransformerというニューラルネットワーク構造を持つモデルで、文章全体の文脈を考慮した密なベクトルを生成します。

```
文章: "銀行に行った"（金融の文脈）
→ [0.12, -0.85, 0.33, ...]  ← 「金融機関」寄りのベクトル

文章: "川の銀行で釣りをした"（河川の文脈）
→ [0.45,  0.21, -0.67, ...]  ← 「川岸」寄りのベクトル
```

**特徴：**
- 固定768次元の密なベクトル（全要素に値がある）
- 同じ単語でも文脈によって異なるベクトルが生成される
- 意味的に近い文章は近いベクトルになる傾向がある

---

## 比較する2つの手法

| 比較項目 | 従来手法（MeCab + TF-IDF） | BERTベース |
|----------|--------------------------|-----------|
| **トークン化** | MeCabで分かち書き（必須） | SentencePieceで自動分割（不要） |
| **ベクトル次元** | 語彙数に依存（スパース） | 固定768次元（密） |
| **意味の捉え方** | 単語の出現頻度ベース | 文脈を考慮した意味表現 |
| **計算コスト** | 軽量・高速 | 重い（GPUがあると速い） |
| **未知語への対応** | 辞書外の語は扱えない | サブワードで対応可能 |
| **向いているタスク** | キーワード検索・大量テキスト処理 | 意味検索・RAG・文章分類 |

---

## ノートブックの内容

```
1. ライブラリのインストール
2. サンプルテキストの準備（スポーツ・料理・テクノロジーの3カテゴリ）
3. 従来手法：MeCab（分かち書き）+ TF-IDF
4. BERTベース手法：cl-tohoku/bert-base-japanese
5. PCAによるベクトルの可視化（2次元プロット）
6. コサイン類似度マトリクスによる比較
7. まとめ
```

---

## 実行方法

### Google Colabで実行する場合

1. 上の **Open in Colab** バッジをクリック
2. セル1（インストール）を実行
3. **ランタイムを再起動**（メニュー → ランタイム → ランタイムを再起動）
   - 日本語フォントをmatplotlibに認識させるために必要です
4. セル2以降を順番に実行

> 初回実行時はBERTモデルのダウンロード（約400MB）が発生します。  
> Colabの無料GPUを使うと処理が速くなります（ランタイム → ランタイムのタイプを変更 → T4 GPU）。

### ローカル環境で実行する場合

```bash
# リポジトリをクローン
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 依存パッケージのインストール（Ubuntu/Debianの場合）
sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8 fonts-ipafont
pip install mecab-python3 fugashi ipadic unidic-lite transformers torch scikit-learn matplotlib seaborn jupyter

# ノートブックを起動
jupyter notebook japanese_vectorization_comparison.ipynb
```

---

## 使用ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| [MeCab](https://taku910.github.io/mecab/) | 日本語形態素解析エンジン |
| [mecab-python3](https://github.com/SamuraiT/mecab-python3) | MeCabのPythonバインディング |
| [fugashi](https://github.com/polm/fugashi) | transformers向けMeCabラッパー |
| [ipadic](https://pypi.org/project/ipadic/) | MeCab用日本語辞書 |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF・PCA・コサイン類似度 |
| [transformers](https://huggingface.co/transformers/) | BERTモデルの読み込み・推論 |
| [PyTorch](https://pytorch.org/) | transformersのバックエンド |
| [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) | 可視化 |

### 使用モデル

- [`cl-tohoku/bert-base-japanese`](https://huggingface.co/cl-tohoku/bert-base-japanese) — 東北大学が公開している日本語BERTモデル

---

## ライセンス

MIT License
