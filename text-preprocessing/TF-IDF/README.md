# TF-IDF の仕組みと特徴を体験する

TF-IDFの仕組みを**実際の数値とグラフ**で確認するJupyter Notebookです。Google Colabですぐに実行できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/tfidf_features.ipynb)

---

## 目次

1. [TF-IDFとは](#tf-idfとは)
2. [トークン化とは](#トークン化とは)
3. [ベクトル化とは](#ベクトル化とは)
4. [ノートブックで確認できること](#ノートブックで確認できること)
5. [実行方法](#実行方法)
6. [使用ライブラリ](#使用ライブラリ)

---

## TF-IDFとは

**TF-IDF（Term Frequency - Inverse Document Frequency）** は、テキストをベクトルに変換する手法のひとつです。「その文書でよく出てきて、かつ他の文書にはあまり出てこない単語」を重要とみなします。

### TF（Term Frequency）：単語の出現頻度

その文書の中で、ある単語が何回出現するかを表します。

```
文書: "寿司は美味しい。寿司が好きだ。ラーメンも好きだ。"

TF("寿司")   = 2回 / 全7単語 = 0.29
TF("好き")   = 2回 / 全7単語 = 0.29
TF("ラーメン") = 1回 / 全7単語 = 0.14
```

### IDF（Inverse Document Frequency）：単語の珍しさ

全文書のうち何割の文書にその単語が出現するかの逆数です。どの文書にも出てくる単語（助詞など）はIDFが低くなります。

```
全文書数: 100件

"寿司"  → 10件に出現 → IDF = log(100/10) = 2.0  ← 珍しい → 重要
"好き"  → 50件に出現 → IDF = log(100/50) = 0.7
"です"  → 99件に出現 → IDF = log(100/99) = 0.01 ← ありふれている → 重要でない
```

### TF-IDF = TF × IDF

```
TF-IDF("寿司") = 0.29 × 2.0  = 0.58  ← 高い
TF-IDF("好き") = 0.29 × 0.7  = 0.20
TF-IDF("です") = 0.14 × 0.01 = 0.001 ← ほぼ0
```

### TF-IDFマトリクスの可視化

![TF-IDFマトリクス](https://raw.githubusercontent.com/QurhtDvf/llm/main/text-preprocessing/TF-IDF/download-3.png)

---

## トークン化とは

**トークン化（Tokenization）** とは、文章を機械が処理しやすい最小単位（トークン）に分割する処理です。

英語はスペースで単語が区切られているため自然に分割できますが、日本語はスペースがないため工夫が必要です。

```
英語: "I love sushi"   → ["I", "love", "sushi"]  # スペースで分割できる

日本語: "私は寿司が好きです" → どこで区切る？
```

### 日本語のトークン化：形態素解析（分かち書き）

MeCabなどのツールを使って文法的な単位に分割します。

```
"私は寿司が好きです" → ["私", "は", "寿司", "が", "好き", "です"]
```

TF-IDFでは、この分かち書きによって得られた単語を使ってベクトルを構築します。

---

## ベクトル化とは

**ベクトル化（Vectorization）** とは、テキストを数値の配列（ベクトル）に変換する処理です。機械学習モデルは文字列を直接扱えないため、数値への変換が必要です。

```
"寿司が好きです"  → [0.58, 0.00, 0.20, 0.00, ...]
"AIが発展している" → [0.00, 0.71, 0.00, 0.65, ...]
```

### TF-IDFベクトルの特徴

TF-IDFのベクトルは **スパース（疎）** です。語彙全体の中で実際に出現した単語の次元だけに値が入り、残りはすべて0になります。

```
語彙: ["寿司", "AI", "好き", "技術", "ラーメン", ...]  ← 数百〜数万次元

"寿司が好きです" → [0.58, 0.00, 0.20, 0.00, 0.00, ...]
                          ↑AI   ↑技術  ↑ラーメン（出現しないので0）
```

これに対してBERTなどのモデルが生成する **Dense（密）なベクトル** は、固定768次元でほぼすべての次元に値が入り、文の意味や文脈を表現します。

---

## ノートブックで確認できること

| セル | 内容 |
|------|------|
| 2. TFを確認 | 1文書内の単語出現頻度を手動計算 |
| 3. IDFを確認 | 複数文書でIDF値を手動計算し、ありふれた単語が低くなることを確認 |
| 4. TF × IDF | scikit-learnでTF-IDFマトリクスを生成・表示 |
| 5. 特徴①可視化 | ヒートマップで「ありふれた単語は低スコア」を視覚的に確認 |
| 6. 特徴②文脈無視 | 「好き」vs「嫌い」のコサイン類似度が高くなることを確認 |
| 7. 特徴③未知語 | 語彙リスト外の単語（「生成AI」「RAG」）がベクトルに反映されないことを確認 |
| 8. スパース性 | TF-IDFとBERTのベクトル密度を並べて可視化 |

### TF-IDFの特徴まとめ

| 特徴 | 内容 |
|------|------|
| ✅ ありふれた単語は低スコア | 「です」「ます」などはIDF値が低くなる |
| ✅ 珍しい単語は高スコア | 特定の文書にしか出ない単語が重要とみなされる |
| ⚠️ 文脈・感情が無視される | 「好き」と「嫌い」を区別できない |
| ⚠️ 未知語は扱えない | 語彙リスト外の単語はすべて0になる |
| ⚠️ スパースベクトル | ほとんどの次元が0になる |

---

## 実行方法

### Google Colabで実行する場合

1. 上の **Open in Colab** バッジをクリック
2. **セル0**（インストール）を実行
3. メニューから **「ランタイム → セッションを再起動」**
4. **セル1以降**を順番に実行（セル0の再実行は不要）

> 初回実行時は`apt-get`と`pip`のインストールに数分かかる場合があります。

### ローカル環境で実行する場合

```bash
# リポジトリをクローン
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 依存パッケージのインストール（Ubuntu/Debianの場合）
sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8 fonts-ipafont
pip install mecab-python3 fugashi ipadic scikit-learn matplotlib seaborn pandas jupyter

# ノートブックを起動
jupyter notebook tfidf_features.ipynb
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
| [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) | 可視化 |
| [pandas](https://pandas.pydata.org/) | データ整形・表示 |

---

## 関連ノートブック

日本語テキストのベクトル化手法をより広く比較したい場合は、こちらもご覧ください。

- [`japanese_vectorization_comparison.ipynb`](./japanese_vectorization_comparison.ipynb) — 従来手法（MeCab + TF-IDF）とBERTベース手法の比較

---

## ライセンス

MIT License
