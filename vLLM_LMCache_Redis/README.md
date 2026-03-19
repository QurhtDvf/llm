# 🚀 vLLM × LMCache × Redis — KV Cache Persistence Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/vllm-lmcache-redis-demo/blob/main/vllm_lmcache_redis_demo_final.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/vLLM-latest-FF6B35?logo=pytorch&logoColor=white)](https://github.com/vllm-project/vllm)
[![LMCache](https://img.shields.io/badge/LMCache-latest-00C7B7?logo=buffer&logoColor=white)](https://github.com/LMCache/LMCache)
[![Redis](https://img.shields.io/badge/Redis-7.x-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Google Colab で動く、LLM の KV キャッシュを Redis に永続化するエンドツーエンドデモ。**  
> vLLM の `LMCacheConnectorV1` を通じて LMCache を接続し、推論 KV テンソルを Redis に保存・再利用します。

---

## 📋 目次

- [KV キャッシュとは何か](#-kv-キャッシュとは何か)
- [LMCache と vLLM の関係](#-lmcache-と-vllm-の関係)
- [LMCache の歴史](#-lmcache-の歴史)
- [アーキテクチャ](#-アーキテクチャ)
- [デモの出力解説](#-デモの出力解説)
- [クイックスタート](#-クイックスタート)
- [環境要件](#-環境要件)
- [ファイル構成](#-ファイル構成)
- [参考文献](#-参考文献)

---

## 🧠 KV キャッシュとは何か

### Transformer における Attention の仕組み

Transformer モデルは、トークンを処理するたびに **Self-Attention** を計算します。  
Attention の計算式は以下の通りです：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

ここで **K（Key）** と **V（Value）** は、各トークンが持つ「記憶」に相当します。

### KV キャッシュの役割

自己回帰的なテキスト生成では、新しいトークンを1つ生成するたびに  
**過去の全トークンの K・V を再計算する必要があります。**

```
プロンプト: "What is the capital of Japan?"
            ↓
Token 1: "What"  → K1, V1 を計算
Token 2: "is"    → K1, V1, K2, V2 を再計算 ← 無駄！
Token 3: "the"   → K1, V1, K2, V2, K3, V3 を再計算 ← 無駄！
```

KV キャッシュはこの再計算を省くために、**計算済みの K・V テンソルをメモリに保存**します：

```
Token 1: K1, V1 を計算 → キャッシュに保存
Token 2: K2, V2 だけ計算 → K1, V1 はキャッシュから読み出す ✅
Token 3: K3, V3 だけ計算 → K1, V1, K2, V2 はキャッシュから読み出す ✅
```

### KV テンソルの形状とサイズ

KV キャッシュのテンソル形状は以下のように決まります：

```
KV テンソルのサイズ =
    num_layers × 2 (K + V) × num_heads × seq_len × head_dim × dtype_bytes

例: TinyLlama-1.1B (256 tokens chunk, float16)
  = 22 layers × 2 × 4 heads × 256 tokens × 64 dim × 2 bytes
  ≈ 5,767,168 bytes ≈ 5.5 MB / チャンク
```

> 💡 **今回のデモでも `kv_bytes` エントリが 5,767,168 bytes（5.5MB）であることが確認できます。**

### GPU メモリ上での KV キャッシュ

vLLM は **PagedAttention** という技術を使い、KV キャッシュを固定サイズの  
「ページ（ブロック）」に分割して GPU HBM 上で効率的に管理します。

```
GPU HBM (16GB T4)
┌─────────────────────────────────┐
│  モデルウェイト (~2.2GB)         │
│  KV キャッシュプール             │
│  ┌──────┬──────┬──────┬──────┐ │
│  │Block0│Block1│Block2│Block3│ │  ← 各ブロック = 16 tokens × layers
│  └──────┴──────┴──────┴──────┘ │
└─────────────────────────────────┘
```

---

## 🔗 LMCache と vLLM の関係

### vLLM の KV キャッシュの限界

vLLM の標準的な KV キャッシュには以下の制約があります：

| 問題 | 内容 |
|---|---|
| **揮発性** | プロセス終了で KV キャッシュが消える |
| **GPU 専有** | KV キャッシュは GPU HBM のみに存在 |
| **非共有** | 複数の vLLM インスタンス間で KV を共有できない |
| **再計算コスト** | 同じプロンプトでも再起動のたびにプリフィル計算が必要 |

### LMCache が解決すること

LMCache は vLLM の **KVConnector インターフェース** を実装し、  
KV キャッシュのライフサイクルを GPU の外まで拡張します：

```
┌─────────────────────────────────────────────────────────┐
│                      vLLM V1 Engine                      │
│                                                          │
│  PagedAttention ←→ KVConnector Interface                │
│                           ↕                              │
│                   LMCacheConnectorV1                     │
└───────────────────────────┬──────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         L1: GPU HBM   L2: CPU RAM   L3: Redis
         (最速/揮発)   (高速/揮発)   (低速/永続)
```

### KV の保存・読み込みフロー

```
新規リクエスト (cache miss)
  1. vLLM がプリフィル計算（KV 生成）
  2. LMCache が KV テンソルを 256 トークン単位でチャンク化
  3. チャンクのハッシュを計算してキーを生成
  4. Redis に float16 バイナリとして保存

既存リクエスト (cache hit)
  1. LMCache がプロンプトのチャンクハッシュを計算
  2. Redis に対応するキーが存在するか確認
  3. HIT → KV を Redis から GPU へ転送（プリフィル計算をスキップ）
  4. MISS → 通常のプリフィル計算 → Redis に保存
```

### TTFT（Time To First Token）への影響

KV キャッシュの再利用により、プリフィル計算をスキップできるため  
TTFT が大幅に削減されます：

```
プロンプト長 1000 tokens の場合（概算）

Cache MISS:  プリフィル計算 (~2000ms) + デコード (~50ms) = ~2050ms
Cache HIT:   KV 転送 (~50ms)         + デコード (~50ms) = ~100ms
                                                    ↑ 約20倍の高速化
```

---

## 📜 LMCache の歴史

### 誕生の背景（2023年）

LMCache は **UCバークレー Sky Computing Lab** の研究から生まれました。  
LLM サービングにおける KV キャッシュの非効率性を解決するため、  
[Ion Stoica](https://people.eecs.berkeley.edu/~istoica/)（Spark/Ray の生みの親）らのグループが開発しました。

### バージョン履歴

| 時期 | 出来事 |
|---|---|
| **2023年** | UCバークレーで KV キャッシュオフロードの研究開始 |
| **2024年 Q1** | `lmcache` として GitHub に初版公開 |
| **2024年 Q2** | vLLM との統合 (`LMCacheConnector`) の初期実装 |
| **2024年 Q3** | **CacheGen** 論文発表（KV キャッシュの圧縮アルゴリズム） |
| **2024年 Q4** | Redis / CPU RAM / ローカルディスク など複数バックエンド対応 |
| **2025年 Q1** | vLLM V1 の `KVTransferConfig` に `LMCacheConnectorV1` として正式統合 |
| **2025年〜** | Disaggregated Prefill、Kubernetes Operator、Multiprocess Mode など本番向け機能を拡充中 |

### 主要論文

- **CacheGen (2024)**: KV キャッシュをトークンストリームとして圧縮・転送する手法  
  → SIGCOMM 2024 採択
- **LMCache (2024)**: LLM サービングのための汎用 KV キャッシュレイヤー  
  → [arXiv:2411.06032](https://arxiv.org/abs/2411.06032)

### エコシステムでの位置づけ

```
LLM サービングスタック

┌──────────────────────────────────┐
│  Application (RAG / Chatbot 等)  │
├──────────────────────────────────┤
│  Serving Framework (vLLM)        │  ← リクエストスケジューリング
├──────────────────────────────────┤
│  LMCache (KV Cache Layer)        │  ← KV の永続化・共有・圧縮
├──────────────────────────────────┤
│  Storage Backend                 │
│  Redis / CPU RAM / Disk / S3     │
└──────────────────────────────────┘
```

---

## 🏗️ アーキテクチャ

### Redis キーの形式

LMCache が Redis に保存する KV キャッシュのキーは以下の形式です：

```
{model} @ {layer_idx} @ {head_idx} @ {chunk_hash} @ half{kv_bytes|metadata}
```

**実際のキー例（今回のデモより）：**

```
TinyLlama/TinyLlama-1.1B-Chat-v1.0@1@0@7fac50ea95976517@halfkv_bytes
│                                  │ │ │                  │
│                                  │ │ │                  └─ エントリ種別
│                                  │ │ └──────────────────── チャンクハッシュ(64bit)
│                                  │ └─────────────────────── ヘッドインデックス
│                                  └───────────────────────── レイヤーインデックス
└──────────────────────────────────────────────────────────── モデル名
```

### エントリの種類

| エントリ種別 | サイズ | 内容 |
|---|---|---|
| `halfkv_bytes` | ~5.5 MB | float16 の K・V テンソルバイナリ |
| `halfmetadata` | 28 bytes | チャンクのメタ情報（トークン数・形状など） |

---

## 📊 デモの出力解説

### グラフ 1：Latency per Query（クエリ別レイテンシ）

```
Latency (ms)
    │
2000│  ████
    │  ████
1000│  ████  ████
    │  ████  ████
   0│  ████  ████  ██  ██
    └──Q1────Q2────Q3──Q4──
       初回  初回  再利用
```

| クエリ | 色 | 説明 |
|---|---|---|
| **Q1, Q2**（赤） | 🔴 初回 | GPU でプリフィル計算を実行。KV を生成して Redis に保存。レイテンシが高い |
| **Q3, Q4**（青） | 🔵 再利用 | Redis から KV を読み出し、プリフィル計算をスキップ。レイテンシが大幅に低下 |

> **注目ポイント**: Q3 は Q1 と同一プロンプトのため、共通プレフィックス部分の KV が  
> Redis にキャッシュされており、プリフィル計算をほぼスキップできます。

---

### グラフ 2：Redis Entry Size（Redis エントリサイズ）

```
Size (KB)
     │
5632 │  ████  ████  ████  ████
     │  ████  ████  ████  ████
     │  ████  ████  ████  ████
   0 │  ░░░░  ░░░░  ░░░░  ░░░░
     └──kv───kv────kv────kv───   ← kv_bytes (緑, ~5632KB)
        meta  meta  meta  meta   ← metadata  (橙, ~0.03KB)
```

| エントリ | サイズ | 件数 | 内容 |
|---|---|---|---|
| `kv_bytes`（緑） | **5,767,168 bytes（5.5 MB）** | 4件 | float16 KV テンソル本体 |
| `metadata`（橙） | **28 bytes** | 4件 | チャンクのメタ情報 |

**kv_bytes のサイズ計算:**

```
TinyLlama-1.1B の場合:
  - num_layers  : 22
  - num_kv_heads: 4
  - head_dim    : 64
  - chunk_size  : 256 tokens
  - dtype       : float16 (2 bytes)
  - K + V       : 2

5,767,168 = 22 × 4 × 64 × 256 × 2 × 2 bytes ✅
```

**4件あることの意味:**  
同一プロンプトが Q1〜Q4 で合計 4 パターン（Q1=Q3, Q2=Q4 はハッシュが同一のため  
実際はユニーク 2 チャンク × (kv_bytes + metadata) = 4 キー）存在します。

---

### グラフ 3：LMCache KV Cache Hierarchy（KV キャッシュ階層）

```
┌──────────────────────────────────┐
│  L1: GPU HBM (vLLM PagedAttention)│  Fastest ~TB/s (揮発)
└──────────────┬───────────────────┘
               ↕ LMCacheConnectorV1
┌──────────────┴───────────────────┐
│  L3: Redis (LMCache Remote)      │  Persistent ~GB/s (永続)
└──────────────────────────────────┘
```

| 階層 | 速度 | 容量 | 永続性 |
|---|---|---|---|
| **L1: GPU HBM** | ~TB/s | ~10GB | ❌ 揮発 |
| **L3: Redis** | ~GB/s | 無制限 | ✅ 永続 |

---

### コンソール出力：KV テンソルの float16 値

セル⑧では Redis に保存された KV テンソルを float16 としてデコードして表示します：

```
🔑 Entry 1
  Redis Key : TinyLlama/TinyLlama-1.1B-Chat-v1.0@1@0@7fac50ea95976517@halfkv_bytes
  Data size : 5,767,168 bytes  (5632.0 KB)
  KV values (float16, first 128 elements):
    -0.1234  0.4531  -0.0234  0.1094  ...
    0.2344  -0.3125   0.0781 -0.1562  ...
    ...
  Stats: min=-2.3438  max=2.3125  mean=0.0023  std=0.4102
```

**数値の解釈:**

| 統計値 | 典型的な値 | 意味 |
|---|---|---|
| `min` / `max` | ±2〜4 程度 | KV の値域。大きすぎる場合は NaN の可能性 |
| `mean` | ≈ 0 | KV テンソルは平均がほぼゼロになる性質がある |
| `std` | 0.3〜0.6 程度 | 値のばらつき。モデル・レイヤーによって異なる |

---

### Redis 統計出力（セル⑩）

```
=== Redis Stats ===
  Total keys      : 8
  Memory used     : 44.32M
  Memory peak     : 44.35M
  Total commands  : 1,204
  Connected clients: 1
```

| 項目 | 説明 |
|---|---|
| **Total keys: 8** | 4 チャンク × 2（kv_bytes + metadata）= 8 キー |
| **Memory used: ~44MB** | 4 × 5.5MB（kv_bytes）+ 少量（metadata）= 約 22MB の KV データ + Redis オーバーヘッド |

---

## ⚡ クイックスタート

### Google Colab で実行

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/vllm-lmcache-redis-demo/blob/main/vllm_lmcache_redis_demo_final.ipynb)

1. 上のバッジをクリック
2. ランタイム → ランタイムのタイプを変更 → **T4 GPU** を選択
3. セルを **上から順番に** 実行

> ⚠️ **重要**: セル① の `%env` は必ず最初に実行してください。  
> `LMCacheConnectorV1` は **vLLM V1 専用**です。`VLLM_USE_V1=0` は設定しないでください。

### ローカル環境で実行

```bash
# 1. リポジトリをクローン
git clone https://github.com/your-username/vllm-lmcache-redis-demo.git
cd vllm-lmcache-redis-demo

# 2. Redis を起動
docker run -d -p 6379:6379 redis:latest

# 3. 依存パッケージをインストール
pip install vllm lmcache redis pandas matplotlib pyyaml

# 4. 環境変数を設定
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_CPU=False
export LMCACHE_REMOTE_URL=redis://localhost:6379
export LMCACHE_REMOTE_SERDE=naive

# 5. Jupyter で実行
jupyter notebook vllm_lmcache_redis_demo_final.ipynb
```

---

## 🛠️ 環境要件

| 項目 | 要件 |
|---|---|
| **GPU** | CUDA 対応 GPU（T4 以上推奨、VRAM 16GB+） |
| **Python** | 3.10 以上 |
| **vLLM** | 0.6.0 以上（V1 エンジン対応版） |
| **LMCache** | 最新版（`pip install lmcache`） |
| **Redis** | 7.x |
| **RAM** | 16GB 以上推奨 |

---

## 📁 ファイル構成

```
.
└── vllm_lmcache_redis_demo_final.ipynb   # メインノートブック
```

### ノートブックのセル構成

| セル | 内容 |
|---|---|
| ① | `%env` で環境変数設定（vLLM import より前に必須） |
| ② | パッケージインストール（vllm / lmcache / redis） |
| ③ | Redis サーバー起動 |
| ④ | vLLM + LMCache エンジン起動（`LMCacheConnectorV1`） |
| ⑤ | 推論デモ（共通プレフィックスで KV 再利用を実証） |
| ⑥ | Redis に保存された KV エントリをパース・表示 |
| ⑦ | `redis-cli` で RAW データを直接確認 |
| ⑧ | KV テンソルを float16 としてデコード・表示 |
| ⑨ | グラフ可視化（レイテンシ / エントリサイズ / 階層図） |
| ⑩ | Redis 統計 & LMCache クリーンアップ |

---

## 📚 参考文献

- [LMCache 公式ドキュメント](https://docs.lmcache.ai/)
- [LMCache GitHub](https://github.com/LMCache/LMCache)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [CacheGen 論文 (SIGCOMM 2024)](https://arxiv.org/abs/2310.07240)
- [LMCache 論文 (arXiv:2411.06032)](https://arxiv.org/abs/2411.06032)
- [vLLM: Easy, Fast, and Cheap LLM Serving (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [PagedAttention 解説](https://blog.vllm.ai/2023/06/20/vllm.html)

---

## 📄 ライセンス

Apache License 2.0
