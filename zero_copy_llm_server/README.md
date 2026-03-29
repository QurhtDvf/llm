# Zero-Copy LLM Inference Server

CPU上でローカル LLM（7–8B パラメータ）を動かしながら、**プロセス間通信（IPC）の方式が推論サーバーのレイテンシとスループットにどう影響するか**を計測・比較するプロトタイプです。Google Colab（CPU ランタイム）および AWS EC2 で動作します。

---

## 何を計測しているか

「クライアントがプロンプトを送ってから推論結果を受け取るまでの往復時間（E2E レイテンシ）」を 3 つの異なる IPC 方式で比較します。

```
クライアント ──[通信方式]──▶ 推論サーバー (LLM)
              ◀──────────── 生成テキスト
```

計測指標は以下の 3 つです。

| 指標 | 意味 |
|------|------|
| **中央値レイテンシ (ms)** | リクエストの典型的な応答時間 |
| **P99 レイテンシ (ms)** | 99 パーセンタイル、外れ値を含む最悪ケース近似 |
| **QPS (req/s)** | 1 秒あたりに処理できるリクエスト数 |

---

## 3 つの通信方式の詳細

### Benchmark 1 — Baseline: `pickle` + TCP loopback

```
Client                          Server process
  │                                  │
  │  pickle.dumps(prompt)            │
  │──── TCP 127.0.0.1:port ─────────▶│
  │     (user→kernel→NIC buf         │
  │      →kernel→user = 4 copies)    │
  │                             LLM inference
  │                             pickle.dumps(result)
  │◀─── TCP ─────────────────────────│
```

**特徴と問題点:**

- Python 標準の `pickle` でプロンプトをシリアライズし、TCP ループバックで送受信する最もシンプルな実装です。
- データは「ユーザー空間 → カーネル（TCP スタック）→ カーネル → ユーザー空間」と **4 回コピー**されます。
- **最大の問題はモデルのロードコスト**です。このベンチマークではサーバープロセスがリクエストのたびに新しく起動し、4.5GB のモデルを毎回ゼロから読み込みます。Colab CPU では 1 回あたり **30〜60 秒**かかります。
- 結果として、20 リクエストで **20〜30 分以上**かかる場合があります。

**実測値の目安（Colab 無料 CPU）:**

| 指標 | 値 |
|------|-----|
| 中央値レイテンシ | ~9,000 ms |
| 主なボトルネック | モデルロード (~30s) + 推論 (~5s) |

---

### Benchmark 2 — Shared memory + zero-copy (`memoryview`)

```
Client process          Shared RAM (物理メモリ共有)       Server process
  │                    ┌────────────────────────┐              │
  │  mv_in[0:4] = len  │  [4B len][prompt bytes]│              │
  │  mv_in[4:] = prompt│  [4B len][result bytes]│              │
  │  ctrl[0] = 1 ──────┼──────────────────────────── flag=1 ──▶│
  │                    │                        │         LLM inference
  │  ctrl[0]==2 ? ◀────┼──────────────────────────── flag=2 ──│
  │  result = mv_out[] │                        │              │
  └────────────────────┴────────────────────────┘──────────────┘
                            コピー = 0 回
```

**特徴と仕組み:**

- `multiprocessing.shared_memory` で OS レベルの共有メモリ領域を確保します。クライアントとサーバーは**同じ物理メモリアドレスを直接読み書き**するため、データのコピーが発生しません（ゼロコピー）。
- `memoryview` を使うことで、Python レベルでも中間バッファを生成せずにスライスアクセスが可能です。
- 制御フラグ（1 バイト）でリクエスト/レスポンスのタイミングを同期します（スピンポーリング）。
- **モデルは依然としてサーバープロセス起動のたびにロードされます**。そのため Benchmark 1 との差はほぼ通信コストのみであり、CPU 7B の場合は数 ms 程度の差です。

**実測値の目安（Colab 無料 CPU）:**

| 指標 | 値 |
|------|-----|
| 中央値レイテンシ | ~5,000 ms |
| TCP との差 | 数 ms（推論が支配的のため小さい） |

> **補足:** セル終了時に `BufferError: cannot close exported pointers exist` が表示されることがありますが、Python の GC 順序の問題であり動作には影響しません。

---

### Benchmark 3 — Persistent server + UNIX domain socket

```
Client                UNIX socket (/tmp/llm_persist.sock)     Server process
  │                                                           ┌──────────────┐
  │  (初回のみ)                                               │ モデルロード  │
  │                                                           │ (1 回だけ)   │
  │──── prompt UTF-8 ─────────────────────────────────────▶  │              │
  │  (user→UNIX kernel buf→user = 2 copies,                  │ LLM inference│
  │   ネットワークスタック不使用)                              │              │
  │◀─── result UTF-8 ──────────────────────────────────────  │              │
  │                                                           │ 次のリクエスト待ち│
  │──── prompt UTF-8 ─────────────────────────────────────▶  │              │
  │◀─── result UTF-8 ──────────────────────────────────────  └──────────────┘
```

**特徴と仕組み:**

- モデルを **1 度だけロードして常駐**させた長期プロセスにリクエストを送ります。これが 3 つの方式のなかで最も重要な最適化です。
- UNIX ドメインソケットは同一ホスト上の TCP より低レイテンシです。TCP はネットワークスタック（IP ヘッダの付与、チェックサム計算など）を通りますが、UNIX ソケットはカーネル内バッファの直接コピーで済みます。
- モデルロードのコスト（30〜60 秒）がリクエストレイテンシから消えるため、**純粋な推論時間のみ**が計測されます。
- プロダクション環境（本番サービス）の LLM サーバーはすべてこの「常駐モデル」方式です。

**実測値の目安（Colab 無料 CPU）:**

| 指標 | 値 |
|------|-----|
| 中央値レイテンシ | ~4,000〜5,000 ms |
| モデルロードコスト | 0（初回のみ） |

---

## 計測結果の解釈

### CPU 7B 環境でわかること

```
1 リクエストの内訳（Benchmark 1）:
  モデルロード  : ~30,000 ms  ████████████████████████████████ 85%
  LLM 推論     :  ~5,000 ms  █████                             13%
  TCP 通信     :      ~3 ms  ▏                                  0.01%
  pickle 処理  :      ~1 ms  ▏                                  0.003%
```

CPU 7B では**モデルロードと推論がレイテンシの 99.9% 以上を占めます**。これが「CPU 単体では通信最適化の費用対効果が薄い」理由です。

### 通信最適化が重要になる条件

| 条件 | 通信コストの割合 | ゼロコピーの効果 |
|------|----------------|----------------|
| CPU 7B、逐次処理 | < 0.1% | 薄い |
| GPU (A100) + 小型モデル | 5〜20% | 中程度 |
| マルチエージェント (N 並列) | N 倍に積算 | **重要** |
| ストリーミング (token-by-token) | トークンごとに発生 | **重要** |
| 高頻度 API (1000+ req/s) | 積み重なる | **重要** |

---

## セットアップ

### Google Colab

1. ランタイム → ランタイムのタイプを変更 → **CPU** を選択
2. ノートブックを上から順に実行

### ローカル環境

```bash
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install huggingface_hub tqdm msgpack matplotlib
```

### AWS EC2

推奨インスタンス: `c5.2xlarge`（8 vCPU, 16GB RAM）以上

```bash
# 依存パッケージ
sudo apt-get update && sudo apt-get install -y python3-pip fonts-noto-cjk
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install huggingface_hub tqdm msgpack matplotlib

# モデルのダウンロード先を instance store (NVMe) にするとロードが速い
# EBS (~60s) vs NVMe (~10s) for 4.5GB model
```

GPU インスタンス（p3, g4dn など）を使う場合は以下でビルドし直してください。

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python \
    --upgrade --force-reinstall --no-cache-dir
```

---

## ファイル構成

```
.
├── zero_copy_llm_server.ipynb   # メインノートブック
├── README.md                    # このファイル
└── models/                      # ダウンロードされたモデル (gitignore 推奨)
    └── Qwen2.5-7B-Instruct-Q4_K_M.gguf
```

`.gitignore` に以下を追加することを推奨します。

```
models/
*.gguf
llm_benchmark.png
```

---

## 使用モデル

| 項目 | 内容 |
|------|------|
| モデル | Qwen2.5-7B-Instruct |
| 量子化 | Q4_K_M（4bit、精度と速度のバランス） |
| 配布元 | [bartowski/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF) |
| ファイルサイズ | ~4.5 GB |
| 必要 RAM | 最低 6 GB（OS 込みで 8 GB 以上推奨） |

> **注意:** Qwen 公式リポジトリ (`Qwen/Qwen2.5-7B-Instruct-GGUF`) の Q4_K_M は 2 つのシャードに分割されており単一ファイルが存在しないため 404 エラーになります。bartowski 版を使用してください。

---

## 技術的な詳細

### ゼロコピーの仕組み

```python
# 通常のコピーあり
data = bytes(shm.buf[:N])            # N バイトコピーが発生

# ゼロコピー (memoryview)
arr = memoryview(shm.buf)[:N]        # コピーなし、同じメモリを参照
```

### 共有メモリのレイアウト

```
オフセット 0       : [4 bytes] プロンプトのバイト長
オフセット 4       : [最大 2048 bytes] プロンプト UTF-8
オフセット 2052    : [4 bytes] 結果のバイト長
オフセット 2056    : [最大 4096 bytes] 結果 UTF-8

制御フラグ (別 SharedMemory, 1 byte):
  0 = idle
  1 = request_ready  (クライアントが書き込み完了)
  2 = response_ready (サーバーが書き込み完了)
```

### UNIX ソケット vs TCP の違い

| 項目 | TCP loopback | UNIX domain socket |
|------|-------------|-------------------|
| ネットワークスタック | 通る（IP/TCP ヘッダ処理あり） | 通らない |
| データコピー | 4 回 | 2 回 |
| レイテンシ | ~0.1〜3 ms | ~0.01〜0.5 ms |
| 用途 | ネットワーク越し | 同一ホスト内のみ |

---

## ライセンス

MIT
