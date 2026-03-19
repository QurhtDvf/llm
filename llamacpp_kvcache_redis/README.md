# 🦙 llama.cpp-kvcache-redis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Redis](https://img.shields.io/badge/Redis-7.x-DC382D?logo=redis&logoColor=white)](https://redis.io/)

> **llama.cpp のフォーク。CPU サーバー上で動く小型 LLM の KV キャッシュを  
> Redis に永続化し、複数エージェントプロセス間で共有するための実装。**

---

## 📋 目次

- [なぜこのフォークを作ったか](#-なぜこのフォークを作ったか)
- [LMCache 互換を目指した理由](#-lmcache-互換を目指した理由)
- [アーキテクチャ](#-アーキテクチャ)
- [ベンチマーク結果](#-ベンチマーク結果)
- [改造内容](#-改造内容)
- [ビルド方法](#-ビルド方法)
- [Python からの使い方](#-python-からの使い方)
- [Redis キー形式](#-redis-キー形式)
- [設計上の注意点](#-設計上の注意点)

---

## 🎯 なぜこのフォークを作ったか

### 背景：AIエージェントの KV キャッシュ問題

LLM を使った AIエージェントは、毎回のリクエストで以下のような構造のプロンプトを送ります：

```
[システムプロンプト: 固定・数百〜数千トークン ]  ← 毎回同じ
[ツール定義       : 固定・数百トークン       ]  ← 毎回同じ
[会話履歴         : 毎回変わる               ]
[今回の質問       : 毎回変わる               ]
```

**問題**: 固定部分のプリフィル計算が毎回発生し、CPU 環境では数秒〜数十秒かかる。

```
Agent A → llama.cpp → システムプロンプトを毎回計算 (17秒)
Agent B → llama.cpp → システムプロンプトを毎回計算 (17秒)
Agent C → llama.cpp → システムプロンプトを毎回計算 (17秒)
```

**解決**: 固定部分の KV キャッシュを Redis に保存し、2回目以降は復元する。

```
Agent A → llama.cpp-kvcache-redis → Redis HIT → KV 復元 (1秒)
Agent B → llama.cpp-kvcache-redis → Redis HIT → KV 復元 (1秒)
Agent C → llama.cpp-kvcache-redis → Redis HIT → KV 復元 (1秒)
```

### vLLM + LMCache では解決できない

既存の [LMCache](https://github.com/LMCache/LMCache) は vLLM 専用であり、
CPU サーバー・小型モデル・GGUF 量子化を主戦場とする llama.cpp には対応していない。

| | vLLM + LMCache | **本フォーク** |
|---|---|---|
| CPU サーバー | ❌ GPU 必須 | ✅ CPU only |
| 小型モデル (1〜7B) | ❌ オーバースペック | ✅ 最適 |
| GGUF 量子化 | △ 限定的 | ✅ ネイティブ |
| 複数プロセス KV 共有 | ✅ | ✅ |
| Redis 永続化 | ✅ | ✅ |

---

## 🔗 LMCache 互換を目指した理由

### LMCache の設計思想

LMCache（vLLM 用）は KV キャッシュを以下の粒度で管理します：

```
Redis キー形式:
  {model}@{layer}@{head}@{chunk_hash}@halfkv_bytes
  {model}@{layer}@{head}@{chunk_hash}@halfmetadata

チャンク単位 : 256 トークン
ハッシュ     : SHA256(token_ids) の先頭16文字
```

この設計の優れている点：

1. **チャンク単位の部分一致** — 共通プレフィックスだけを再利用できる
2. **トークンハッシュによるキー** — プロンプトが同一かどうかを高速判定
3. **Redis による永続化** — プロセス再起動後も KV を再利用できる

### 本フォークが目指した互換性

本フォークは LMCache と **完全に同じ思想** でキーを設計しています：

```
LMCache (vLLM):
  model@layer@head@{sha256[:16]}@halfkv_bytes

本フォーク (llama.cpp):
  model@{sha256[:16]}@chunk
```

将来的に vLLM と llama.cpp が **同じ Redis を共有** できることを視野に入れています。

### llama.cpp への改造が最小限で済んだ理由

llama.cpp のソースコード解析の結果、既存の `cell_ranges_t` 構造体が
すでに「任意範囲の読み書き」を抽象化していることが判明しました：

```cpp
// 既存コード（llama-kv-cache.cpp）
for (const auto & range : cr.data) {
    io.write_tensor(k, range.first * k_size_row, buf_size);
}
```

`state_write_chunk` は `cell_ranges_t` を外部から指定するだけで実装できます。
**既存の `state_write_meta` / `state_write_data` は一切変更不要** でした。

---

## 🏗️ アーキテクチャ

```
Agent Process A ─┐
Agent Process B ─┼──→ [subprocess] llmcache_demo.py
Agent Process C ─┘         │
                            │  ctypes
                            ▼
                   libllama.so (フォーク版)
                   ├── llmcache_kv_write_chunk()  ← 新規 C++ API
                   ├── llmcache_kv_read_chunk()
                   ├── llmcache_kv_chunk_size()
                   └── llmcache_kv_clear()
                            │
                            ▼
                        Redis
                   {model}@{hash16}@chunk
```

### なぜ subprocess 経由で呼ぶか

`llama-cpp-python` は独自の `libllama.so` をプロセス起動時にロードします。
同プロセスでフォーク版の `.so` を `ctypes.CDLL` で追加ロードしても、
Linux の `dlopen` は **同名シンボルを先勝ちで解決** するため
フォーク版の新規シンボルに到達できません。

**解決策**: 全処理を `subprocess` で別プロセスとして実行します。
別プロセスならフォーク版 `.so` だけをロードできます。

---

## 📊 ベンチマーク結果

Google Colab（CPU モード）での測定値：

| 項目 | 値 |
|---|---|
| モデル | TinyLlama-1.1B-Chat Q4_K_M (636 MB) |
| トークン数 | 402 tokens |
| チャンク数 | 1 chunk (256 tokens/chunk) |
| **プリフィル時間（初回）** | **17,559 ms** |
| **KV 復元時間（2回目以降）** | **1,028 ms** |
| **スピードアップ** | **17.1x** |
| KV チャンクサイズ | 5,770,776 bytes (5.5 MB) |

---

## 🔧 改造内容

### 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `src/llama-kv-cache.h` | 変更 | 3メソッドの宣言を追加 |
| `src/llama-kv-cache.cpp` | 変更 | 3メソッドの実装を追加（+30行） |
| `src/llama-kv-chunk-api.cpp` | **新規** | `llmcache_kv_*` 公開 API |
| `src/llmcache-wrapper.cpp` | **新規** | モデルロード・推論・KV クリアのラッパー |
| `src/CMakeLists.txt` | 変更 | 新規ファイルをビルドに追加 |
| `include/llama.h` | 変更 | `llmcache_kv_*` の宣言を追加 |

### 追加した C++ API

```cpp
// ---- llama-kv-cache.h に追加（クラスメンバ）----

// tokens[token_start, token_start+token_count) の KV をシリアライズして io に書き出す
void state_write_chunk(
    llama_io_write_i & io,
    uint32_t token_start,
    uint32_t token_count) const;

// io から KV をデシリアライズして tokens[token_start, ...) に注入する
bool state_read_chunk(
    llama_io_read_i & io,
    uint32_t token_start,
    uint32_t token_count);

// token_count トークン分のシリアライズサイズを返す（バッファ確保用）
size_t get_chunk_size_bytes(uint32_t token_count) const;


// ---- include/llama.h に追加（公開 API、extern "C"）----

// KV チャンクを dst バッファに書き出す
bool llmcache_kv_write_chunk(
    struct llama_context * ctx,
    void                 * dst,
    size_t               * dst_size,
    uint32_t               token_start,
    uint32_t               token_count);

// src バッファから KV チャンクを復元する
bool llmcache_kv_read_chunk(
    struct llama_context * ctx,
    const void           * src,
    size_t                 src_size,
    uint32_t               token_start,
    uint32_t               token_count);

// KV シリアライズに必要なバッファサイズを返す
size_t llmcache_kv_chunk_size(
    struct llama_context * ctx,
    uint32_t               token_count);
```

---

## 🔨 ビルド方法

### 必要なもの

- Linux (Ubuntu 22.04+)
- GCC 12+ / Clang 15+
- CMake 3.21+
- Redis 7.x

### 手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# 2. パッチを適用（このリポジトリの fork.patch を使用）
git apply /path/to/fork.patch

# 3. ビルド（共有ライブラリとして）
cmake -B build \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4

# 4. ビルド確認（llmcache_ シンボルが存在するか）
nm --demangle -D build/bin/libllama.so | grep llmcache_
```

期待される出力：

```
T llmcache_kv_chunk_size
T llmcache_kv_read_chunk
T llmcache_kv_write_chunk
T llmcache_kv_clear
T llmcache_decode_tokens
T llmcache_load_model
T llmcache_new_context
T llmcache_tokenize
```

---

## 🐍 Python からの使い方

### 重要：なぜ subprocess で実行するか

```python
# ❌ これは動かない（llama-cpp-python との .so 競合）
import ctypes
lib = ctypes.CDLL('/path/to/forked/libllama.so')
# → llama-cpp-python がロードした libllama.so が先勝ちするため
#   llmcache_* シンボルに到達できない

# ✅ これが正解（別プロセスで実行）
import subprocess
r = subprocess.run(
    ['python3', 'llmcache_demo.py'],
    env={'LD_LIBRARY_PATH': '/path/to/build/bin'}
)
```

### Step 1: ctypes シグネチャの定義

フォーク版を使うスクリプトの先頭で以下を定義します。

```python
import ctypes, subprocess

BUILD = '/path/to/llama.cpp/build'
SO    = f'{BUILD}/bin/libllama.so'

# ggml 系を先にグローバルロード
for p in sorted(subprocess.run(
        ['find', BUILD, '-name', 'libggml*.so'],
        capture_output=True, text=True).stdout.strip().split('\n')):
    if p:
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)

lib = ctypes.CDLL(SO, mode=ctypes.RTLD_GLOBAL)

# モデル・コンテキスト管理
lib.llama_backend_init.restype  = None
lib.llama_backend_init.argtypes = []

lib.llmcache_load_model.restype  = ctypes.c_void_p
lib.llmcache_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int]

lib.llmcache_new_context.restype  = ctypes.c_void_p
lib.llmcache_new_context.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

lib.llmcache_free_context.restype  = None
lib.llmcache_free_context.argtypes = [ctypes.c_void_p]

lib.llmcache_free_model.restype  = None
lib.llmcache_free_model.argtypes = [ctypes.c_void_p]

# トークナイズ・デコード
lib.llmcache_tokenize.restype  = ctypes.c_int32
lib.llmcache_tokenize.argtypes = [
    ctypes.c_void_p,   # model
    ctypes.c_char_p,   # text
    ctypes.c_int32,    # text_len  ← -1 は不可。len(text.encode()) を渡すこと
    ctypes.c_void_p,   # out_tokens (int32 配列)
    ctypes.c_int32,    # max_tokens
]

lib.llmcache_decode_tokens.restype  = ctypes.c_int
lib.llmcache_decode_tokens.argtypes = [
    ctypes.c_void_p,   # ctx
    ctypes.c_void_p,   # tokens (int32 配列)
    ctypes.c_int32,    # n_tokens
]

# KV キャッシュ操作
lib.llmcache_kv_clear.restype  = None
lib.llmcache_kv_clear.argtypes = [ctypes.c_void_p]

lib.llmcache_kv_write_chunk.restype  = ctypes.c_bool
lib.llmcache_kv_write_chunk.argtypes = [
    ctypes.c_void_p,                      # ctx
    ctypes.c_void_p,                      # dst バッファ
    ctypes.POINTER(ctypes.c_size_t),      # dst_size (in/out)
    ctypes.c_uint32,                      # token_start
    ctypes.c_uint32,                      # token_count
]

lib.llmcache_kv_read_chunk.restype  = ctypes.c_bool
lib.llmcache_kv_read_chunk.argtypes = [
    ctypes.c_void_p,   # ctx
    ctypes.c_void_p,   # src バッファ
    ctypes.c_size_t,   # src_size
    ctypes.c_uint32,   # token_start
    ctypes.c_uint32,   # token_count
]

lib.llmcache_kv_chunk_size.restype  = ctypes.c_size_t
lib.llmcache_kv_chunk_size.argtypes = [
    ctypes.c_void_p,   # ctx
    ctypes.c_uint32,   # token_count
]
```

### Step 2: モデルのロードとトークナイズ

```python
lib.llama_backend_init()

# n_gpu_layers=0 → CPU のみ（GPU がある場合は正の値を指定）
model = lib.llmcache_load_model(b'/path/to/model.gguf', 0)
ctx   = lib.llmcache_new_context(model, ctypes.c_uint32(2048))

# トークナイズ
# ⚠️ text_len は必ず len(prompt_enc) を渡すこと（-1 は不可）
PROMPT     = 'You are a helpful AI assistant. ' * 50
prompt_enc = PROMPT.encode()
MAX_TOKENS = 2048
token_arr  = (ctypes.c_int32 * MAX_TOKENS)()
n_tokens   = lib.llmcache_tokenize(
    ctypes.c_void_p(model),
    prompt_enc,
    ctypes.c_int32(len(prompt_enc)),
    ctypes.cast(token_arr, ctypes.c_void_p),
    ctypes.c_int32(MAX_TOKENS)
)
tokens = list(token_arr[:n_tokens])
print(f'トークン数: {n_tokens}')
```

### Step 3: KV キャッシュの保存（初回）

```python
import hashlib, numpy as np, redis

r_client   = redis.Redis(host='localhost', port=6379, decode_responses=False)
CHUNK_SIZE = 256
MODEL_NAME = 'MyModel'

def chunk_hash(tokens: list, start: int, size: int = 256) -> str:
    """LMCache 互換のチャンクハッシュを計算する"""
    chunk = tokens[start : start + size]
    buf   = np.array(chunk, dtype=np.int32).tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]

# プリフィル計算
ret = lib.llmcache_decode_tokens(
    ctypes.c_void_p(ctx),
    ctypes.cast(token_arr, ctypes.c_void_p),
    ctypes.c_int32(n_tokens)
)

# 256トークン単位で KV を Redis に保存
n_chunks = n_tokens // CHUNK_SIZE
for ci in range(n_chunks):
    start    = ci * CHUNK_SIZE
    h        = chunk_hash(tokens, start)
    buf_size = lib.llmcache_kv_chunk_size(
        ctypes.c_void_p(ctx), ctypes.c_uint32(CHUNK_SIZE))
    buf      = (ctypes.c_uint8 * buf_size)()
    out_size = ctypes.c_size_t(buf_size)

    ok = lib.llmcache_kv_write_chunk(
        ctypes.c_void_p(ctx),
        ctypes.cast(buf, ctypes.c_void_p),
        ctypes.byref(out_size),
        ctypes.c_uint32(start),
        ctypes.c_uint32(CHUNK_SIZE)
    )
    if ok:
        key = f'{MODEL_NAME}@{h}@chunk'
        r_client.setex(key, 3600, bytes(buf[:out_size.value]))
        print(f'  保存: chunk_{ci}  hash={h}  size={out_size.value//1024}KB')
```

### Step 4: KV キャッシュの復元（2回目以降）

```python
# KV をクリアして Redis から復元する
lib.llmcache_kv_clear(ctypes.c_void_p(ctx))

restored = 0
for ci in range(n_chunks):
    start = ci * CHUNK_SIZE
    h     = chunk_hash(tokens, start)
    key   = f'{MODEL_NAME}@{h}@chunk'
    data  = r_client.get(key)

    if data is None:
        print(f'  MISS: chunk_{ci}')
        break  # 連続 HIT が途切れたら停止

    buf = (ctypes.c_uint8 * len(data))(*data)
    ok  = lib.llmcache_kv_read_chunk(
        ctypes.c_void_p(ctx),
        ctypes.cast(buf, ctypes.c_void_p),
        ctypes.c_size_t(len(data)),
        ctypes.c_uint32(start),
        ctypes.c_uint32(CHUNK_SIZE)
    )
    if ok:
        restored += CHUNK_SIZE
        print(f'  HIT:  chunk_{ci}  ({restored} tokens 復元済み)')
    else:
        print(f'  FAIL: chunk_{ci}')
        break

print(f'復元完了: {restored} / {n_tokens} tokens')
```

### Step 5: エージェントでの典型的な使い方

```python
import subprocess, os, json

BUILD = '/path/to/llama.cpp/build'

# 固定のシステムプロンプト（毎回同じ → KV がキャッシュされる）
SYSTEM_PROMPT = """
You are a helpful AI assistant specialized in customer support.
Always respond in a polite and professional manner.
... (長い固定プロンプト) ...
"""

def ask_agent(question: str) -> dict:
    """
    システムプロンプトの KV を Redis から復元して質問に回答する。
    初回はプリフィル計算を行い KV を Redis に保存する。
    2回目以降は Redis から KV を復元してプリフィルをスキップする。
    """
    r = subprocess.run(
        ['python3', '/path/to/llmcache_demo.py'],
        capture_output=True, text=True,
        env={
            **os.environ,
            'LD_LIBRARY_PATH': f'{BUILD}/bin',
            'LD_PRELOAD': '',
        }
    )
    if r.returncode == 0:
        result = json.loads(r.stdout.strip().split('\n')[-1])
        if result['restored_tokens'] > 0:
            print(f"✅ KV キャッシュ HIT ({result['restored_tokens']} tokens 復元)")
            print(f"   復元時間: {result['t_restore_ms']:.0f} ms")
        else:
            print(f"⏱  プリフィル計算: {result['t_prefill_ms']:.0f} ms")
            print(f"   KV を Redis に保存しました")
        return result
    else:
        raise RuntimeError(f'エラー: {r.stderr}')

# 使用例
ask_agent('What are your business hours?')   # 初回: プリフィル計算 (17秒)
ask_agent('How can I return a product?')     # 2回目: KV 復元 (1秒) → 17倍高速
ask_agent('Do you offer discounts?')         # 3回目: KV 復元 (1秒) → 17倍高速
```

---

## 🗄️ Redis キー形式

```
{model_name}@{chunk_hash}@chunk    → KV テンソルバイナリ (float16)
```

### チャンクハッシュの計算（LMCache 互換）

```python
import hashlib, numpy as np

def chunk_hash(token_ids: list[int], start: int, chunk_size: int = 256) -> str:
    """
    LMCache と同じ方法でチャンクハッシュを計算する。
    token_ids[start:start+chunk_size] を int32 配列として SHA256 を計算し
    先頭16文字を返す。
    """
    chunk = token_ids[start : start + chunk_size]
    buf   = np.array(chunk, dtype=np.int32).tobytes()
    return hashlib.sha256(buf).hexdigest()[:16]
```

### KV テンソルのバイナリ形式

```
[state_write_meta の出力]
  ├── pos      (int32)  × token_count
  ├── n_seq_id (uint32) × token_count
  └── seq_id   (int32)  × token_count

[state_write_data の出力]
  ├── v_trans  (uint32)
  ├── n_layer  (uint32)
  └── for each layer:
        ├── k_type     (int32)
        ├── k_size_row (uint64)
        └── K テンソル (float16 binary) × token_count rows
      for each layer:
        ├── v_type     (int32)
        ├── v_size_row (uint64)
        └── V テンソル (float16 binary) × token_count rows
```

---

## ⚠️ 設計上の注意点

### 1. チャンクは完全一致のみ
現在の実装はプロンプトのトークン列が完全に一致する場合のみ KV を再利用します。
LMCache のように「部分プレフィックス一致」は未実装です（将来の拡張予定）。

### 2. llama.cpp 本家の追従
llama.cpp は更新が非常に活発です。以下のファイルに変更があった場合は
パッチを再適用する必要があります：
- `src/llama-kv-cache.h` / `.cpp`
- `include/llama.h`

### 3. チャンクサイズ
現在は 256 トークン固定です。モデルやユースケースに応じて変更できますが、
LMCache との互換性を保つには 256 トークンを推奨します。

### 4. text_len に -1 を渡さない
`llmcache_tokenize` の第3引数 `text_len` に `-1` を渡すと
内部で `std::length_error` が発生します。
必ず `len(prompt.encode())` の値を渡してください。

---

## 📚 参考文献

- [LMCache 公式ドキュメント](https://docs.lmcache.ai/)
- [LMCache GitHub](https://github.com/LMCache/LMCache)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [LMCache 論文 (arXiv:2411.06032)](https://arxiv.org/abs/2411.06032)
- [CacheGen 論文 (SIGCOMM 2024)](https://arxiv.org/abs/2310.07240)

---

## 📄 ライセンス

MIT License（llama.cpp と同じ）
