# 🚀 vLLM on Google Colab

Google Colab（T4 GPU）で vLLM を動かすためのノートブックです。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/vllm_colab_final.ipynb)

---

## 📋 目次

- [vLLM とは](#vllm-とは)
- [vLLM の歴史](#vllm-の歴史)
- [主な技術的特徴](#主な技術的特徴)
- [他の推論サーバーとの比較](#他の推論サーバーとの比較)
- [使い方](#使い方)
- [T4 GPU での注意点](#t4-gpu-での注意点)

---

## vLLM とは

vLLM（Virtual Large Language Model）は、UC Berkeley の Sky Computing Lab が開発した **オープンソースの LLM 高速推論・サービングエンジン**です。

PagedAttention と呼ばれる独自のメモリ管理アルゴリズムを中核に据え、GPU メモリの利用効率を大幅に改善することで、従来の推論エンジンと比べて高いスループットを実現します。OpenAI 互換の API サーバーとして動作するため、既存のアプリケーションに容易に組み込めることも大きな特長です。

現在は Amazon Rufus や LinkedIn の AI 機能など、多くのプロダクション環境で採用されています。

---

## vLLM の歴史

### 2023年：誕生と急成長

- **2023年6月** — vLLM が正式リリース。PagedAttention を導入し、GPU メモリの無駄を大幅に削減することに成功した。当時の既存推論エンジンは GPU メモリの 20〜40% しか活用できていなかったが、vLLM はこれを大幅に改善した。
- **2023年8月** — a16z の Open Source AI Grant に採択。コア開発者の Woosuk Kwon と Zhuohan Li が資金支援を受ける。
- **2023年9月** — SOSP 2023（ACM オペレーティングシステム原理シンポジウム）にて PagedAttention に関する論文を発表。
- **2023年末** — PyPI への定期リリースが始まり、コミュニティへの普及が加速。

### 2024年：爆発的な拡大

- **2024年初頭** — サポートするモデルは数種類だったが、年末までに約 100 のモデルアーキテクチャに対応。LLM・マルチモーダル・Encoder-Decoder・投機的デコーディングなど幅広くカバー。
- **2024年7月** — ZhenFund からの寄付を受け取り、Linux Foundation の LF AI & Data に参加。
- **2024年9月** — v0.6.0 をリリース。CPU スケジューリングなどの最適化によりレイテンシを約 5 倍改善、スループットを約 2.7 倍向上。
- **2024年後半** — コアアーキテクチャを大幅に再設計した「V1 アーキテクチャ」の開発を開始。フルタイムコントリビューターが 15 名以上、主要スポンサー組織が 20 社以上に拡大。IBM・AWS・NVIDIA との戦略的パートナーシップも確立。

### 2025年〜現在：産業標準へ

- **2025年初頭** — V1 アーキテクチャを正式公開。DeepSeek V3/R1 の登場とともに、推論エンジンへの関心が爆発的に高まる。
- **2025年** — PyTorch Foundation の傘下に移行し、PyTorch との深い連携を計画。Red Hat による Neural Magic の買収に伴い、vLLM の商用サポート体制がさらに強化される。
- **2026年** — GitHub スター数は 5 万以上。オープンソース LLM サービングのデファクトスタンダードとして確立されている。

---

## 主な技術的特徴

### PagedAttention

vLLM 最大の技術革新です。OS のメモリページング機構からインスピレーションを得て、KV キャッシュを固定長の「ブロック」に分割して管理します。これにより、メモリの断片化を防ぎ、GPU メモリの利用効率を大幅に向上させます。同一のモデルでより多くのリクエストを同時処理できるようになります。

### Continuous Batching（継続的バッチ処理）

従来の静的バッチ処理は、バッチ内のすべてのリクエストが完了するまで次のリクエストを受け付けませんでした。vLLM の継続的バッチ処理は、生成が終わったシーケンスをリアルタイムで新しいリクエストに置き換えるため、GPU をほぼ 100% 稼働させ続けることができます。

### OpenAI 互換 API

`/v1/completions` や `/v1/chat/completions` といった OpenAI API と互換性のあるエンドポイントを提供します。既存の OpenAI クライアントライブラリをそのまま使えるため、移行コストが非常に低く抑えられます。

### 幅広いハードウェアサポート

| ハードウェア | サポート状況 |
|---|---|
| NVIDIA GPU（V100〜H100） | ✅ フルサポート |
| AMD GPU（MI200/MI300シリーズ） | ✅ サポート |
| Google TPU（v4/v5/v6） | ✅ サポート |
| AWS Inferentia / Trainium | ✅ サポート |
| Intel Gaudi / GPU | ✅ サポート |
| CPU（量子化モデル） | ✅ サポート |

### その他の主な機能

- **テンソル並列・パイプライン並列** — 複数 GPU への分散推論
- **量子化サポート** — AWQ・GPTQ・FP8 など多様な量子化形式に対応
- **投機的デコーディング** — 小型モデルを使ったデコード高速化
- **LoRA サポート** — 複数のアダプタを単一のベースモデルで効率的に管理
- **プレフィックスキャッシング** — 共通プレフィックスの KV キャッシュを再利用して高速化

---

## 他の推論サーバーとの比較

| | vLLM | TensorRT-LLM | TGI (HuggingFace) | Ollama | SGLang | llama.cpp |
|---|---|---|---|---|---|---|
| **開発元** | UC Berkeley | NVIDIA | Hugging Face | Ollama Inc. | UC Berkeley | Georgi Gerganov |
| **セットアップ難易度** | 低〜中 | 高 | 低〜中 | 非常に低 | 低〜中 | 非常に低 |
| **スループット** | ◎ 高い | ◎ 最高（NVIDIA 限定） | ○ 中〜高 | △ 低い | ◎ 高い | △ 低〜中 |
| **メモリ効率** | ◎ PagedAttention | ○ | ○ | △ | ◎ RadixAttention | ◎ GGUF 量子化 |
| **OpenAI 互換 API** | ✅ | 要設定 | ✅ | ✅ | ✅ | ✅ |
| **対応ハードウェア** | マルチ | NVIDIA 専用 | マルチ | マルチ | マルチ | マルチ（CPU 含む）|
| **GPU なし（CPU のみ）** | ❌ | ❌ | ❌ | △ | ❌ | ✅ |
| **本番環境向け** | ✅ | ✅ | ✅ | △ | ✅ | △ |
| **同時リクエスト処理** | ✅ | ✅ | ✅ | ❌ | ✅ | △ |
| **GitHub スター数** | 5 万以上 | 9 千以上 | 1.5 万以上 | 1 万以上 | 1.5 万以上 | 8.5 万以上 |

### 各ツールの使い分け

**vLLM を選ぶ場面**
プロダクション環境で多数の同時リクエストをさばく必要があるとき。セットアップが比較的簡単で、OpenAI 互換 API をすぐに使いたいとき。NVIDIA 以外のハードウェア（AMD・TPU など）も使いたいとき。

**TensorRT-LLM を選ぶ場面**
NVIDIA H100/B200 など最新世代の GPU を大量に保有しており、最大限のスループットが必要なとき。セットアップに 1〜2 週間の工数をかけられるとき。

**TGI を選ぶ場面**
すでに Hugging Face のエコシステムを活用しており、そこから移行コストをかけたくないとき。非常に長いプロンプト（20 万トークン以上）を扱うとき。

**Ollama を選ぶ場面**
ローカル開発・プロトタイピングで素早く試したいとき。同時リクエスト処理が不要な個人用途のとき。

**SGLang を選ぶ場面**
RAG やマルチターン会話など、共通プレフィックスが多いワークロードのとき。構造化出力（JSON など）を大量に生成するとき。

**llama.cpp を選ぶ場面**
GPU を持っていない、または Mac・Raspberry Pi・スマートフォンなどのエッジデバイスで動かしたいとき。GGUF 形式の量子化モデルを使って VRAM を節約したいとき。C/C++ ベースで依存関係をゼロにしたいとき。

---

## 使い方

### 前提条件

- Google Colab アカウント
- GPU ランタイム（T4 推奨）

### 手順

1. ノートブックを Colab で開く
2. **ランタイム → ランタイムのタイプを変更 → GPU (T4)** を選択
3. ②のインストールセルを実行
4. **ランタイム → セッションを再起動**
5. ③以降を順番に実行

### インストールされるパッケージ

```
vllm==0.6.6
numpy==1.26.4
protobuf==4.25.3
transformers==4.46.3
```

---

## T4 GPU での注意点

Google Colab の無料プランで提供される T4 GPU（compute capability 7.5）には、最新版 vLLM との互換性の問題があります。このノートブックでは動作確認済みの組み合わせに固定しています。

| 問題 | 原因 | 対処 |
|---|---|---|
| `numpy.dtype size changed` | numpy バイナリ非互換 | `numpy==1.26.4` を指定してランタイム再起動 |
| `Free memory less than desired` | VRAM 不足 | `LLM()` を先に終了 or ランタイム再起動 |
| `-arch=compute_ is unsupported` | vLLM 0.7+ が T4 非対応 | `vllm==0.6.6` を使用 |
| `all_special_tokens_extended` エラー | transformers バージョン非互換 | `transformers==4.46.3` を指定 |
| `Connection refused` | サーバー起動前に curl | ログで `startup complete` を確認してから実行 |
| `MessageFactory` エラー | protobuf バージョン競合 | 無視してOK（動作に影響なし）|

> **注意：** `LLM()` による直接推論（③）と OpenAI 互換サーバー（④）は同時に使用できません。どちらか一方を使う場合は、もう一方を起動する前にランタイムを再起動してください。

---

## 参考リンク

- [vLLM 公式サイト](https://vllm.ai)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM ドキュメント](https://docs.vllm.ai)
- [PagedAttention 論文 (SOSP 2023)](https://arxiv.org/abs/2309.06180)
- [vLLM Blog](https://blog.vllm.ai)
