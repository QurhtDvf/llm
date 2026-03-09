MSがリリースしたマルチモーダルモデルのPhi-4-reasoning-vision-15Bの公式推奨スペックは40GB+ GPU。 これに対し、CPU環境でも動かせるように工夫。 画像を諦めてテキスト推論に絞り、llama-server + OpenAI互換API で動作できるようにした。

Microsoft released the multimodal model Phi-4-reasoning-vision-15B on March 4, 2026. The official recommended specification requires a 40GB+ GPU.
I devised a way to run it even in a CPU-only environment. By abandoning the image modality and focusing on text reasoning only, I made it possible to run the model with llama-server and an OpenAI-compatible API.
