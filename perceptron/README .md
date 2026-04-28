
# Pythonでのパーセプトロン実装

このリポジトリでは、ニューラルネットワークの基本的な構成要素であるシンプルなパーセプトロンを、PythonとNumPyを使ってゼロから実装する手順を詳しく説明しています。ANDゲートやORゲートのような線形分離可能なデータをパーセプトロンがどのように学習して分類するかを理解することを目的としています。

## プロジェクトの現状

### 1. パーセプトロンの実装 (完了)

基本的な論理ゲートを学習できるパーセプトロンモデルを実装しました。この実装には以下が含まれます。

*   **`Perceptron`クラスの定義**:
    *   **コンストラクタ (`__init__`)**: 指定された入力数と学習率でパーセプトロンを初期化します。重みとバイアスは、小さなランダム値で初期化されます。
        ```python
        def __init__(self, num_inputs, learning_rate=0.1):
            self.weights = np.random.rand(num_inputs)
            self.bias = np.random.rand(1)[0]
            self.learning_rate = learning_rate
        ```
    *   **活性化関数 (`_activation_function`)**: 単純なステップ関数が使用され、入力が0より大きい場合は1を返し、そうでない場合は0を返します。
        ```python
        def _activation_function(self, x):
            return 1 if x > 0 else 0
        ```
    *   **予測メソッド (`predict`)**: 入力と重みの内積を計算し、バイアスを追加した後、活性化関数を適用して出力（0または1）を生成します。
        ```python
        def predict(self, inputs):
            weighted_sum = np.dot(inputs, self.weights) + self.bias
            return self._activation_function(weighted_sum)
        ```
    *   **訓練メソッド (`train`)**: パーセプトロン学習ルールを実装します。指定されたエポック数にわたって訓練データを繰り返し処理し、誤分類が発生するたびに重みとバイアスを調整します。調整はエラー、入力、および学習率に比例します。
        ```python
        def train(self, training_inputs, labels, epochs):
            for epoch in range(epochs):
                errors = 0
                for inputs, label in zip(training_inputs, labels):
                    prediction = self.predict(inputs)
                    error = label - prediction
                    if error != 0:
                        self.weights += self.learning_rate * error * inputs
                        self.bias += self.learning_rate * error
                        errors += 1
                if errors == 0:
                    print(f"Epoch {epoch + 1}: No errors. Perceptron converged.")
                    break
        ```

*   **AND/ORゲートの訓練と検証**:
    *   パーセプトロンは、ANDゲートとORゲートの標準的な入力パターン（例：`[0,0]`、`[0,1]`、`[1,0]`、`[1,1]`）で訓練されます。
    *   訓練後、モデルの予測はこれらの論理ゲートの期待される出力と比較して検証され、線形分離可能な問題を学習して正しく分類する能力が示されます。

完全な実装は `perceptron.py` で見つけることができます。

## 始めるにあたって

`perceptron.py` コードを実行するには:

1.  `numpy` がインストールされていることを確認してください:
    ```bash
    pip install numpy
    ```
2.  Pythonスクリプトを実行します:
    ```bash
    python perceptron.py
    ```
    このスクリプトは以下を実行します:
    *   ANDゲート用のパーセプトロンを初期化して訓練します。
    *   その予測を検証します。
    *   ORゲート用のパーセプトロンを初期化して訓練します。
    *   その予測を検証します。

## ファイル

*   `perceptron.py`: `Perceptron` クラスと、AND/ORゲートの訓練およびテスト用のサンプルコードが含まれています。
