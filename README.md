# 手書き数字認識 Webアプリ

ブラウザ上で描いた手書きの数字(0-9)を、AI（CNNモデル）がリアルタイムで認識するWebアプリケーションです。
クラウド環境（Render）の無料枠でも高速に動作するように、TensorFlow Lite を用いてモデルの軽量化と最適化を行っています。

### 🔗 [デモサイトはこちら](https://handwriting-mnist-app.onrender.com)

## 主な機能

* **キャンバス描画:** HTML5 Canvasを使用し、マウスやタッチ操作でスムーズな描画が可能。
* **高精度な認識:** 畳み込みニューラルネットワーク（CNN）による画像認識。
* **軽量・高速化:** モデルを `.tflite` 形式に量子化・変換し、メモリ使用量を大幅に削減。
* **レスポンシブ:** PC、スマートフォン、タブレットのどの端末でも利用可能。

## 使用技術

### AI & Backend
* **Python 3.x**
* **TensorFlow / Keras:** モデルの学習 (`save.py`)。
* **TensorFlow Lite:** 推論時のメモリ最適化と高速化。
* **Flask:** Web APIサーバー構築 (`back.py`)。
* **Gunicorn:** 本番環境用WSGIサーバー。
* **NumPy / Pillow:** 画像データの前処理。

### Frontend
* **HTML5 / CSS3**
* **JavaScript (ES6):** Fetch APIによる非同期通信。
* **Bootstrap 5:** UIデザイン。

### Infrastructure
* **Render:** クラウドデプロイ (PaaS)。

## ディレクトリ構成

```text
handwriting-app/
│
├── back.py              # Flaskアプリケーション本体 (TFLite対応)
├── save.py              # モデル学習用スクリプト (CNN構築)
├── convert.py           # モデル変換用スクリプト (.h5 -> .tflite)
├── Procfile             # Render起動設定
├── requirements.txt     # 依存ライブラリ一覧
├── .gitignore           # Git除外設定
├── README.md            # プロジェクト説明書
│
├── static/              # 静的ファイル
│   ├── index.html       # フロントエンド
│   └── favicon.png      # アイコン
│
└── models/              # モデル格納ディレクトリ
    └── saved_model/
        ├── my_model.h5  # 学習済みKerasモデル (元データ)
        └── model.tflite # 軽量化された推論用モデル (本番で使用)
````

## ローカルでの実行方法

手元のPCで開発や動作確認を行う手順です。

### 1\. クローンと移動

gitよりプロジェクトをクローンしてください。

### 2\. 依存ライブラリのインストール

```bash
# 仮想環境の利用を推奨
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
```

### 3\. (オプション) モデルの再学習と変換

モデルをゼロから作り直したい場合のみ実行してください。

```bash
# 1. CNNモデルの学習 (.h5の生成)
python save.py

# 2. TFLite形式への変換 (.tfliteの生成)
python convert.py
```

### 4\. サーバーの起動

```bash
python back.py
```

ブラウザで `http://127.0.0.1:8000` にアクセスするとアプリが起動します。

## ☁️ Renderへのデプロイ設定

このプロジェクトはRenderの無料プラン（RAM 512MB）で動作するように調整されています。

  * **Build Command:** `pip install -r requirements.txt`
  * **Start Command:** `gunicorn back:app --workers 1 --threads 8 --timeout 0`
  * **Environment:** Python 3

### 工夫した点 (Optimization)

Renderの無料枠ではTensorFlowの通常のモデル読み込みでメモリ不足（OOM）が発生するため、以下の対策を行いました。

  * **TFLite化:** `convert.py` を作成し、モデルを軽量な `.tflite` 形式に変換。推論エンジンを `tf.lite.Interpreter` に変更。
  * **ライブラリ選定:** `tensorflow` の代わりに `tensorflow-cpu` を使用し、フットプリントを削減。
  * **ワーカー制限:** Gunicornのワーカー数を1に制限し、スレッド並列でリクエストを処理。
