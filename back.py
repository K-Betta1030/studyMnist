import logging
import os
import sys
import json
import base64
import signal
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from keras.models import load_model
from PIL import Image
from io import BytesIO

# --- Flaskアプリケーションのセットアップ ---
app = Flask(__name__, static_folder='static')

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception as e:
        logging.warning(f"標準出力のエンコーディング変更に失敗: {e}")

# --- 終了処理（Ctrl+C）のハンドリング ---
def signal_handler(sig, frame):
    """Ctrl+C (SIGINT) を検知したときに呼ばれる関数"""
    print("\n")
    logging.info("停止シグナルを受信しました。サーバーを終了します...")
    sys.exit(0)

# シグナルを登録 (SIGINT = Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


# --- TensorFlow/GPU設定 ---
try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logging.info("GPU メモリ設定: set_memory_growth=True")
    else:
        logging.info("利用可能なGPUが見つかりません。CPUで実行します。")
except Exception as e:
    logging.error(f"GPU設定中にエラーが発生: {e}")


# --- モデルのロード ---
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/saved_model/my_model.h5"))
logging.info(f"モデルパス: {model_path}")

if not os.path.exists(model_path):
    logging.error(f"致命的エラー: モデルファイルが見つかりません: {model_path}")
    sys.exit(1) 

try:
    model = load_model(model_path)
    logging.info(f"モデルが正常にロードされました: {model_path}")
except Exception as e:
    logging.error(f"モデルのロードに失敗しました: {e}")
    sys.exit(1)

# --- 画像前処理関数 ---
def preprocess_image(image_data: str) -> np.ndarray | None:
    try:
        logging.info("画像データのデコードを開始...")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        if image.mode != 'L':
            image = image.convert('L')
        
        if image.size != (28, 28):
            image = image.resize((28, 28), Image.LANCZOS)

        image_array = np.array(image, dtype=np.float32)
        
        # 白黒反転 & 正規化
        image_array = 255.0 - image_array
        image_array = image_array / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        return image_array

    except Exception as e:
        logging.error(f"画像の前処理中にエラーが発生しました: {e}", exc_info=True)
        return None

# --- Flaskルーティング ---

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("POST /predict を受信しました")
    try:
        if not request.is_json:
            return jsonify({'error': 'Invalid content type'}), 400

        data = request.get_json()
        if 'imageData' not in data:
            return jsonify({'error': 'Missing imageData'}), 400

        processed_image = preprocess_image(data['imageData'])
        if processed_image is None:
            return jsonify({'error': 'Image preprocessing failed'}), 500

        prediction = model.predict(processed_image)
        digit = np.argmax(prediction)
        probability = np.max(prediction)
        
        logging.info(f"予測結果: {digit} (確率: {probability:.4f})")
        
        return jsonify({
            'result': str(digit),
            'probability': float(probability)
        }), 200

    except Exception as e:
        logging.error(f"予測エラー: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

# --- アプリケーションの実行 ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)