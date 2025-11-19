import logging
import os
import sys
import json
import base64
import signal
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from io import BytesIO

app = Flask(__name__, static_folder='static')

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO)

# --- 終了処理 ---
def signal_handler(sig, frame):
    logging.info("停止シグナルを受信しました。")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# --- TFLiteモデルのロード ---
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/saved_model/model.tflite"))
logging.info(f"モデルパス: {model_path}")

if not os.path.exists(model_path):
    logging.error(f"致命的エラー: モデルファイルが見つかりません: {model_path}")
    sys.exit(1)

try:
    # 軽量なインタープリタを使用
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # 入力と出力の情報を取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logging.info("TFLiteモデルがロードされました")
except Exception as e:
    logging.error(f"モデルロード失敗: {e}")
    sys.exit(1)

# --- 前処理 ---
def preprocess_image(image_data: str) -> np.ndarray | None:
    try:
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        if image.mode != 'L':
            image = image.convert('L')
        if image.size != (28, 28):
            image = image.resize((28, 28), Image.LANCZOS)

        image_array = np.array(image, dtype=np.float32)
        image_array = 255.0 - image_array
        image_array = image_array / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        return image_array
    except Exception as e:
        logging.error(f"前処理エラー: {e}")
        return None

# --- ルーティング ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json: return jsonify({'error': 'Invalid content type'}), 400
        data = request.get_json()
        if 'imageData' not in data: return jsonify({'error': 'Missing imageData'}), 400

        processed_image = preprocess_image(data['imageData'])
        if processed_image is None: return jsonify({'error': 'Preprocessing failed'}), 500

        # --- TFLiteでの予測実行 ---
        # 1. データをセット
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # 2. 推論実行
        interpreter.invoke()
        
        # 3. 結果を取得
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
