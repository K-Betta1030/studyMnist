import tensorflow as tf
import os

# パスの設定
model_dir = os.path.join("models", "saved_model")
h5_path = os.path.join(model_dir, "my_model.h5")
tflite_path = os.path.join(model_dir, "model.tflite")

# 1. 既存のKerasモデル(.h5)をロード
print(f"Loading model from {h5_path}...")
model = tf.keras.models.load_model(h5_path)

# 2. TFLiteコンバーターを作成
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. 変換を実行
print("Converting model to TFLite format...")
tflite_model = converter.convert()

# 4. ファイルに保存
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"変換完了! 保存先: {tflite_path}")
print(f"ファイルサイズ: {os.path.getsize(tflite_path) / 1024:.2f} KB")
