import os

#OpenMPのエラー回避設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ReLU
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam

# 1 MNISTデータセットの読み込み
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2 データの前処理 (正規化 & 次元追加)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 3 データ拡張 (弱める)
datagen = ImageDataGenerator(
    rotation_range=10,  # 回転を抑える
    width_shift_range=0.1,  # 移動範囲を減らす
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)
datagen.fit(x_train)

# 4 CNN モデルを修正（ResNet風のブロックを使用）
def conv_block(filters, input_shape=None):
    """ ResNet風の畳み込みブロック """
    layers = [
        Conv2D(filters, (3,3), padding='same', input_shape=input_shape) if input_shape else Conv2D(filters, (3,3), padding='same'),
        BatchNormalization(),
        ReLU()
    ]
    return layers

model = Sequential([
    *conv_block(32, input_shape=(28, 28, 1)),
    *conv_block(32),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    *conv_block(64),
    *conv_block(64),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    *conv_block(128),
    *conv_block(128),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, kernel_regularizer=l2(0.001)),
    ReLU(),
    Dropout(0.3),

    Dense(10, activation='softmax')
])

# 5 最適化アルゴリズム (`Adam`) を調整
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6 モデルの学習
model.build(input_shape=(None, 28, 28, 1))
model.summary()

history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=30, validation_data=(x_test, y_test))

# 7 モデルの評価
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"テスト精度: {test_acc:.4f}, テスト損失: {test_loss:.4f}")

# 8 学習済みモデルの保存
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/saved_model"))
os.makedirs(save_dir, exist_ok=True)

model_path = os.path.join(save_dir, "my_model.h5")
model.save(model_path)
print(f"モデルが保存されました: {model_path}")
