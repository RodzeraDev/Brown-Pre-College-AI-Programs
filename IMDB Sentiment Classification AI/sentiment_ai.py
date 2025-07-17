import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("GPUs:", tf.config.list_physical_devices('GPU'))

num_words = 10000
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)

maxlen = 250
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

def reverse_sequence(data):
    return np.array([x[::-1] for x in data])

x_train_rev = reverse_sequence(x_train)
y_train_rev = y_train
x_train_aug = np.concatenate([x_train, x_train_rev])
y_train_aug = np.concatenate([y_train, y_train_rev])

model = models.Sequential([
    layers.Embedding(input_dim=num_words, output_dim=64, input_length=maxlen),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train_aug, y_train_aug, epochs=5, batch_size=512, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print(f"\n✅ Final test accuracy: {acc:.4f}")

sample = x_test[0]
prediction = model.predict(sample.reshape(1, -1))
print("Predicted:", "Positive" if prediction[0][0] > 0.5 else "Negative")
print("Actual:   ", "Positive" if y_test[0] == 1 else "Negative")
