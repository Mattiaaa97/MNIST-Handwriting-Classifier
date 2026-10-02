import tensorflow as tf
from tensorflow import keras
from modello_ph import modello, x_train, y_train, x_test, y_test

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

metric = tf.keras.metrics.SparseCategoricalAccuracy()

modello.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[metric])

modello.fit(x_train, y_train, epochs=5)

modello.evaluate(x_test, y_test)

print("Allenamento completato! ✅")
