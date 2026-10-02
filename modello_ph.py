import tensorflow as tf
from tensorflow import keras
from vedere import x_train, y_train, x_test, y_test, import_mnist

x_train, y_train, x_test, y_test = import_mnist()

modello = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                      tf.keras.layers.Dense(128, activation='relu'),
                                      tf.keras.layers.Dense(10, activation='softmax')])
modello.summary()

print('Il modello della Rete Neurale è pronto')
