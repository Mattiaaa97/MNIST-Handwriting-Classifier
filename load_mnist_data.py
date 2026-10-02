import tensorflow as tf
from numpy import ndarray

def import_mnist() -> tuple[ndarray, ndarray, ndarray, ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = import_mnist()

print("Verifichimo le dimensioni -> ", x_train.shape)
print(y_train[0])
