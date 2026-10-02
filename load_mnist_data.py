import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import ndarray


def import_mnist() -> tuple[ndarray, ndarray, ndarray, ndarray]:
    # Carichiamo i dati originali
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalizziamo le immagini (portandole tra 0 e 1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Restituiamo i 4 elementi "sciolti" per rispettare il Type Hint sopra
    return x_train, y_train, x_test, y_test


# Adesso li riceviamo correttamente in 4 variabili separate
x_train, y_train, x_test, y_test = import_mnist()

print("Verifichimo le dimensioni -> ", x_train.shape)
print(y_train[0])

