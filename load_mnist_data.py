import tensorflow as tf
from numpy import ndarray

def importa_e_normalizza_mnist() -> tuple[ndarray, ndarray, ndarray, ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test

def stampa_info_dataset(x_train: ndarray, y_train: ndarray) -> None:
    print("Verifichiamo le dimensioni ->", x_train.shape)
    print("Primo target ->", y_train[0])

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = importa_e_normalizza_mnist()
    stampa_info_dataset(x_train, y_train)
