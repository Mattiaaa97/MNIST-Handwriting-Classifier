import tensorflow as tf
from numpy import ndarray
from load_mnist_data import importa_e_normalizza_mnist
from modello_ph import costruisci_modello

def compila_modello(modello: tf.keras.Model) -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()
    modello.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=[metric])

def allena_modello(modello: tf.keras.Model, x_train: ndarray, y_train: ndarray, epoche: int = 5):
    return modello.fit(x_train, y_train, epochs=epoche)

def valuta_modello(modello: tf.keras.Model, x_test: ndarray, y_test: ndarray) -> None:
    modello.evaluate(x_test, y_test)
    print("Allenamento completato! ✅")

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = importa_e_normalizza_mnist()
    modello = costruisci_modello()
    compila_modello(modello)
    allena_modello(modello, x_train, y_train)
    valuta_modello(modello, x_test, y_test)
