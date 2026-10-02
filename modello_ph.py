import tensorflow as tf

def costruisci_modello() -> tf.keras.Model:
    modello = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return modello

def mostra_architettura(modello: tf.keras.Model) -> None:
    modello.summary()
    print("Il modello della Rete Neurale è pronto")

if __name__ == '__main__':
    modello = costruisci_modello()
    mostra_architettura(modello)
