# 🔢 MNIST Handwritten Digit Classifier

Progetto in Python per riconoscere le cifre scritte a mano da 0 a 9 usando il dataset MNIST e TensorFlow.

---

## 📌 Cosa fa il progetto

- Scarica le immagini delle cifre dal dataset MNIST
- Divide i valori dei pixel per 255 per portarli tra 0 e 1
- Crea una rete neurale semplice con 3 livelli (Flatten, Dense 128, Dense 10)
- Allena il modello per 5 giri (epoche) con Adam
- Controlla quante cifre vengono indovinate alla fine

---

## 📁 I tre file

* load_mnist_data.py: scarica le immagini, le divide per 255 e controlla le dimensioni
* modello_ph.py: crea la rete neurale e mostra il riassunto a schermo
* training_data.py: avvia l'allenamento per 5 epoche e calcola il risultato finale

---

## ⚙️ Come si usa

1. Installa i pacchetti necessari:
pip install tensorflow numpy

2. Avvia l'allenamento:
python training_data.py

---

Autore: Mattia Dellanoce
