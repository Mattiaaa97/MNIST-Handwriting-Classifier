# 🔢 MNIST Handwritten Digit Classifier

Progetto in Python per riconoscere le cifre scritte a mano da 0 a 9 usando il dataset MNIST e TensorFlow[span_0](start_span)[span_0](end_span)[span_1](start_span)[span_1](end_span).

---

## 📌 Cosa fa il progetto

- Scarica le immagini delle cifre dal dataset MNIST[span_2](start_span)[span_2](end_span)
- Divide i valori dei pixel per 255 per portarli tra 0 e 1[span_3](start_span)[span_3](end_span)
- Crea una rete neurale semplice con 3 livelli (Flatten, Dense 128, Dense 10)[span_4](start_span)[span_4](end_span)
- Allena il modello per 5 giri (epoche) con Adam[span_5](start_span)[span_5](end_span)
- Controlla quante cifre vengono indovinate alla fine[span_6](start_span)[span_6](end_span)

---

## 📁 I tre file

* `load_mnist_data.py`: scarica le immagini, le divide per 255 e controlla le dimensioni[span_7](start_span)[span_7](end_span)
* `modello_ph.py`: crea la rete neurale e mostra il riassunto a schermo[span_8](start_span)[span_8](end_span)
* `training_data.py`: avvia l'allenamento per 5 epoche e calcola il risultato finale[span_9](start_span)[span_9](end_span)

---

## ⚙️ Come si usa

1. Installa i pacchetti necessari:
pip install tensorflow numpy

2. Avvia l'allenamento:
python training_data.py

---

**Autore:** Mattia Dellanoce
