
# MNIST Handwritten Digit Recognition (PyTorch)

This project implements a simple **Convolutional Neural Network (CNN)** for handwritten digit classification using the **MNIST** dataset. The model is trained, evaluated, and the results are visualized through loss/accuracy graphs and sample predictions.

---

## 📚 Libraries Used

* `torch`
* `torchvision`
* `matplotlib`
* `numpy`

---

## 📦 Installation

```bash
pip install torch torchvision matplotlib numpy
```

---

## 🚀 How to Run

```bash
python mnist_cnn.py
```

---

## 📊 Results

* **Training and Test Loss**
 ![image](https://github.com/user-attachments/assets/a110fc80-3636-4a75-bfdb-a533e9fbf54e)


* **Training and Test Accuracy**
  ![Accuracy](./path_to_accuracy_image.png)

* **Sample Predictions**
  ![Predictions](./path_to_predictions_image.png)

> 📌 *Note: Make sure to place the `.png` images in the correct project directory and update the `path_to_*.png` placeholders accordingly.*

---

## 🧩 Model Architecture

* 2 Convolutional Layers
* 2 Dropout Layers
* 2 Fully Connected (FC) Layers
* Activation Function: **ReLU**
* Output Layer: **LogSoftmax**

---

## 📈 Training Performance

* Maximum Accuracy: **\~99%**
* Training Duration: **10 Epochs**
* Optimizer: **Adam**

---


