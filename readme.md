# 🧠 Handwritten Digit Recognition (Rust, From Scratch)

    

A high-performance handwritten digit recognition system built entirely in **Rust**, implementing a neural network **from scratch without any ML libraries**.

> ⚡ Focus: Systems Programming + Machine Learning + Performance

---

## 🚀 Features

* 🧠 Neural network from scratch (no TensorFlow/PyTorch)
* ⚙️ Manual forward propagation + backpropagation
* 🎯 MNIST training + evaluation pipeline
* 🎮 Interactive drawing canvas (Raylib)
* 💾 Model save/load (`.mnistai`)
* 📦 Embedded dataset support (`include_str!`)
* ⚡ Optimized Rust implementation

---

## 🏗️ Tech Stack

* **Language:** Rust (Edition 2024)
* **Graphics:** raylib
* **Core Concepts:**

  * Feedforward Neural Networks
  * Backpropagation
  * ReLU Activation
  * Softmax
  * Gradient Descent

---

## 📁 Project Structure

```
src/
├── main.rs        # CLI + runtime loop
├── network.rs     # Neural network (forward + backprop)
├── training.rs    # Training + evaluation
├── dataset.rs     # MNIST CSV loader
├── draw.rs        # Drawing canvas (Raylib)
├── utils.rs       # Math utilities
```

---

## 🧠 Model Architecture

| Layer  | Size |
| ------ | ---- |
| Input  | 784  |
| Hidden | 64   |
| Output | 10   |

---

## ⚙️ How It Works

```
Image (28x28) → Flatten → Neural Network → Softmax → Prediction
```

* Normalize pixel values → `[0,1]`
* Forward pass through network
* Compute loss
* Backpropagate gradients
* Update weights

---

## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/st_Bw2JLXkc/0.jpg)](https://youtu.be/st_Bw2JLXkc)

> Click the image to watch the full demo on YouTube.

---

## 🖥️ Running the Project

### 1️⃣ Clone

```bash
git clone https://github.com/princeb777/handwritten-digits-recognition.git
cd handwritten-digits-recognition
```

### 2️⃣ Add Dataset

Download MNIST CSV files and place them in:

```
data/
├── train.csv
├── test.csv
```

### 3️⃣ Build

```bash
cargo build --release
```

### 4️⃣ Run

```bash
cargo run
```

---

## 🎮 Usage

On running:

```
0 : Use Default MNIST model  
1 : Load saved model  
2 : Train the MNIST model  
```

Then:

* Draw digit using mouse
* Release mouse → prediction appears in terminal

---

## 💾 Model Persistence

Models are saved as:

```
model_name.mnistai
```

Reload anytime without retraining.

---

## ⚡ Performance Highlights

* 🚀 No external ML frameworks
* 🧮 Static array computations (fast & cache-friendly)
* 🔧 Release optimizations:

  * LTO enabled
  * Binary stripping
* ⚙️ Minimal runtime overhead

---

## 📊 Example Output

```
Prediction = (7, 98.23%)
```

---

## 🎯 Why This Project Stands Out

**Most ML projects:**

* Use high-level frameworks
* Hide implementation details

**This project:**

* Builds everything from scratch
* Demonstrates deep ML understanding
* Combines systems programming with AI
* Shows strong low-level optimization skills

---

## 📈 Training Results

After **5 epochs**, the model achieves **~94.76% accuracy** on the MNIST test set.

```
0 : Use Default MNIST model
1 : Load the MNIST model
2 : Train the MNIST model
2
Training the model... you might want to grab a cup of coffee while it runs.
Loading Training Data (embedded)
Loaded 60000 samples
Loading Testing Data (embedded)
Loaded 10000 samples
acc 0.9088
acc 0.9259
acc 0.9343
acc 0.9422
acc 0.9476
Enter the file name to save the model
Press Enter to skip
```

---

## 🔮 Future Improvements

* [ ] Multiple hidden layers

* [ ] SIMD optimizations

* [ ] GPU acceleration

* [ ] WebAssembly version

* [ ] Improved UI/UX


---

## 👨‍💻 Author

**Prince Banjare**

* Rust | Machine Learning | Systems Programming
* Interested in high-performance AI systems

---

## ⭐ Support

If you like this project:

⭐ Star this repo
