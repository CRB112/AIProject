# AIProject

A collection of machine learning algorithms built from scratch in Python, written as a hands-on way to understand the fundamentals before leaning on frameworks. Each algorithm is self-contained and interactive — inputs are prompted at runtime or read from a file.

---

## Algorithms

### Neural Network — `algorithms/NeuralNetwork/NN.py`
A fully connected feedforward network trained on the Iris dataset (binary classification — setosa vs. versicolor). Built with NumPy only. Supports configurable hidden layers, learning rate, and epoch count, all entered at runtime.

- Forward pass with ReLU activations and sigmoid output
- Backpropagation with gradient descent
- Sepal scatter plot shown before training for data visualization

### K-Means Clustering — `algorithms/clustering/clustering.py`
An implementation of the K-Means algorithm on randomly generated 2D cluster data. Centroid positions are visualized before and after convergence.

- Random centroid initialization
- Iterative assignment and centroid update
- Tolerance-based convergence with a max iteration cap

### Hidden Markov Model — `algorithms/hiddenMarkov/hiddenMarkov.py`
Runs the Viterbi algorithm over a randomly generated observation sequence. Reads state definitions from `Input.txt` — if the file is missing or empty, one is generated automatically.

Input format for `Input.txt`:
```
start: s0
s0:[0.7,0.3]:[0.4,0.6]
s1:[0.4,0.6]:[0.5,0.5]
```
Each line defines a state with its transition weights and emission probabilities.

### Linear Regression — `algorithms/linearReg/linearReg.py`
Ordinary least squares regression on x,y data read from `input.txt`. If the file doesn't exist, random data is generated and saved. Plots the data and the best-fit line, then accepts a new X value for prediction.

Input format for `input.txt`:
```
1.0,2.3
2.5,4.1
3.8,6.0
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Run

```bash
python run.py
```

This installs all required dependencies and presents a menu to select which algorithm to run. You can also run any script directly:

```bash
python algorithms/NeuralNetwork/NN.py
python algorithms/clustering/clustering.py
python algorithms/hiddenMarkov/hiddenMarkov.py
python algorithms/linearReg/linearReg.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Core math and matrix operations |
| `matplotlib` | Plotting and visualization |
| `pandas` | Data formatting (Neural Network) |
| `scikit-learn` | Iris dataset and train/test split |
| `torch` / `torchvision` / `torchaudio` | Available for future use |
| `seaborn` | Available for future use |

Full pinned versions are in `requirements.txt`.

---

## Notes

- The neural network uses plain NumPy — no PyTorch in that module yet. It was written to build intuition for what frameworks do under the hood.
- The Hidden Markov Model auto-generates input if none is provided, so it works out of the box.
- Linear regression will prompt for point count and generate data if `input.txt` is missing.