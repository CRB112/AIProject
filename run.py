import subprocess
import sys
import os

REQUIRED = [
    "numpy",
    "matplotlib",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "torchvision",
    "torchaudio",
    "seaborn",
    "pillow",
]

ALGORITHMS = {
    "1": ("Neural Network (Iris classifier)",   "algorithms/NeuralNetwork/NN.py"),
    "2": ("K-Means Clustering",                 "algorithms/clustering/clustering.py"),
    "3": ("Hidden Markov Model (Viterbi)",       "algorithms/hiddenMarkov/hiddenMarkov.py"),
    "4": ("Linear Regression",                  "algorithms/linearReg/linearReg.py"),
}


def install_dependencies():
    print("\n--- Installing dependencies ---\n")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + REQUIRED,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        print("All dependencies installed.\n")
    except subprocess.CalledProcessError:
        print("pip install failed. Try running: pip install -r requirements.txt")
        sys.exit(1)


def run_algorithm(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    subprocess.run([sys.executable, path])


def main():
    print("=" * 48)
    print("  AIProject — Connor's PyTorch Practice Suite")
    print("=" * 48)

    install_dependencies()

    print("Select an algorithm to run:\n")
    for key, (label, _) in ALGORITHMS.items():
        print(f"  [{key}] {label}")
    print("  [a] Run all")
    print("  [q] Quit\n")

    choice = input("Choice -> ").strip().lower()

    if choice == "q":
        return
    elif choice == "a":
        for _, (label, path) in ALGORITHMS.items():
            print(f"\n--- Running: {label} ---\n")
            run_algorithm(path)
    elif choice in ALGORITHMS:
        label, path = ALGORITHMS[choice]
        print(f"\n--- Running: {label} ---\n")
        run_algorithm(path)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()