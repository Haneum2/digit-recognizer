<!-- Created: 2026-09-28 21:45 -->
# Handwritten Digit Recognizer

Draw a digit (0-9) with the mouse and the program shows the recognized digit and its confidence. It comes in two versions that share one trained model:

| Version | Stack | Folder |
|---|---|---|
| Desktop | Python, tkinter, PyTorch | `desktop_version/` |
| Web | HTML/JavaScript, onnxruntime-web | `web_version/` |

## Model

- A small CNN (4 conv layers + 2 dense layers) trained on MNIST (60,000 training images).
- Trained with random rotation, shift, and zoom so it copes with how people actually draw.
- **Test accuracy: 99.51%** on the 10,000 MNIST test images.
- Before recognition, the drawing goes through the same preprocessing as MNIST: crop to the ink, fit into 20x20, place in 28x28, and center by center of mass.
- The trained weights are included (`desktop_version/mnist_cnn.pt`, exported to `web_version/mnist_cnn.onnx`), so no training is needed to run either version.

## Run the desktop version

Requires Python 3.10+.

```
cd desktop_version
pip install torch torchvision pillow
python digits.py
```

On Windows you can also double-click `run_digits.bat` (it expects Python 3.12 installed per user). If `mnist_cnn.pt` is deleted, the first run downloads MNIST and retrains (a few minutes on CPU).

## Run the web version

The page loads the model file, so it must be served over HTTP (opening `index.html` directly does not work):

```
cd web_version
python -m http.server 8000
```

Then open http://localhost:8000/. The model runs entirely in the browser.

## Output

- **Result**: the recognized digit
- **Confidence**: probability of that digit (green >= 90%, orange >= 60%, red below)
- **Next**: the two next most likely digits

The desktop version also prints `Result: 7  Confidence: 98.3%` to the console.
