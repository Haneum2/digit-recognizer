<!-- Created: 2026-09-28 21:38 -->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

Desktop version of the digit recognizer: a tkinter window where the user draws a digit and a PyTorch CNN (trained on MNIST) shows the result and confidence.

## Commands

Python is not on PATH; use the full interpreter path (Python 3.12, per-user install). Run from this folder:

```
PY="$LOCALAPPDATA/Programs/Python/Python312/python.exe"
$PY digits.py                      # run the app (same as run_digits.bat)
$PY -m pip install torch torchvision pillow --index-url https://download.pytorch.org/whl/cpu
```

Self-check (no test framework) — a drawn stroke must read as 1 and a ring as 0:

```
$PY -c "
from PIL import Image, ImageDraw
import digits
m = digits.load_model()
img = Image.new('L', (280, 280), 0); ImageDraw.Draw(img).line((140, 40, 140, 240), fill=255, width=20)
assert digits.recognize(m, img)[0] == 1
img = Image.new('L', (280, 280), 0); ImageDraw.Draw(img).ellipse((30, 20, 150, 200), outline=255, width=20)
assert digits.recognize(m, img)[0] == 0
"
```

## Architecture

Everything lives in `digits.py`:

- `load_model()` loads `mnist_cnn.pt` if present; otherwise downloads MNIST into `data/`, trains the CNN with augmentation (a few minutes on CPU, ~99.5% test accuracy), and saves `mnist_cnn.pt`. **After changing `make_model()`, delete `mnist_cnn.pt`**, or the old weights will fail to load.
- `to_mnist()` makes the drawing look like MNIST: crop to ink, fit into 20x20, paste into 28x28, shift so the center of mass is in the middle, normalize. The input layout depends on this preprocessing matching the training data.
- `App` draws on the tkinter canvas and also on an off-screen PIL image; only the PIL image goes to the model. It predicts on every mouse-button release, so multi-stroke digits also produce partial predictions.

Two launchers:
- `run_digits.bat` runs `python.exe`, so a console shows the printed results. It uses `%~dp0`, so it works from any location as long as it stays in this folder.
- `Digit Recognizer.lnk` runs `pythonw.exe` with no console. It stores **absolute paths**, so recreate it if this folder moves. When it has to train first, nothing appears on screen until training finishes.
