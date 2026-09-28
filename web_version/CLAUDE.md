<!-- Created: 2026-09-28 21:38 -->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

Web version of the digit recognizer: the user draws a digit in the browser and sees the recognized digit and its confidence. It is a single static page, `index.html`, with no build step. The page runs the desktop version's trained CNN in the browser using onnxruntime-web, loaded from jsdelivr.

## Commands

The page must be served over HTTP; browsers block loading `mnist_cnn.onnx` from `file://`.

```
PY="$LOCALAPPDATA/Programs/Python/Python312/python.exe"
$PY -m http.server 8000            # run from this folder, then open http://localhost:8000/
```

Re-export the model after retraining the desktop version (run from `../desktop_version`; needs `pip install onnx`):

```
$PY -c "import torch, digits; torch.onnx.export(digits.load_model(), torch.zeros(1, 1, 28, 28), '../web_version/mnist_cnn.onnx', input_names=['input'], output_names=['logits'], dynamo=False)"
```

## Architecture

- `mnist_cnn.onnx` is exported from `../desktop_version/mnist_cnn.pt`. Input name `input` (1x1x28x28, normalized), output name `logits`; the page applies softmax itself.
- `toMnist()` in `index.html` mirrors `to_mnist()` in `../desktop_version/digits.py`: crop to ink, fit into 20x20, paste into 28x28, center by center of mass, normalize with MNIST mean/std. **Keep the two in sync**; accuracy depends on this preprocessing matching the training data.
