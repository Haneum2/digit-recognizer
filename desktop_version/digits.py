"""Handwritten digit recognizer: draw a digit with the mouse, a CNN guesses it.

Requires: pip install torch torchvision pillow
First run downloads MNIST and trains a CNN (a few minutes on CPU), then caches it.
"""
import os
import tkinter as tk

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "mnist_cnn.pt")
CANVAS = 280  # drawing area in pixels (10x the 28x28 MNIST size)
BRUSH = 10    # brush radius
NORM = transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std


def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(128, 10),
    )


def load_model(epochs=4):
    """Load the cached CNN, or train one on MNIST and cache it."""
    model = make_model()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        return model.eval()

    print("Downloading MNIST and training (first run only)...")
    data_dir = os.path.join(HERE, "data")
    # Random small rotations/shifts/zooms make the model robust to how people actually draw.
    augment = transforms.Compose([
        transforms.RandomAffine(12, translate=(0.12, 0.12), scale=(0.85, 1.15)),
        transforms.ToTensor(), NORM,
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=augment)
    test = datasets.MNIST(data_dir, train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor(), NORM]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1000)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            correct = sum((model(x).argmax(1) == y).sum().item() for x, y in test_loader)
        print(f"Epoch {epoch + 1}/{epochs} - test accuracy {correct / len(test):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    return model


def to_mnist(img):
    """Preprocess like MNIST: crop to ink, fit in 20x20, center by center of mass in 28x28."""
    box = img.getbbox()
    if box is None:
        return None  # empty canvas
    digit = img.crop(box)
    digit.thumbnail((20, 20), Image.LANCZOS)
    out = Image.new("L", (28, 28), 0)
    out.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))

    a = np.asarray(out, dtype=np.float32) / 255.0
    ys, xs = np.indices(a.shape)
    cy, cx = (ys * a).sum() / a.sum(), (xs * a).sum() / a.sum()
    a = np.roll(a, (round(14 - cy), round(14 - cx)), axis=(0, 1))
    return NORM(torch.from_numpy(a)[None])[None]  # shape (1, 1, 28, 28)


def recognize(model, img):
    """Return (digit, confidence, all probabilities) or None for an empty drawing."""
    x = to_mnist(img)
    if x is None:
        return None
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0]
    return int(p.argmax()), float(p.max()), p.tolist()


class App:
    def __init__(self, root, model):
        self.model = model
        self.canvas = tk.Canvas(root, width=CANVAS, height=CANVAS, bg="black", cursor="cross")
        self.canvas.pack(padx=10, pady=10)
        # Off-screen copy of the drawing that is fed to the model.
        self.image = Image.new("L", (CANVAS, CANVAS), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.result = tk.Label(root, text="Draw a digit", font=("Arial", 28, "bold"))
        self.result.pack()
        self.confidence = tk.Label(root, text="", font=("Arial", 16))
        self.confidence.pack()
        self.others = tk.Label(root, text="", font=("Consolas", 10), fg="gray")
        self.others.pack()
        tk.Button(root, text="Clear", command=self.clear).pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.predict())

    def paint(self, e):
        x0, y0, x1, y1 = e.x - BRUSH, e.y - BRUSH, e.x + BRUSH, e.y + BRUSH
        self.canvas.create_oval(x0, y0, x1, y1, fill="white", outline="white")
        self.draw.ellipse((x0, y0, x1, y1), fill=255)

    def predict(self):
        r = recognize(self.model, self.image)
        if r is None:
            return
        digit, conf, probs = r
        color = "green" if conf >= 0.9 else "orange" if conf >= 0.6 else "red"
        self.result.config(text=f"Result: {digit}")
        self.confidence.config(text=f"Confidence: {conf:.1%}", fg=color)
        runners = sorted(range(10), key=lambda d: -probs[d])[1:3]
        self.others.config(text="Next: " + ", ".join(f"{d} ({probs[d]:.1%})" for d in runners))
        print(f"Result: {digit}  Confidence: {conf:.1%}")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, CANVAS, CANVAS), fill=0)
        self.result.config(text="Draw a digit")
        self.confidence.config(text="")
        self.others.config(text="")


if __name__ == "__main__":
    model = load_model()
    root = tk.Tk()
    root.title("Digit Recognizer")
    App(root, model)
    root.mainloop()
