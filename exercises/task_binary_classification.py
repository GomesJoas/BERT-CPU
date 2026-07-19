"""Exercise 05 — Binary classification on the Adult dataset (the baseline).

This is the **capstone**: the first exercise that runs the *whole* pipeline end
to end on real data. Everything the earlier files built in isolation — the
autograd engine, a `Linear` layer, an activation, an optimiser, a loss — is
assembled here into a working income classifier and **actually trained**.

    datasets.load_adult  ->  X, y (model-ready NumPy)
              |
         nn.Linear + relu + nn.Linear     (the model — a small MLP)
              |
         cross_entropy(logits, y)         (scalar loss; one graph back to W)
              |
         loss.backward()                  (fills every parameter's .grad)
              |
         optim.Adam(...).step()           (nudges each parameter downhill)

It is the *baseline*: complete, runnable, nothing to fill in. (The other
exercises, q01–q04, deliberately never train — they gradient-check a `forward`.
q05 is the sanctioned exception, because a classifier only means something once
it is trained.) A later pass will turn parts of this into student TODOs and add
guided questions and tests.

The whole Adult training set fits in memory, so there is no need for
mini-batches: each epoch is one **full-batch** step over *all* training rows at
once — the four-line cycle from the ``loss.py`` / ``optim.py`` docstrings::

    opt.zero_grad()                 # clear last step's gradients
    loss = cross_entropy(model(X).T, y)
    loss.backward()                 # one graph: loss -> ... -> every weight
    opt.step()                      # p.data -= lr * (adam update)

A slice of the training base is held out as a **validation** set. At the end of
each epoch we also measure the loss there (a forward pass only, no gradient
step): if the training loss keeps falling while the validation loss turns back
up, the model is starting to over-fit.

Conventions
-----------
Column-oriented like the rest of the library: features run down axis 0 and
samples across axis 1, so ``X`` is ``(n_features, n_samples)`` and ``nn.Linear``
consumes it directly. The model outputs logits ``(2, batch)``; we transpose to
``(batch, 2)`` for ``cross_entropy`` (whose class axis is last). Binary income
(``<=50K`` / ``>50K``) is framed as a **2-class** problem so the existing
``cross_entropy`` is reused unchanged.

HOW TO RUN
==========
    python -m exercises.q05_binary_classification
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import time

# Make the script runnable *directly* (``python exercises/q05_binary_classification.py``)
# as well as via ``python -m exercises.q05_binary_classification``. Running a file
# directly only puts its own folder (``exercises/``) on the import path, so we add
# the repo root so ``datasets`` and ``bert_cpu`` resolve either way.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
from bert_cpu import engine as cpu
from bert_cpu import nn
from bert_cpu import optim
from bert_cpu.loss import cross_entropy


# ============================================================================ #
# The model — a small multilayer perceptron
# ============================================================================ #
class AdultMLP(nn.Module):
    """MLP com função de ativação configurável."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        activation: str = "relu",
    ) -> None:

        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.activation = activation.lower()

    def forward(self, x: cpu.Tensor) -> cpu.Tensor:

        h = self.fc1(x)

        if self.activation == "relu":
            h = h.relu()

        elif self.activation == "gelu":
            h = h.gelu()

        elif self.activation == "tanh":
            h = h.tanh()

        else:
            raise ValueError(
                f"Ativação '{self.activation}' não suportada."
            )

        return self.fc2(h)


# ============================================================================ #
# Evaluation
# ============================================================================ #
def accuracy(model: AdultMLP, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of samples whose arg-max logit matches the label.

    A pure forward pass (no graph needed for a metric); ``model`` predicts the
    class with the larger logit for every column of ``X``.
    """
    logits = model(cpu.Tensor(X)).data          # (2, n_samples)
    preds = logits.argmax(axis=0)               # class per sample
    return float((preds == y).mean())


# ============================================================================ #
# Training
# ============================================================================ #
def train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2):
    """Carve a validation set out of the training base (column-oriented split).

    Shuffles the sample indices once and holds out ``val_frac`` of them for
    validation. Returns ``(X_tr, y_tr, X_val, y_val)``.
    """
    n = X.shape[1]                              # samples across axis 1
    perm = np.random.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return X[:, tr_idx], y[tr_idx], X[:, val_idx], y[val_idx]


def train(
    model: AdultMLP,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    activation: str,
    hidden: int,
    epochs: int = 100,
    lr: float = 1e-2,
):

    opt = optim.Adam(model.parameters(), lr=lr)

    Xt = cpu.Tensor(X_tr, requires_grad=False)
    Xv = cpu.Tensor(X_val, requires_grad=False)

    total_flops = 0

    history = []

    for epoch in range(1, epochs + 1):

        inicio = time.perf_counter()

        cpu.reset_flops()

        opt.zero_grad()

        loss = cross_entropy(model(Xt).T, y_tr)

        loss.backward()

        opt.step()

        train_loss = float(loss.data)

        val_loss = float(
            cross_entropy(model(Xv).T, y_val).data
        )

        epoch_flops = cpu.flop_count()

        total_flops += epoch_flops

        tempo = time.perf_counter() - inicio

        history.append(
            {
                "activation": activation,
                "hidden": hidden,
                "lr": lr,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_flops": epoch_flops,
                "total_flops": total_flops,
                "gflops": total_flops / 1e9,
                "epoch_time": tempo,
            }
        )

        print(
            f"Epoch {epoch:3d}/{epochs}"
            f" | train={train_loss:.4f}"
            f" | val={val_loss:.4f}"
            f" | FLOPs={epoch_flops:,}"
            f" | tempo={tempo:.3f}s"
        )

    gflops = total_flops / 1e9

    print(
        f"\nTotal FLOPs: {total_flops:,}"
        f" (~{gflops:.2f} GFLOPs)"
    )

    return gflops, history


def run_experiment(
    activation: str,
    hidden: int,
    lr: float,
    train_ds,
    test_ds,
):

    print("\n" + "=" * 70)
    print(
        f"EXPERIMENTO: "
        f"{activation.upper()} | "
        f"Hidden={hidden} | "
        f"LR={lr}"
    )
    print("=" * 70)

    cpu.set_seed(0)

    X_tr, y_tr, X_val, y_val = train_val_split(
        train_ds.X,
        train_ds.y,
        val_frac=0.2,
    )

    print(
        f"Train/val split: "
        f"{X_tr.shape[1]} train / "
        f"{X_val.shape[1]} val (20%)\n"
    )

    model = AdultMLP(
        train_ds.n_features,
        hidden=hidden,
        activation=activation,
    )

    print(
        f"Model: Linear({train_ds.n_features}, {hidden})"
        f" -> {activation.upper()} -> "
        f"Linear({hidden}, 2)"
    )

    gflops, history = train(
        model,
        X_tr,
        y_tr,
        X_val,
        y_val,
        activation=activation,
        hidden=hidden,
        epochs=100,
        lr=lr,
    )

    train_acc = accuracy(model, X_tr, y_tr)
    val_acc = accuracy(model, X_val, y_val)
    test_acc = accuracy(model, test_ds.X, test_ds.y)

    print(
        f"\nFinal accuracy "
        f"train={train_acc:.4f} "
        f"val={val_acc:.4f} "
        f"test={test_acc:.4f}"
    )

    return {

        "activation": activation,
        "hidden": hidden,
        "lr": lr,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "gflops": gflops,
        "history": history,
    }

# ============================================================================ #
# Entry point
# ============================================================================ #
def main() -> None:

    print("=" * 70)
    print("Adult income classification — end-to-end baseline on bert_cpu")
    print("=" * 70)

    cpu.set_seed(0)

    train_ds = datasets.load_adult("train")
    test_ds = datasets.load_adult("test")

    print(f"\nData: {train_ds}   {test_ds}")
    print(f"Features per sample: {train_ds.n_features}")

    results = []
    history = []

    # =====================================================
    # ESCOLHA O EXPERIMENTO
    # =====================================================

    experiments = [

        #
        # Q01
        # Activation Functions
        #

        {"activation": "relu", "hidden": 64, "lr": 0.01},
        {"activation": "gelu", "hidden": 64, "lr": 0.01},
        {"activation": "tanh", "hidden": 64, "lr": 0.01},

        #
        # Hidden Layer
        #

        #{"activation": "relu", "hidden": 32, "lr": 0.01},
        #{"activation": "relu", "hidden": 64, "lr": 0.01},
        #{"activation": "relu", "hidden": 128, "lr": 0.01},
        #{"activation": "relu", "hidden": 256, "lr": 0.01},

        #
        # Learning Rate
        #

        #{"activation": "relu", "hidden": 64, "lr": 0.001},
        #{"activation": "relu", "hidden": 64, "lr": 0.005},
        #{"activation": "relu", "hidden": 64, "lr": 0.010},
        #{"activation": "relu", "hidden": 64, "lr": 0.050},

    ]

    # =====================================================

    for exp in experiments:

        result = run_experiment(

            activation=exp["activation"],
            hidden=exp["hidden"],
            lr=exp["lr"],

            train_ds=train_ds,
            test_ds=test_ds,
        )

        results.append(result)

        history.extend(result["history"])

    # =====================================================

    df = pd.DataFrame(results)

    history_df = pd.DataFrame(history)

    df.drop(columns=["history"]).to_csv(
        "resultados.csv",
        index=False,
    )

    history_df.to_csv(
        "historico.csv",
        index=False,
    )

    print("\nResultados salvos em resultados.csv")

    print("Histórico salvo em historico.csv")

    print("\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)

    for r in results:

        print(

            f"{r['activation']:>8}"

            f" | H={r['hidden']:>3}"

            f" | LR={r['lr']:<6}"

            f" | train={r['train_acc']:.4f}"

            f" | val={r['val_acc']:.4f}"

            f" | test={r['test_acc']:.4f}"

            f" | GFLOPs={r['gflops']:.2f}"

        )
    # A majority-class baseline (always predict <=50K) scores ~0.76 on test;
    # this MLP should comfortably clear that, landing around 0.84–0.85.


if __name__ == "__main__":
    main()
