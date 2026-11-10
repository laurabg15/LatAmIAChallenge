import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # guarda las figuras en archivo sin abrir ventanas
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.flop_counter import FlopCounterMode
from src.dataset import N_FRAME_FEATURES


def evaluate_predictions(y_true, y_pred, num_classes=5):
    """
    Calcula Accuracy top-1 y F1 macro.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', labels=list(range(num_classes)), zero_division=0)
    return acc, f1


def count_parameters(model):
    """
    Parámetros del modelo.
    PyTorch: pesos entrenables. Random Forest: por nodo interno umbral + variable, por hoja un valor por clase.
    Dummy: no aprende parámetros.
    """
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not hasattr(model, "estimators_"):
        return 0
    total = 0
    for tree in model.estimators_:
        internal = int((tree.tree_.children_left != -1).sum())
        leaves = tree.tree_.node_count - internal
        total += internal * 2 + leaves * len(model.classes_)
    return total


def preprocessing_flops(n_frames):
    """
    Estimación simple de FLOPs del preprocesamiento por muestra (src/dataset.py, sin aumentación):
    máscara de faltantes, normalización por hombros, forma de mano y resumen en el tiempo.
    """
    per_frame = 55 * 5 + 55 * 2 * 2 + 2 * 20 * 2 + N_FRAME_FEATURES * 6
    return n_frames * per_frame


def estimate_gflops(model, example_input, n_frames=55):
    """
    GFLOPs por muestra con batch 1, incluyendo el preprocesamiento.
    PyTorch: contador oficial (FlopCounterMode). Random Forest: comparaciones recorridas + votos.
    n_frames: longitud del video crudo usada para el preprocesamiento (55 = mediana del train).
    """
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(example_input)
        model_flops = counter.get_total_flops()
        # FlopCounterMode no cuenta las LSTM: 4 compuertas x 2 x (entrada + oculta) x oculta por frame
        for m in model.modules():
            if isinstance(m, torch.nn.LSTM):
                model_flops += 4 * 2 * (m.input_size + m.hidden_size) * m.hidden_size * example_input.shape[1]
    elif hasattr(model, "estimators_"):
        nodes = model.decision_path(example_input)[0].sum()
        model_flops = int(nodes) + len(model.estimators_) * len(model.classes_)
    else:
        model_flops = 0  # Dummy: no hace cálculos
    return (model_flops + preprocessing_flops(n_frames)) / 1e9


def plot_confusion_matrices(confusions, class_labels, model_name, path):
    """
    Guarda una figura con la matriz de confusión de cada fold LOSO.
    Args:
        confusions (dict): {firmante de validación: matriz (clases x clases)}.
        class_labels (list): Etiquetas de los ejes, p. ej. "sign_01\\nCafé".
    """
    fig, axes = plt.subplots(1, len(confusions), figsize=(6 * len(confusions), 5.5))
    for ax, (signer, cm) in zip(np.atleast_1d(axes), confusions.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(len(class_labels)), class_labels, rotation=90)
        ax.set_yticks(range(len(class_labels)), class_labels)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"Fold {signer}")
    fig.suptitle(f"Matrices de confusión LOSO: {model_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
