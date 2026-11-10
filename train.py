import os
import torch
import numpy as np
import pandas as pd
import random
import joblib
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import confusion_matrix
from src.dataset import SignLanguageDataset, load_and_normalize
from src.models import MODEL_MODES, build_model
from src.validation import LeaveOneSignerOutValidation
from src.evaluate import evaluate_predictions, count_parameters, estimate_gflops, plot_confusion_matrices

# Configuración del entrenamiento
SEEDS = [42, 43, 44]  # se promedian los resultados de las 3 semillas
DATA_DIR = "data/train/landmarks"
MODEL_PATH = "models/best_model.joblib"
REPORTS_DIR = "reports"
MODELS = ["dummy", "linear", "random_forest", "cnn1d", "gru", "gru2", "lstm", "tcn"]
EPOCHS = 50          # modelos PyTorch
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RF_COPIES = 3        # modelos sklearn: veces que se recorre el train con aumentación (750 -> 2250)

# Videos que se dejan fuera SOLO del entrenamiento (la validación siempre queda completa).
# Ej: ["S04_M04_V03"] (vista donde MediaPipe detecta las manos en reposo). Se probó y no mejoró: vacío.
EXCLUDE_FROM_TRAIN = []

SIGN_NAMES = {"sign_01": "Café", "sign_02": "Aromática", "sign_03": "Hola",
              "sign_04": "Qué necesitas", "sign_05": "Con gusto"}


def set_seed(seed):
    """Fija las semillas para garantizar reproducibilidad (requisito de Fase 1)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(name, train_dataset, num_classes, seed):
    """Entrena un modelo nuevo con el dataset de entrenamiento (con aumentación)."""
    set_seed(seed)
    model = build_model(name, num_classes, seed)

    if not isinstance(model, torch.nn.Module):
        # sklearn, sin épocas: se generan RF_COPIES versiones aumentadas de cada muestra y se entrena una vez
        pairs = [train_dataset[i] for _ in range(RF_COPIES) for i in range(len(train_dataset))]
        X = np.stack([x.numpy() for x, _ in pairs])
        y = np.array([int(label) for _, label in pairs])
        return model.fit(X, y)

    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(EPOCHS):
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def predict(model, dataset):
    """Predice las etiquetas de un dataset (sin aumentación)."""
    X = torch.stack([dataset[i][0] for i in range(len(dataset))])
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            return model(X).argmax(dim=1).numpy()
    return model.predict(X.numpy())


def main():
    # 1. Fijar semillas obligatorias de la Fase 1
    set_seed(SEEDS[0])
    print(f"Semillas: {SEEDS}")

    # 2. Escanear la carpeta de landmarks reales y armar las listas
    sign_classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

    data_paths = []
    labels = []
    signers = []  # Necesario para el LOSO

    for label_idx, sign_name in enumerate(sign_classes):
        sign_folder = os.path.join(DATA_DIR, sign_name)
        for file_name in sorted(os.listdir(sign_folder)):
            if file_name.endswith('.npy'):
                data_paths.append(os.path.join(sign_folder, file_name))
                labels.append(label_idx)

                # Formato del nombre: S{seña}_M{firmante}_V{variación}_C{cámara}_Clip{n}.npy
                # ej: 'S01_M02_V01_C01_Clip076.npy' -> firmante 'M02'
                # (el bloque S es la seña, no el firmante; usarlo rompería el LOSO)
                parts = file_name.split('_')
                if len(parts) < 2 or not parts[1].startswith('M'):
                    raise ValueError(f"Nombre de archivo inesperado, no se pudo extraer el firmante: {file_name}")
                signers.append(parts[1])

    print(f"Dataset real cargado. Total de muestras encontradas: {len(data_paths)}")

    # Videos excluidos del entrenamiento
    excluded = np.array([any(os.path.basename(p).startswith(e) for e in EXCLUDE_FROM_TRAIN) for p in data_paths])
    print(f"Excluidos del entrenamiento: {excluded.sum()} videos ({EXCLUDE_FROM_TRAIN})")

    # 3. Preprocesamiento determinista (recorte, selección de puntos, normalización), una sola vez
    samples = [load_and_normalize(np.load(p)) for p in data_paths]
    labels = np.array(labels)

    # 4. Inicializar Validación Leave-One-Signer-Out (LOSO) con los firmantes reales
    loso = LeaveOneSignerOutValidation(signers)
    folds = loso.get_folds()
    print(f"Total de folds LOSO generados: {len(folds)}")

    # 5. Entrenar y evaluar cada modelo en cada fold y cada semilla
    os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)
    class_labels = [f"{c}\n{SIGN_NAMES.get(c, '')}" for c in sign_classes]
    fold_rows, results = [], []
    for name in MODELS:
        mode = MODEL_MODES[name]
        train_dataset = SignLanguageDataset(samples, labels, mode=mode, augment=True)
        eval_dataset = SignLanguageDataset(samples, labels, mode=mode, augment=False)
        print(f"\n=== Modelo: {name} (entrada: {mode}) ===")

        confusions = {}
        for fold_info in folds:
            signer = fold_info['test_signer']
            train_idx = fold_info['train_idx'][~excluded[fold_info['train_idx']]]
            val_idx = fold_info['val_idx']

            accs, f1s = [], []
            for seed in SEEDS:
                model = train_model(name, Subset(train_dataset, train_idx), len(sign_classes), seed)
                y_pred = predict(model, Subset(eval_dataset, val_idx))
                acc, f1 = evaluate_predictions(labels[val_idx], y_pred, len(sign_classes))
                accs.append(acc)
                f1s.append(f1)
                fold_rows.append({"modelo": name, "fold": signer, "semilla": seed, "accuracy": acc, "f1_macro": f1})
                if seed == SEEDS[0]:  # matriz de confusión con la primera semilla
                    confusions[signer] = confusion_matrix(labels[val_idx], y_pred, labels=range(len(sign_classes)))

            print(f"  Fold {signer} (train {len(train_idx)}, val {len(val_idx)}): "
                  f"accuracy {' / '.join(f'{a:.3f}' for a in accs)} (media {np.mean(accs):.3f}) | "
                  f"F1 macro media {np.mean(f1s):.3f}")

        plot_confusion_matrices(confusions, class_labels, name,
                                os.path.join(REPORTS_DIR, "figures", f"confusion_{name}.png"))

        # Complejidad con batch 1 sobre el último modelo entrenado
        example = eval_dataset[0][0].unsqueeze(0)
        if not isinstance(model, torch.nn.Module):
            example = example.numpy()
        model_rows = [r for r in fold_rows if r["modelo"] == name]
        results.append({
            "modelo": name,
            "accuracy": np.mean([r["accuracy"] for r in model_rows]),
            "f1_macro": np.mean([r["f1_macro"] for r in model_rows]),
            "gflops": estimate_gflops(model, example),
            "parametros": count_parameters(model),
        })

    # 6. Tabla comparativa final (media de folds y semillas) y guardado de resultados
    print(f"\n=== Resultados LOSO (media de {len(folds)} folds x {len(SEEDS)} semillas) ===")
    print(f"{'Modelo':<15}{'Accuracy':>10}{'F1 macro':>10}{'GFLOPs':>12}{'Parámetros':>12}")
    for r in results:
        print(f"{r['modelo']:<15}{r['accuracy']:>10.3f}{r['f1_macro']:>10.3f}"
              f"{r['gflops']:>12.6f}{r['parametros']:>12,}")
    pd.DataFrame(fold_rows).to_csv(os.path.join(REPORTS_DIR, "resultados_por_fold.csv"), index=False)
    pd.DataFrame(results).to_csv(os.path.join(REPORTS_DIR, "resultados_resumen.csv"), index=False)
    print(f"\nResultados guardados en {REPORTS_DIR}/ (CSV y matrices de confusión en figures/)")

    # 7. Elegir el mejor (accuracy, luego F1, luego menos GFLOPs), reentrenar con todos los firmantes y guardar
    best = max(results, key=lambda r: (round(r["accuracy"], 4), round(r["f1_macro"], 4), -r["gflops"]))
    print(f"\nMejor modelo: {best['modelo']}. Reentrenando con los {len(set(signers))} firmantes...")
    mode = MODEL_MODES[best["modelo"]]
    all_train_idx = np.flatnonzero(~excluded)
    final_model = train_model(best["modelo"],
                              Subset(SignLanguageDataset(samples, labels, mode=mode, augment=True), all_train_idx),
                              len(sign_classes), SEEDS[0])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"name": best["modelo"], "mode": mode, "classes": sign_classes, "model": final_model}, MODEL_PATH)
    print(f"Modelo final guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
