import os
import torch
import numpy as np
import pandas as pd
import random
import joblib
import argparse, json, time
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import confusion_matrix
from src.dataset import SignLanguageDataset, load_and_normalize
from src.models import MODEL_MODES, build_model
from src.validation import LeaveOneSignerOutValidation
from src.evaluate import evaluate_predictions, count_parameters, estimate_gflops, plot_confusion_matrices

# Configuración del entrenamiento (valores por defecto; se pueden cambiar por línea de comandos, ver parse_args)
SEEDS = [42, 43, 44]  # se promedian los resultados de las 3 semillas
DATA_DIR = "data/train/landmarks"
MODEL_PATH = "models/best_model.joblib"
REPORTS_DIR = "reports"
ALL_MODELS = ["dummy", "linear", "random_forest", "cnn1d", "gru", "gru2", "lstm", "tcn"]
MODELS = list(ALL_MODELS)
AUGMENT = True       # --no_aug lo desactiva (comparación con/sin augmentation)
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


def parse_args():
    ap = argparse.ArgumentParser(description="LOSO + entrenamiento final, con CodeCarbon")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--no_aug", action="store_true", help="entrenar sin augmentation")
    ap.add_argument("--mode", choices=["loso", "final", "both"], default="both",
                    help="loso: solo validación | final: solo reentrenar con todos | both: como antes")
    ap.add_argument("--final_model", default=None, help="modelo a congelar en --mode final (si no, el mejor LOSO)")
    ap.add_argument("--tag", default="", help="sufijo para no pisar salidas (ej: gru_aug)")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--online_tracker", action="store_true", help="CodeCarbon online (por defecto offline)")
    ap.add_argument("--country", default="CHL")
    return ap.parse_args()


def make_tracker(name, out_dir, online, country):
    """CodeCarbon. Offline por defecto (el clúster puede no tener internet)."""
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    os.makedirs(out_dir, exist_ok=True)
    kw = dict(project_name=name, output_dir=out_dir, output_file="emissions.csv",
              measure_power_secs=5, log_level="error")
    return EmissionsTracker(**kw) if online else OfflineEmissionsTracker(country_iso_code=country, **kw)


def energy_of(tracker, kg):
    d = getattr(tracker, "final_emissions_data", None)
    return {"emissions_kgCO2eq": kg, "energy_kwh": getattr(d, "energy_consumed", None)}


def main():
    global EPOCHS, SEEDS, MODELS, DATA_DIR, MODEL_PATH, REPORTS_DIR
    args = parse_args()
    EPOCHS, SEEDS, MODELS, DATA_DIR = args.epochs, args.seeds, args.models, args.data_dir
    use_aug = not args.no_aug
    if args.tag:  # salidas separadas por experimento
        REPORTS_DIR = os.path.join("reports", args.tag)
        MODEL_PATH = os.path.join("models", f"{args.tag}.joblib")

    # 1. Fijar semillas obligatorias de la Fase 1
    set_seed(SEEDS[0])
    print(f"Semillas: {SEEDS} | augmentation: {use_aug} | modo: {args.mode} | modelos: {MODELS}")

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

    results = []
    if args.mode in ("loso", "both"):
        tracker = make_tracker(f"loso_{args.tag or 'run'}", REPORTS_DIR, args.online_tracker, args.country)
        tracker.start(); t0 = time.time()
        # 5. Entrenar y evaluar cada modelo en cada fold y cada semilla
        os.makedirs(os.path.join(REPORTS_DIR, "figures"), exist_ok=True)
        class_labels = [f"{c}\n{SIGN_NAMES.get(c, '')}" for c in sign_classes]
        fold_rows, results = [], []
        for name in MODELS:
            mode = MODEL_MODES[name]
            train_dataset = SignLanguageDataset(samples, labels, mode=mode, augment=use_aug)
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
        loso_energy = {**energy_of(tracker, tracker.stop()), "seconds": time.time() - t0}
        json.dump({"args": vars(args), "energy": loso_energy, "results": results},
                  open(os.path.join(REPORTS_DIR, "loso_info.json"), "w"), indent=2, default=float)
        print(f"Energía LOSO (CodeCarbon): {loso_energy}")

    # 7. Reentrenar con todos los firmantes y guardar. Modelo: --final_model, o el mejor del LOSO
    #    (accuracy, luego F1, luego menos GFLOPs).
    if args.mode in ("final", "both"):
        if args.final_model:
            best_name = args.final_model
        else:
            assert results, "sin resultados LOSO: indica --final_model"
            best_name = max(results, key=lambda r: (round(r["accuracy"], 4), round(r["f1_macro"], 4), -r["gflops"]))["modelo"]
        print(f"\nModelo final: {best_name}. Reentrenando con los {len(set(signers))} firmantes...")
        mode = MODEL_MODES[best_name]
        all_train_idx = np.flatnonzero(~excluded)
        tracker = make_tracker(f"final_{best_name}_{args.tag or 'run'}", REPORTS_DIR, args.online_tracker, args.country)
        tracker.start(); t0 = time.time()
        final_model = train_model(best_name,
                                  Subset(SignLanguageDataset(samples, labels, mode=mode, augment=use_aug), all_train_idx),
                                  len(sign_classes), SEEDS[0])
        final_energy = {**energy_of(tracker, tracker.stop()), "seconds": time.time() - t0}
        print(f"Energía entrenamiento final (CodeCarbon): {final_energy}")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"name": best_name, "mode": mode, "classes": sign_classes, "model": final_model}, MODEL_PATH)
        json.dump({"model": best_name, "energy": final_energy, "args": vars(args)},
                  open(os.path.join(REPORTS_DIR, "final_info.json"), "w"), indent=2, default=float)
        print(f"Modelo final guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
