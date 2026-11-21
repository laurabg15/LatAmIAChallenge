import os
import json
import time
import random
import argparse
import torch
import numpy as np
import pandas as pd
import joblib
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import confusion_matrix
from src.dataset import SignLanguageDataset, load_and_normalize
from src.models import MODEL_MODES, build_model
from src.validation import LeaveOneSignerOutValidation
from src.evaluate import evaluate_predictions, count_parameters, estimate_gflops, plot_confusion_matrices

# Configuración del entrenamiento (valores por defecto; los marcados con * se pueden cambiar
# por línea de comandos sin tocar el código, ver parse_args)
SEEDS = [42, 43, 44]  # * se promedian los resultados de las 3 semillas
DATA_DIR = "data/train/landmarks"  # *
MODEL_PATH = "models/best_model.joblib"
REPORTS_DIR = "reports"
MODELS = ["dummy", "linear", "random_forest", "cnn1d", "gru", "gru2", "lstm", "tcn"]  # *
EPOCHS = 50          # * modelos PyTorch
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RF_COPIES = 3        # modelos sklearn: veces que se recorre el train con aumentación (750 -> 2250)
CODECARBON_COUNTRY = "CHL"  # * país (ISO 3166 alfa-3) para la intensidad de carbono en modo offline

# Videos que se dejan fuera SOLO del entrenamiento (la validación siempre queda completa).
# Ej: ["S04_M04_V03"] (vista donde MediaPipe detecta las manos en reposo). Se probó y no mejoró: vacío.
EXCLUDE_FROM_TRAIN = []

SIGN_NAMES = {"sign_01": "Café", "sign_02": "Aromática", "sign_03": "Hola",
              "sign_04": "Qué necesitas", "sign_05": "Con gusto"}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Validación LOSO y entrenamiento del modelo final, con medición de energía (CodeCarbon).")
    ap.add_argument("--mode", choices=["loso", "final", "both"], default="both",
                    help="loso: solo validación | final: solo reentrenar con todos los firmantes | "
                         "both: validación y luego reentrenar el mejor (por defecto)")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS,
                    help="modelos a evaluar en el LOSO (por defecto, todos)")
    ap.add_argument("--final_model", choices=MODELS, default=None,
                    help="modelo a reentrenar con todos los firmantes; obligatorio con --mode final. "
                         "Si se omite en --mode both, se usa el mejor del LOSO")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                    help="semillas del LOSO; la primera se usa también en el modelo final")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="épocas de los modelos PyTorch")
    ap.add_argument("--tag", default="",
                    help="nombre del experimento: guarda en reports/<tag>/ y models/<tag>.joblib "
                         "para no sobrescribir los resultados oficiales")
    ap.add_argument("--data_dir", default=DATA_DIR, help="carpeta con los landmarks de entrenamiento")
    ap.add_argument("--country", default=CODECARBON_COUNTRY,
                    help="país ISO alfa-3 para CodeCarbon offline (ej: CHL, COL)")
    ap.add_argument("--online_tracker", action="store_true",
                    help="CodeCarbon online (requiere internet; por defecto offline)")
    ap.add_argument("--tracking_mode", choices=["machine", "process"], default="machine",
                    help="CodeCarbon: machine mide todo el equipo (PC propio); process solo este proceso "
                         "(usar en servidores compartidos como cronos)")
    args = ap.parse_args()
    if args.mode == "final" and args.final_model is None:
        ap.error("--mode final requiere --final_model (no hay LOSO para elegir el mejor)")
    return args


def start_tracker(project_name, output_dir, online, country, tracking_mode="machine"):
    """Inicia CodeCarbon. Offline por defecto (el clúster puede no tener internet).
    Cada medición se agrega como una fila en <output_dir>/emissions.csv.
    tracking_mode="process" mide solo este proceso: en un servidor compartido, "machine" sumaría
    las CPU y la RAM de todos los usuarios. allow_multiple_runs evita que el lock de /tmp de otro
    usuario con CodeCarbon bloquee esta medición."""
    os.makedirs(output_dir, exist_ok=True)
    kwargs = dict(project_name=project_name, output_dir=output_dir, output_file="emissions.csv",
                  measure_power_secs=5, log_level="error", tracking_mode=tracking_mode,
                  allow_multiple_runs=True)
    tracker = EmissionsTracker(**kwargs) if online else OfflineEmissionsTracker(country_iso_code=country, **kwargs)
    tracker.start()
    return tracker, time.time()


def stop_tracker(tracker, t0):
    """Detiene CodeCarbon y devuelve emisiones (kg CO2eq), energía (kWh) y duración (s)."""
    emissions = tracker.stop()
    data = getattr(tracker, "final_emissions_data", None)
    return {"emissions_kgCO2eq": emissions, "energy_kwh": getattr(data, "energy_consumed", None),
            "seconds": time.time() - t0}


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def limit_threads_to_slurm():
    """En Slurm, usa solo las CPU asignadas al trabajo. Las librerías ven todos los hilos de la
    máquina (80 en cronos) y, sin límite, crearían más hilos que CPU disponibles.
    Fuera de Slurm no hace nada: PyTorch elige sus hilos como siempre."""
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus:
        torch.set_num_threads(int(cpus))
    return torch.get_num_threads()


def set_seed(seed):
    """Fija las semillas para garantizar reproducibilidad (requisito de Fase 1)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(name, train_dataset, num_classes, seed, epochs=EPOCHS):
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
    for _ in range(epochs):
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
    args = parse_args()
    seeds = args.seeds
    reports_dir = os.path.join(REPORTS_DIR, args.tag) if args.tag else REPORTS_DIR
    model_path = os.path.join("models", f"{args.tag}.joblib") if args.tag else MODEL_PATH
    run_name = args.tag or "run"

    # 1. Fijar semillas obligatorias de la Fase 1 y limitar hilos a las CPU de Slurm (si aplica)
    set_seed(seeds[0])
    n_threads = limit_threads_to_slurm()
    print(f"Semillas: {seeds} | modo: {args.mode} | modelos LOSO: {args.models} | épocas: {args.epochs} | "
          f"hilos PyTorch: {n_threads} | CodeCarbon: {args.tracking_mode}")

    # 2. Escanear la carpeta de landmarks reales y armar las listas
    sign_classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])

    data_paths = []
    labels = []
    signers = []  # Necesario para el LOSO

    for label_idx, sign_name in enumerate(sign_classes):
        sign_folder = os.path.join(args.data_dir, sign_name)
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

    results = []
    if args.mode in ("loso", "both"):
        tracker, t0 = start_tracker(f"loso_{run_name}", reports_dir, args.online_tracker, args.country,
                                    args.tracking_mode)

        # 4. Inicializar Validación Leave-One-Signer-Out (LOSO) con los firmantes reales
        loso = LeaveOneSignerOutValidation(signers)
        folds = loso.get_folds()
        print(f"Total de folds LOSO generados: {len(folds)}")

        # 5. Entrenar y evaluar cada modelo en cada fold y cada semilla
        os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)
        class_labels = [f"{c}\n{SIGN_NAMES.get(c, '')}" for c in sign_classes]
        fold_rows = []
        for name in args.models:
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
                for seed in seeds:
                    model = train_model(name, Subset(train_dataset, train_idx), len(sign_classes), seed, args.epochs)
                    y_pred = predict(model, Subset(eval_dataset, val_idx))
                    acc, f1 = evaluate_predictions(labels[val_idx], y_pred, len(sign_classes))
                    accs.append(acc)
                    f1s.append(f1)
                    fold_rows.append({"modelo": name, "fold": signer, "semilla": seed, "accuracy": acc, "f1_macro": f1})
                    if seed == seeds[0]:  # matriz de confusión con la primera semilla
                        confusions[signer] = confusion_matrix(labels[val_idx], y_pred, labels=range(len(sign_classes)))

                print(f"  Fold {signer} (train {len(train_idx)}, val {len(val_idx)}): "
                      f"accuracy {' / '.join(f'{a:.3f}' for a in accs)} (media {np.mean(accs):.3f}) | "
                      f"F1 macro media {np.mean(f1s):.3f}")

            plot_confusion_matrices(confusions, class_labels, name,
                                    os.path.join(reports_dir, "figures", f"confusion_{name}.png"))

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
        print(f"\n=== Resultados LOSO (media de {len(folds)} folds x {len(seeds)} semillas) ===")
        print(f"{'Modelo':<15}{'Accuracy':>10}{'F1 macro':>10}{'GFLOPs':>12}{'Parámetros':>12}")
        for r in results:
            print(f"{r['modelo']:<15}{r['accuracy']:>10.3f}{r['f1_macro']:>10.3f}"
                  f"{r['gflops']:>12.6f}{r['parametros']:>12,}")
        pd.DataFrame(fold_rows).to_csv(os.path.join(reports_dir, "resultados_por_fold.csv"), index=False)
        pd.DataFrame(results).to_csv(os.path.join(reports_dir, "resultados_resumen.csv"), index=False)

        loso_energy = stop_tracker(tracker, t0)
        save_json({"args": vars(args), "energy": loso_energy, "results": results},
                  os.path.join(reports_dir, "loso_info.json"))
        print(f"\nResultados guardados en {reports_dir}/ (CSV, loso_info.json y matrices de confusión en figures/)")
        print(f"Energía LOSO (CodeCarbon): {loso_energy}")

    # 7. Reentrenar con todos los firmantes y guardar. Modelo: --final_model o, si no se indica,
    #    el mejor del LOSO (accuracy, luego F1, luego menos GFLOPs)
    if args.mode in ("final", "both"):
        if args.final_model:
            final_name = args.final_model
        else:
            final_name = max(results, key=lambda r: (round(r["accuracy"], 4), round(r["f1_macro"], 4),
                                                     -r["gflops"]))["modelo"]
        print(f"\nModelo final: {final_name}. Reentrenando con los {len(set(signers))} firmantes...")
        mode = MODEL_MODES[final_name]
        all_train_idx = np.flatnonzero(~excluded)

        tracker, t0 = start_tracker(f"final_{final_name}_{run_name}", reports_dir, args.online_tracker, args.country,
                                    args.tracking_mode)
        final_model = train_model(final_name,
                                  Subset(SignLanguageDataset(samples, labels, mode=mode, augment=True), all_train_idx),
                                  len(sign_classes), seeds[0], args.epochs)
        final_energy = stop_tracker(tracker, t0)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({"name": final_name, "mode": mode, "classes": sign_classes, "model": final_model}, model_path)
        save_json({"model": final_name, "model_path": model_path, "energy": final_energy, "args": vars(args)},
                  os.path.join(reports_dir, "final_info.json"))
        print(f"Energía entrenamiento final (CodeCarbon): {final_energy}")
        print(f"Modelo final guardado en {model_path}")


if __name__ == "__main__":
    main()
