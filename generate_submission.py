import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from src.dataset import load_and_normalize, to_features

# Rutas por defecto (ajustar cuando los organizadores publiquen el test)
TEST_DIR = "data/test/landmarks"
TEMPLATE_PATH = "data/test/submission_template.csv"
MODEL_PATH = "models/best_model.joblib"
OUTPUT_PATH = "submission.csv"


def generate_and_validate_submission(test_dir=TEST_DIR, model_path=MODEL_PATH,
                                     output_path=OUTPUT_PATH, template_path=TEMPLATE_PATH):
    """
    Genera submission.csv con el modelo entrenado por train.py y lo valida contra el template oficial.
    Cada muestra se procesa sola (batch 1) y sin aumentación, igual que en la medición de GFLOPs.
    """
    # 1. Cargar el modelo final
    bundle = joblib.load(model_path)
    model, mode, classes = bundle["model"], bundle["mode"], bundle["classes"]
    print(f"Modelo cargado: {bundle['name']} (entrada: {mode})")

    # 2. Predecir cada archivo de test
    files = sorted(f for f in os.listdir(test_dir) if f.endswith('.npy'))
    video_ids, predictions = [], []
    for file_name in files:
        x = to_features(*load_and_normalize(np.load(os.path.join(test_dir, file_name))), mode)[None]
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                pred = int(model(torch.tensor(x)).argmax(dim=1)[0])
        else:
            pred = int(model.predict(x)[0])
        video_ids.append(file_name.replace('.npy', ''))
        predictions.append(classes[pred])

    df_sub = pd.DataFrame({'video_id': video_ids, 'predicted_sign': predictions})

    # 3. Mismo orden que el template (si existe)
    if os.path.exists(template_path):
        order = pd.read_csv(template_path, dtype=str)['video_id']
        df_sub = df_sub.set_index('video_id').reindex(order).reset_index()

    # 4. Guardar: UTF-8 sin BOM, sin índice de pandas
    df_sub.to_csv(output_path, index=False, encoding='utf-8')
    print(f"{len(df_sub)} predicciones guardadas en: {output_path}")

    # 5. Validación
    errors = []
    if list(df_sub.columns) != ['video_id', 'predicted_sign']:
        errors.append(f"Columnas {list(df_sub.columns)}; se esperaban ['video_id', 'predicted_sign']")
    if df_sub['predicted_sign'].isna().any():
        errors.append("Hay IDs del template sin predicción (falta el .npy en la carpeta de test)")
    if df_sub['video_id'].duplicated().any():
        errors.append("Hay video_id duplicados")
    if df_sub['video_id'].str.contains(r'\.(?:mp4|npy)$').any():
        errors.append("Algunos video_id incluyen la extensión .mp4/.npy")
    if not df_sub['predicted_sign'].dropna().isin(classes).all():
        errors.append(f"Hay etiquetas fuera de {classes}")
    if os.path.exists(template_path):
        df_temp = pd.read_csv(template_path, dtype=str)
        if set(df_sub['video_id']) != set(df_temp['video_id']) or len(df_sub) != len(df_temp):
            errors.append("Los IDs del submission no coinciden con los del template")
    else:
        print(f"Aviso: no se encontró {template_path}; se validó solo el formato.")

    if errors:
        print("\n¡SUBMISSION NO VÁLIDO!")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print("¡Validación de submission exitosa!")
    print(df_sub['predicted_sign'].value_counts().sort_index().to_string())


if __name__ == "__main__":
    generate_and_validate_submission()
