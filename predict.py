"""Genera submission.csv con el modelo congelado (models/*.joblib de train.py) y lo valida contra el template.
  python predict.py --model models/final.joblib --test_dir data/test/landmarks --template submission_template.csv
"""
import argparse, glob, os
import joblib, numpy as np, pandas as pd, torch
from src.dataset import load_and_normalize, to_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/best_model.joblib")
    ap.add_argument("--test_dir", default="data/test/landmarks")
    ap.add_argument("--template", default="submission_template.csv")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--id_col", default=None, help="columna con el id (por defecto la 1ª del template)")
    ap.add_argument("--pred_col", default=None, help="columna a rellenar (por defecto la 2ª)")
    ap.add_argument("--label_format", choices=["name", "index"], default="name", help="name: 'sign_01' | index: 0..4")
    a = ap.parse_args()

    ck = joblib.load(a.model)
    model, mode, classes = ck["model"], ck["mode"], ck["classes"]
    tpl = pd.read_csv(a.template)
    id_col, pred_col = a.id_col or tpl.columns[0], a.pred_col or tpl.columns[1]
    files = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(a.test_dir, "**", "*.npy"), recursive=True)}
    keys = [os.path.splitext(os.path.basename(str(i)))[0] for i in tpl[id_col]]
    missing = [k for k in keys if k not in files]
    assert not missing, f"{len(missing)} ids del template sin landmarks, p.ej. {missing[:3]}"

    # mismo preprocesamiento que en entrenamiento, sin augmentation
    X = np.stack([to_features(*load_and_normalize(np.load(files[k])), mode) for k in keys])
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X)).argmax(1).numpy()
    else:
        preds = model.predict(X)

    out = tpl.copy()
    out[pred_col] = [classes[p] if a.label_format == "name" else int(p) for p in preds]
    out.to_csv(a.out, index=False)

    chk = pd.read_csv(a.out)
    assert list(chk.columns) == list(tpl.columns), "columnas distintas al template"
    assert len(chk) == len(tpl) and (chk[id_col].astype(str) == tpl[id_col].astype(str)).all(), "ids/orden distintos"
    assert chk[pred_col].notna().all(), "predicciones vacías"
    print(f"OK: {a.out} ({len(chk)} filas)\n{chk[pred_col].value_counts().to_string()}")


if __name__ == "__main__":
    main()
