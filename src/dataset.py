import torch
from torch.utils.data import Dataset
import numpy as np

# ---------------------------------------------------------------------------
# Formato del .npy crudo (frames, 1662) de MediaPipe Holistic:
#   pose 33 x (x, y, z, visibilidad) -> columnas 0-131
#   rostro 468 x (x, y, z)           -> columnas 132-1535 (no se usa)
#   mano izquierda 21 x (x, y, z)    -> columnas 1536-1598
#   mano derecha 21 x (x, y, z)      -> columnas 1599-1661
# ---------------------------------------------------------------------------

# 55 puntos seleccionados: 13 de la pose (nariz, hombros, codos, muñecas, dedos) + 21 de cada mano
POSE_POINTS = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
COLS_X = [p * 4 for p in POSE_POINTS] + [1536 + j * 3 for j in range(21)] + [1599 + j * 3 for j in range(21)]
L_SHOULDER, R_SHOULDER = 1, 2          # posición de los hombros dentro de los 55 puntos
L_WRIST, R_WRIST = 5, 6                # posición de las muñecas de la pose (puntos 15 y 16)
LHAND = slice(13, 34)                  # mano izquierda dentro de los 55 puntos
RHAND = slice(34, 55)                  # mano derecha dentro de los 55 puntos

# Reflejo: intercambia los pares izquierda/derecha de la pose y las dos manos
MIRROR_ORDER = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11] + list(range(34, 55)) + list(range(13, 34))

# Recorte al tramo activo (sin el reposo del inicio y del final)
REST_RISE = 0.3        # anchos de hombro que debe subir una muñeca respecto a su reposo para contar como seña
MIN_ACTIVE_FRAMES = 3  # frames activos seguidos para considerar que la seña empezó
CROP_MARGIN = 2        # frames de margen antes y después del tramo

# Resumen en el tiempo
N_SEGMENTS = 4         # tramos iguales dentro del recorte (modelos de vector)
SEQ_LEN = 20           # frames de la secuencia remuestreada (modelos temporales); Mindlin (2021): 16 > 32
# Cortes globales del perfil de manos (% del video completo), por si se quieren comparar:
# SEGMENT_CUTS = [0, 10, 26, 36, 64, 74, 88, 100]

# Aumentación de datos (solo en entrenamiento)
MIRROR_PROB = 0.5      # probabilidad de reflejar (zurdo <-> diestro)
MAX_ROTATION = 10      # grados, alrededor del centro de los hombros
JITTER_STD = 0.02      # ruido en anchos de hombro, solo sobre puntos detectados
FRAME_DROP = 0.10      # fracción de frames eliminados al azar
POINT_DROP = 0.05      # fracción de puntos marcados como faltantes
HAND_DROP_PROB = 0.2   # probabilidad de ocultar una mano completa en algunos frames

N_FRAME_FEATURES = 55 * 2 + 2 * 20 * 2                                  # 110 posiciones + 80 forma de mano = 190
N_VECTOR_FEATURES = N_SEGMENTS * N_FRAME_FEATURES + N_FRAME_FEATURES + 2 * N_SEGMENTS  # 958
N_SEQUENCE_CHANNELS = N_FRAME_FEATURES + 2                              # 190 + presencia de cada mano = 192


def load_and_normalize(raw):
    """
    Pasos deterministas (se aplican igual a train y test).
    Args:
        raw: arreglo crudo (frames, 1662).
    Returns:
        xy (frames, 55, 2) normalizado y mask (frames, 55) con True = punto detectado.
    """
    # 1. Seleccionar los 55 puntos, usar x, y y marcar faltantes (MediaPipe escribe ceros)
    cols = np.array(COLS_X)
    xyz = np.stack([raw[:, cols], raw[:, cols + 1], raw[:, cols + 2]], axis=-1).astype(np.float32)
    mask = np.any(xyz != 0, axis=-1)
    xy = xyz[..., :2]

    # 2. Centrar en el punto medio de los hombros y escalar por el ancho de hombros
    both = mask[:, L_SHOULDER] & mask[:, R_SHOULDER]
    if both.any():
        ls, rs = xy[both, L_SHOULDER], xy[both, R_SHOULDER]
        center = ((ls + rs) / 2).mean(axis=0)
        width = max(float(np.linalg.norm(ls - rs, axis=-1).mean()), 1e-3)
    else:
        center, width = np.array([0.5, 0.5], np.float32), 1.0
    xy = (xy - center) / width
    xy[~mask] = 0  # los faltantes quedan en 0; siempre se consulta mask

    # 3. Recortar al tramo activo: frames donde alguna muñeca de la pose subió más de REST_RISE
    #    anchos de hombro respecto a su reposo (su posición más baja en el video, percentil 90 de y).
    #    Usa la pose (siempre detectada) y no la detección de manos, que en algunas cámaras ve las manos en reposo.
    wrists = [L_WRIST, R_WRIST]
    if mask[:, wrists].all(axis=0).any():  # sin muñecas detectadas, se usa el video completo
        wrists_y = np.where(mask[:, wrists], xy[:, wrists, 1], np.nan)
        rise = np.nanpercentile(wrists_y, 90, axis=0) - wrists_y
        active = (rise > REST_RISE).any(axis=1)
        sustained = np.convolve(active, np.ones(MIN_ACTIVE_FRAMES), mode="valid") == MIN_ACTIVE_FRAMES
        starts = np.flatnonzero(sustained)
        if len(starts) > 0:  # si no hay tramo activo, se usa el video completo
            first = max(starts[0] - CROP_MARGIN, 0)
            last = min(starts[-1] + MIN_ACTIVE_FRAMES + CROP_MARGIN, len(xy))
            xy, mask = xy[first:last], mask[first:last]
    return xy, mask


def augment(xy, mask):
    """Aumentación aleatoria para entrenamiento. Recibe y devuelve xy (frames, 55, 2) y mask (frames, 55)."""
    xy, mask = xy.copy(), mask.copy()

    # 1. Reflejo: x -> -x e intercambio de manos y lados de la pose (zurdo <-> diestro)
    if np.random.rand() < MIRROR_PROB:
        xy[..., 0] *= -1
        xy, mask = xy[:, MIRROR_ORDER], mask[:, MIRROR_ORDER]

    # 2. Rotación pequeña alrededor del centro de los hombros (el origen tras normalizar)
    angle = np.deg2rad(np.random.uniform(-MAX_ROTATION, MAX_ROTATION))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    xy = xy @ rotation.T

    # 3. Jitter: ruido gaussiano solo en los puntos detectados
    xy = xy + np.random.normal(0, JITTER_STD, xy.shape).astype(np.float32) * mask[..., None]

    # 4. Eliminación de frames al azar (se conservan al menos 4)
    keep = np.random.rand(len(xy)) > FRAME_DROP
    if keep.sum() >= 4:
        xy, mask = xy[keep], mask[keep]

    # 5. Eliminación de puntos al azar y, a veces, de una mano completa en algunos frames
    mask = mask & (np.random.rand(*mask.shape) > POINT_DROP)
    if np.random.rand() < HAND_DROP_PROB:
        hand = LHAND if np.random.rand() < 0.5 else RHAND
        frames = np.random.rand(len(mask)) < 0.3
        mask[np.ix_(frames, np.arange(hand.start, hand.stop))] = False

    xy[~mask] = 0
    return xy, mask


def to_features(xy, mask, mode="vector", temporal_jitter=False):
    """
    Convierte los puntos normalizados en la entrada del modelo.
    mode="vector":   vector fijo (958,) -> Dummy, Linear y Random Forest
    mode="sequence": secuencia (SEQ_LEN, 192) -> CNN 1D, GRU, LSTM y TCN
    temporal_jitter: solo en entrenamiento; elige al azar qué frame se toma dentro de cada
                     intervalo del remuestreo (muestreo temporal aleatorio, Mindlin 2021).
    """
    t = len(xy)

    # 1. Forma de cada mano: dedos relativos a la muñeca (punto 0 de la mano)
    lh_shape = xy[:, LHAND][:, 1:] - xy[:, LHAND][:, :1]
    rh_shape = xy[:, RHAND][:, 1:] - xy[:, RHAND][:, :1]
    lh_ok = mask[:, LHAND][:, 1:] & mask[:, LHAND][:, :1]
    rh_ok = mask[:, RHAND][:, 1:] & mask[:, RHAND][:, :1]

    # 2. Características por frame (190) y su máscara de válidos
    feats = np.concatenate([xy.reshape(t, -1), lh_shape.reshape(t, -1), rh_shape.reshape(t, -1)], axis=1)
    valid = np.concatenate([np.repeat(mask, 2, axis=1), np.repeat(lh_ok, 2, axis=1), np.repeat(rh_ok, 2, axis=1)], axis=1)
    feats = feats * valid
    presence = np.stack([mask[:, LHAND.start], mask[:, RHAND.start]], axis=1).astype(np.float32)

    if mode == "sequence":
        # 3a. Remuestrear a SEQ_LEN frames (vecino más cercano) y agregar la presencia de cada mano.
        #     Con temporal_jitter cada posición se mueve al azar hasta medio intervalo (misma duración, otros frames).
        pos = np.linspace(0, t - 1, SEQ_LEN)
        if temporal_jitter and t > 1:
            pos = pos + np.random.uniform(-0.5, 0.5, SEQ_LEN) * (t - 1) / max(SEQ_LEN - 1, 1)
        idx = np.clip(np.sort(pos), 0, t - 1).round().astype(int)
        return np.concatenate([feats[idx], presence[idx]], axis=1).astype(np.float32)

    # 3b. Media por tramo + desviación estándar global + presencia de manos por tramo (ignorando faltantes)
    seg_means, seg_presence = [], []
    for seg in np.array_split(np.arange(t), N_SEGMENTS):
        count = valid[seg].sum(axis=0)
        seg_means.append(feats[seg].sum(axis=0) / np.maximum(count, 1))
        seg_presence.append(presence[seg].mean(axis=0) if len(seg) > 0 else np.zeros(2, np.float32))
    count = np.maximum(valid.sum(axis=0), 1)
    mean = feats.sum(axis=0) / count
    std = np.sqrt((((feats - mean) * valid) ** 2).sum(axis=0) / count)
    return np.concatenate(seg_means + [std] + seg_presence).astype(np.float32)


class SignLanguageDataset(Dataset):
    def __init__(self, samples, labels, mode="vector", augment=False):
        """
        Dataset para la carga de secuencias de señas.
        Args:
            samples (list): Lista de tuplas (xy, mask) ya normalizadas con load_and_normalize.
                            Se cargan una sola vez en memoria para no releer el disco en cada época.
            labels (list): Etiquetas numéricas de las señas para cada muestra.
            mode (str): "vector" (Dummy, Linear, Random Forest) o "sequence" (CNN 1D, GRU, LSTM, TCN).
            augment (bool): True solo para entrenamiento.
        """
        self.samples = samples
        self.labels = labels
        self.mode = mode
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xy, mask = self.samples[idx]
        if self.augment:
            xy, mask = augment(xy, mask)
        features = to_features(xy, mask, self.mode, temporal_jitter=self.augment)
        return torch.tensor(features), torch.tensor(self.labels[idx], dtype=torch.long)
