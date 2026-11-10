import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from src.dataset import N_VECTOR_FEATURES, N_SEQUENCE_CHANNELS

# Qué entrada usa cada modelo (ver src/dataset.py::to_features)
MODEL_MODES = {
    "dummy": "vector",          # modelo base: siempre predice la clase más frecuente
    "linear": "vector",
    "random_forest": "vector",
    "cnn1d": "sequence",
    "gru": "sequence",
    "lstm": "sequence",
    "tcn": "sequence",
}


class LinearSignModel(nn.Module):
    def __init__(self, num_features=N_VECTOR_FEATURES, num_classes=5):
        """
        Modelo lineal (equivalente a una regresión logística) sobre el vector resumido.
        BatchNorm normaliza la entrada con las estadísticas del entrenamiento de cada fold.
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # x: (batch, num_features)
        return self.fc(self.norm(x))


class CNN1DSignModel(nn.Module):
    def __init__(self, num_channels=N_SEQUENCE_CHANNELS, num_classes=5, hidden=64):
        """
        CNN 1D temporal: recorre los frames de la secuencia para captar la trayectoria de la seña.
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(num_channels)
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # promedio sobre el tiempo
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (batch, frames, canales) -> Conv1d espera (batch, canales, frames)
        x = self.norm(x.transpose(1, 2))
        x = self.conv(x).squeeze(-1)
        return self.fc(self.dropout(x))


class RecurrentSignModel(nn.Module):
    def __init__(self, cell="gru", num_channels=N_SEQUENCE_CHANNELS, num_classes=5, hidden=64):
        """
        Red recurrente (GRU o LSTM) de una capa: lee la seña frame a frame
        y clasifica con el estado oculto del último frame.
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(num_channels)
        rnn_class = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn_class(num_channels, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (batch, frames, canales); BatchNorm1d espera (batch, canales, frames)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1]))


class TCNSignModel(nn.Module):
    def __init__(self, num_channels=N_SEQUENCE_CHANNELS, num_classes=5, hidden=64, dilations=(1, 2, 4)):
        """
        TCN: CNN 1D más profunda, con bloques residuales y dilatación creciente
        (cada bloque mira una ventana de tiempo más amplia: 3, 5 y 9 frames).
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(num_channels)
        self.inp = nn.Conv1d(num_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=3, dilation=d, padding=d),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            )
            for d in dilations
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (batch, frames, canales) -> Conv1d espera (batch, canales, frames)
        x = self.inp(self.norm(x.transpose(1, 2)))
        for block in self.blocks:
            x = x + block(x)  # conexión residual
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x))


def build_model(name, num_classes=5, seed=42):
    """Crea un modelo nuevo a partir de su nombre."""
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "linear":
        return LinearSignModel(num_classes=num_classes)
    if name == "cnn1d":
        return CNN1DSignModel(num_classes=num_classes)
    if name in ("gru", "lstm"):
        return RecurrentSignModel(cell=name, num_classes=num_classes)
    if name == "tcn":
        return TCNSignModel(num_classes=num_classes)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=seed)
    raise ValueError(f"Modelo desconocido: {name}. Opciones: {list(MODEL_MODES)}")
