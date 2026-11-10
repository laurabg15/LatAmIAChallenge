import numpy as np

class LeaveOneSignerOutValidation:
    def __init__(self, signers_list):
        """
        Gestiona la validación Leave-One-Signer-Out (LOSO) con 3 folds (A, B y C).
        Args:
            signers_list (list o np.array): Identificador del firmante correspondiente a cada muestra.
        """
        self.signers = np.array(signers_list)
        self.unique_signers = np.unique(signers_list)  # Deberían ser los firmantes A, B, C

    def get_folds(self):
        """
        Genera los índices de entrenamiento y validación para cada fold LOSO.
        Returns:
            list: Lista de tuplas (train_idx, val_idx) para cada fold.
        """
        folds = []
        for test_signer in self.unique_signers:
            # El firmante actual es el conjunto de prueba (val)
            val_idx = np.where(self.signers == test_signer)[0]
            # El resto de firmantes forman el conjunto de entrenamiento
            train_idx = np.where(self.signers != test_signer)[0]
            # Comprobación anti-fuga: ningún firmante a ambos lados
            assert not set(self.signers[train_idx]) & set(self.signers[val_idx])

            folds.append({
                'test_signer': test_signer,
                'train_idx': train_idx,
                'val_idx': val_idx
            })
            
        return folds