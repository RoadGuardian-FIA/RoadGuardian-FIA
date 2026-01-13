"""
Modulo Factory per modelli di Machine Learning.

Implementa pattern Factory per la creazione di modelli DecisionTree e RandomForest
con interfaccia comune per training, predizione e valutazione.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import pickle
from typing import Tuple, Dict
import numpy as np


class ModelBase:
    """Classe base astratta per tutti i modelli di classificazione."""
    
    def __init__(self):
        """Inizializza il modello base con attributi comuni."""
        self.model = None
        self.model_name = "BaseModel"
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Scopo: Addestra il modello sui dati di training.

        Parametri:
        - X_train (np.ndarray): Matrice delle feature di training (n_samples, n_features).
        - y_train (np.ndarray): Vettore delle etichette target (n_samples,).

        Valore di ritorno:
        - None: Il modello viene addestrato in-place.

        Eccezioni:
        - ValueError: Se il modello non è stato inizializzato.
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Scopo: Genera predizioni per nuovi dati.

        Parametri:
        - X (np.ndarray): Matrice delle feature da predire (n_samples, n_features).

        Valore di ritorno:
        - np.ndarray: Vettore delle etichette predette (n_samples,).

        Eccezioni:
        - ValueError: Se il modello non è stato addestrato.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Scopo: Valuta le performance del modello su dati di test.

        Parametri:
        - X_test (np.ndarray): Matrice delle feature di test.
        - y_test (np.ndarray): Vettore delle etichette reali.

        Valore di ritorno:
        - Dict[str, float]: Dizionario con metriche {precision, recall, f1_score}.

        Eccezioni:
        - Exception: Eventuali errori propagati da predict().
        """
        y_pred = self.predict(X_test)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        }
    
    def save(self, filepath: str):
        """
        Scopo: Serializza e salva il modello su file usando pickle.

        Parametri:
        - filepath (str): Percorso completo del file di output (.pkl).

        Valore di ritorno:
        - None

        Eccezioni:
        - IOError: Se il file non può essere scritto.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filepath: str):
        """
        Scopo: Carica un modello precedentemente salvato da file.

        Parametri:
        - filepath (str): Percorso completo del file pickle (.pkl).

        Valore di ritorno:
        - None: Il modello viene caricato in self.model.

        Eccezioni:
        - FileNotFoundError: Se il file non esiste.
        - pickle.UnpicklingError: Se il file è corrotto.
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)


pesi_classi_incidenti = {
    'EMERGENZA_CRITICA': 100.0,           # Codice Rosso: feriti gravi/incendio. Errore fatale.
    'ARRESTO_OBBLIGATORIO': 85.0,        # Strada bloccata: rischio di tamponamento a catena.
    'GESTIONE_SINISTRO': 60.0,           # Protocollo standard: veicoli fermi ma area delimitata.
    'ZONA_RESIDENZIALE_CRITICA': 50.0,   # Incidente con rischio pedoni/folla sul luogo.
    'PRUDENZA_INTERSEZIONE': 45.0,       # Incidente in incrocio: gestione flussi trasversali.
    'GUIDA_NOTTURNA_EXTRAURBANA': 30.0    # Incidente in zone buie: rischio visibilità per chi sopraggiunge.
}
class DecisionTreeModel(ModelBase):
    """Implementazione del modello Decision Tree per classificazione incidenti."""
    
    def __init__(self, max_depth: int = 10, random_state: int = 42, class_weight: dict = pesi_classi_incidenti):
        """
        Scopo: Inizializza un classificatore Decision Tree con parametri configurabili.

        Parametri:
        - max_depth (int): Profondità massima dell'albero (default: 10).
        - random_state (int): Seed per riproducibilità (default: 42).
        - class_weight (str): Strategia pesi classi ('balanced' o None).

        Valore di ritorno:
        - None

        Eccezioni:
        - Nessuna eccezione prevista.
        """
        super().__init__()
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight
        )
        self.model_name = "DecisionTree"


class RandomForestModel(ModelBase):
    """Implementazione del modello Random Forest per classificazione incidenti."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42, class_weight: dict = pesi_classi_incidenti):
        """
        Scopo: Inizializza un classificatore Random Forest con parametri configurabili.

        Parametri:
        - n_estimators (int): Numero di alberi nella foresta (default: 100).
        - max_depth (int): Profondità massima di ogni albero (default: 10).
        - random_state (int): Seed per riproducibilità (default: 42).
        - class_weight (str): Strategia pesi classi ('balanced' o None).

        Valore di ritorno:
        - None

        Eccezioni:
        - Nessuna eccezione prevista.
        """
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight
        )
        self.model_name = "RandomForest"


def get_model(model_type: str) -> ModelBase:
    """
    Scopo: Factory function per ottenere un'istanza del modello richiesto.

    Parametri:
    - model_type (str): Tipo di modello ('decision_tree' o 'random_forest').

    Valore di ritorno:
    - ModelBase: Istanza del modello richiesto (DecisionTreeModel o RandomForestModel).

    Eccezioni:
    - ValueError: Se il tipo di modello non è supportato.
    """
    models = {
        'decision_tree': DecisionTreeModel,
        'random_forest': RandomForestModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type]()