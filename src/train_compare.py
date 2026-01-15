"""
Modulo per addestramento e confronto modelli di Machine Learning.

Addestra modelli DecisionTree e RandomForest, li confronta e salva il migliore.
Supporta la gestione di dataset sbilanciati tramite class_weight o SMOTE.
Utilizza Stratified K-Fold Cross-Validation per una valutazione robusta.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# Aggiunge la directory padre al path per importazioni relative
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_factory import DecisionTreeModel, RandomForestModel

# Colonne delle feature categoriche
CATEGORICAL_COLUMNS = ['Severity', 'Incident_Type', 'Road_Type']

# Colonne delle feature booleane (condizioni stradali)
BOOLEAN_COLUMNS = [
    'Daylight', 'Bump', 'Crossing', 'Give_Way', 'Junction',
    'Railway', 'Roundabout', 'Stop', 'Traffic_Signal', 'Turning_Loop'
]

# Nome della colonna target (etichetta da predire)
TARGET_COLUMN = 'Target_Label'

# Strategia di gestione sbilanciamento: 'class_weight', 'smote', o 'none'
# Il dataset è già bilanciato (16.67% per classe), nessun riequilibrio necessario
IMBALANCE_STRATEGY = 'none'

# Impostazioni Cross-Validation
CV_FOLDS = 5  # Numero di fold per Stratified K-Fold Cross-Validation
USE_CROSS_VALIDATION = True  # Impostare a False per usare split hold-out semplice


def load_and_prepare_data(data_path: str):
    """
    Scopo: Carica il dataset CSV e prepara i dati per l'addestramento, 
           applicando encoding alle feature categoriche e conversione booleana.

    Parametri:
    - data_path (str): Percorso assoluto o relativo al file CSV del dataset.

    Valore di ritorno:
    - tuple: (X_final, y, label_encoders, feature_columns)
        - X_final (np.ndarray): Matrice delle feature processate.
        - y (np.ndarray): Vettore delle etichette target.
        - label_encoders (dict): Dizionario {colonna: LabelEncoder} per decodifica.
        - feature_columns (list): Lista ordinata delle colonne feature utilizzate.

    Eccezioni:
    - FileNotFoundError: Se il file CSV non esiste.
    - KeyError: Se manca la colonna target nel dataset.

    Feature attese in input:
    - Severity: {low, medium, high, Unknown}
    - Incident_Type: {Incendio Veicolo, Investimento, Veicolo Fuori Strada, Tamponamento, Collisione con ostacolo}
    - Road_Type: {motorway_trunk, primary_secondary, service, tertiary, residential, living_street, unclassified}
    - Daylight, Bump, Crossing, Give_Way, Junction, Railway, Roundabout, Stop, Traffic_Signal, Turning_Loop: {True, False}
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separa feature e target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    # Crea una copia per evitare modifiche all'originale
    X_processed = X.copy()
    
    # Encoding delle feature categoriche con LabelEncoder
    label_encoders = {}
    for column in CATEGORICAL_COLUMNS:
        if column in X_processed.columns:
            le = LabelEncoder()
            X_processed[column] = le.fit_transform(X_processed[column].astype(str))
            label_encoders[column] = le
    
    # Conversione feature booleane in valori numerici (0/1)
    for column in BOOLEAN_COLUMNS:
        if column in X_processed.columns:
            # Gestisce varie rappresentazioni booleane (True, 'True', 'true', 1, '1')
            X_processed[column] = X_processed[column].map(
                lambda x: 1 if x in [True, 'True', 'true', 1, '1'] else 0
            )
    
    # Seleziona solo le colonne attese nell'ordine corretto
    feature_columns = [col for col in CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS if col in X_processed.columns]
    X_final = X_processed[feature_columns].values
    
    print(f"Data loaded: {len(df)} samples, {X_final.shape[1]} features")
    print(f"Categorical features: {[col for col in CATEGORICAL_COLUMNS if col in X_processed.columns]}")
    print(f"Boolean features: {[col for col in BOOLEAN_COLUMNS if col in X_processed.columns]}")
    print(f"Target classes: {sorted(y.unique())}")
    
    return X_final, y.values, label_encoders, feature_columns


def train_and_compare(data_path: str, model_output_dir: str):
    """
    Scopo: Addestra DecisionTree e RandomForest, li confronta tramite Cross-Validation
           e metriche su test set, quindi salva il modello migliore con i suoi artefatti.

    Parametri:
    - data_path (str): Percorso al file CSV contenente il dataset di training.
    - model_output_dir (str): Directory dove salvare il modello e gli encoder.

    Valore di ritorno:
    - tuple: (best_model_name, metrics)
        - best_model_name (str): Nome del modello vincente ('DecisionTree' o 'RandomForest').
        - metrics (dict): Dizionario con precision, recall, f1_score del modello migliore.

    Eccezioni:
    - Exception: Eventuali errori propagati da load_and_prepare_data o sklearn.

    Note:
    - Utilizza Stratified K-Fold Cross-Validation per valutazione robusta.
    - Addestra il modello finale sull'intero training set (80% dei dati).
    - Salva: best_model.pkl, label_encoders.pkl, feature_columns.pkl.
    """
    
    # Carica e prepara i dati
    X, y, label_encoders, feature_columns = load_and_prepare_data(data_path)
    
    # Split dati: Test set hold-out (20%) + Training set per CV (80%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n{'='*60}")
    print("DATA SPLIT STRATEGY")
    print("="*60)
    print(f"Total samples: {len(X)}")
    print(f"Training set (for CV): {len(X_train_full)} samples (80%)")
    print(f"Hold-out test set: {len(X_test)} samples (20%)")
    
    if USE_CROSS_VALIDATION:
        print(f"\n✓ Using {CV_FOLDS}-Fold Stratified Cross-Validation")
        print(f"  Each fold: ~{len(X_train_full)//CV_FOLDS} samples for validation")
    else:
        print("\n⚠ Using simple hold-out split (no cross-validation)")
    
    # Stampa distribuzione delle classi nel training set
    unique, counts = np.unique(y_train_full, return_counts=True)
    print("\nClass distribution in training set:")
    for cls, cnt in zip(unique, counts):
        pct = (cnt / len(y_train_full)) * 100
        print(f"  {cls}: {cnt:,} ({pct:.2f}%)")
    
    # Gestione dataset sbilanciato in base alla strategia configurata
    use_class_weight = False
    X_train = X_train_full.copy()
    y_train = y_train_full.copy()
    
    if IMBALANCE_STRATEGY == 'smote':
        # Applica SMOTE per oversampling della classe minoritaria
        print("\n⚙ Applying SMOTE to balance classes...")
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train_full, y_train_full)
            
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"Training set after SMOTE: {len(X_train)} samples")
        except ImportError:
            print("⚠ imbalanced-learn not installed, falling back to class_weight='balanced'")
            use_class_weight = True
    elif IMBALANCE_STRATEGY == 'class_weight':
        # Usa pesi inversamente proporzionali alla frequenza delle classi
        print("\n⚙ Using class_weight='balanced' to handle imbalanced classes")
        use_class_weight = True
    else:
        print("\n✓ No imbalance handling needed (dataset is balanced)")
    
    # Inizializza i modelli con i parametri ottimali
    if use_class_weight:
        models = {
            'DecisionTree': DecisionTreeModel(max_depth=10, random_state=42, class_weight='balanced'),
            'RandomForest': RandomForestModel(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')
        }
    else:
        models = {
            'DecisionTree': DecisionTreeModel(max_depth=10, random_state=42),
            'RandomForest': RandomForestModel(n_estimators=200, max_depth=15, random_state=42)
        }
    
    results = {}
    
    # Valutazione tramite Cross-Validation stratificata
    if USE_CROSS_VALIDATION:
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION ({CV_FOLDS}-FOLD STRATIFIED)")
        print("="*60)
        
        # Configura K-Fold stratificato per mantenere proporzioni classi
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"\n{name}:")
            print("-" * 40)
            
            # Esegue cross-validation con F1-score weighted per multi-classe
            cv_scores = cross_val_score(
                model.model, X_train_full, y_train_full, 
                cv=skf, scoring='f1_weighted', n_jobs=-1
            )
            
            print(f"  F1-Score per fold: {[f'{s:.4f}' for s in cv_scores]}")
            print(f"  Mean F1-Score:     {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Min:  {cv_scores.min():.4f}  |  Max: {cv_scores.max():.4f}")
            
            results[name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
    
    # Addestramento finale e valutazione su test set hold-out
    print("\n" + "="*60)
    print("FINAL TRAINING AND HOLD-OUT TEST EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Addestra sull'intero training set
        print("Training on full training set...")
        model.train(X_train, y_train)
        
        # Valuta sul test set hold-out (mai visto durante training/CV)
        print("Evaluating on hold-out test set...")
        metrics = model.evaluate(X_test, y_test)
        
        # Ottiene predizioni per analisi dettagliata
        y_pred = model.predict(X_test)
        
        if name in results:
            results[name].update({
                'model': model,
                'metrics': metrics,
                'y_pred': y_pred
            })
        else:
            results[name] = {
                'model': model,
                'metrics': metrics,
                'y_pred': y_pred
            }
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        # Stampa report di classificazione per classe
        print(f"\n  Classification Report:")
        report = classification_report(y_test, y_pred, zero_division=0)
        for line in report.split('\n'):
            print(f"    {line}")
    
    # Confronta i modelli e seleziona il migliore basandosi su F1-score
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    best_model_name = None
    best_score = -1
    
    for name, result in results.items():
        print(f"\n{name}:")
        
        if USE_CROSS_VALIDATION and 'cv_mean' in result:
            print(f"  Cross-Validation F1: {result['cv_mean']:.4f} (+/- {result['cv_std']*2:.4f})")
        
        # Usa F1-score come metrica principale per la selezione
        score = result['metrics']['f1_score']
        print(f"  Hold-out Test F1:    {score:.4f}")
        
        # Usa media CV se disponibile, altrimenti usa score hold-out
        comparison_score = result.get('cv_mean', score)
        
        if comparison_score > best_score:
            best_score = comparison_score
            best_model_name = name
    
    # Fallback se CV fallisce (usa score hold-out)
    if best_model_name is None:
        for name, result in results.items():
            score = result['metrics']['f1_score']
            if score > best_score:
                best_score = score
                best_model_name = name
    
    # Salva il modello migliore e i suoi artefatti
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print("="*60)
    print(f"F1-Score: {best_score:.4f}")
    print(f"Precision: {results[best_model_name]['metrics']['precision']:.4f}")
    print(f"Recall: {results[best_model_name]['metrics']['recall']:.4f}")
    
    # Crea la directory models se non esiste
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Salva il modello migliore serializzato con pickle
    best_model_path = os.path.join(model_output_dir, 'best_model.pkl')
    results[best_model_name]['model'].save(best_model_path)
    print(f"\nBest model saved to: {best_model_path}")
    
    # Salva i label encoders per uso in fase di predizione
    encoders_path = os.path.join(model_output_dir, 'label_encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Label encoders saved to: {encoders_path}")
    
    # Salva l'ordine delle colonne feature per consistenza nelle predizioni
    feature_columns_path = os.path.join(model_output_dir, 'feature_columns.pkl')
    with open(feature_columns_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"Feature columns saved to: {feature_columns_path}")
    
    return best_model_name, results[best_model_name]['metrics']


if __name__ == "__main__":
    # Definisce i percorsi relativi alla directory base del progetto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'dataset.csv')
    model_output_dir = os.path.join(base_dir, 'models')
    
    # Esegue addestramento, confronto e salvataggio del modello migliore
    best_model, metrics = train_and_compare(data_path, model_output_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)