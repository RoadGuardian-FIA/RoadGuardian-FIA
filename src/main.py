"""
Applicazione principale FastAPI per RoadGuardian.

Fornisce endpoint per la classificazione degli incidenti e il recupero 
dei protocolli di sicurezza utilizzando un modello ML addestrato.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import json
import os
import pickle
import numpy as np


class UnicodeJSONResponse(JSONResponse):
    """Risposta JSON che preserva i caratteri Unicode (es. accenti)."""
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


# Inizializzazione dell'applicazione FastAPI con risposta UTF-8
app = FastAPI(
    title="RG Behavioral Guidelines AI",
    description="API per la classificazione degli incidenti e il recupero dei protocolli di sicurezza (ML-Based)",
    version="2.1.0",
    default_response_class=UnicodeJSONResponse
)

# Variabile globale per il database dei protocolli (caricato all'avvio)
PROTOCOL_DB = {}

# Variabili globali per il modello ML
ML_MODEL = None
LABEL_ENCODERS = {}
FEATURE_COLUMNS = []

# Definizione colonne per encoding (deve corrispondere a train_compare.py)
CATEGORICAL_COLUMNS = ['Severity', 'Incident_Type', 'Road_Type']
BOOLEAN_COLUMNS = [
    'Daylight', 'Bump', 'Crossing', 'Give_Way', 'Junction',
    'Railway', 'Roundabout', 'Stop', 'Traffic_Signal', 'Turning_Loop'
]


# --- Modelli Pydantic ---

class IncidentInput(BaseModel):
    """Input per classificazione incidente: dati telemetrici e infrastrutturali validati."""
    
    Severity: str = Field(..., description="Livello di gravità dell'incidente. Valori ammessi: 'Low', 'Medium', 'High', 'Unknown'.")
    Incident_Type: Optional[str] = Field(None, description="Tipo di incidente (es. 'Incendio Veicolo', 'Tamponamento', 'Investimento').")
    Road_Type: str = Field(..., description="Tipologia di strada (es. 'motorway_trunk', 'residential', 'service').")
    Daylight: bool = Field(..., description="Condizioni di luce: True=Giorno, False=Notte.")
    
    # Feature Booleane Infrastrutturali (elementi presenti sulla strada)
    Bump: bool = Field(False, description="Presenza di dossi artificiali.")
    Crossing: bool = Field(False, description="Presenza di attraversamento pedonale.")
    Give_Way: bool = Field(False, description="Presenza di segnale dare precedenza.")
    Junction: bool = Field(False, description="Presenza di incrocio/intersezione.")
    Railway: bool = Field(False, description="Presenza di passaggio a livello.")
    Roundabout: bool = Field(False, description="Presenza di rotatoria.")
    Stop: bool = Field(False, description="Presenza di segnale di stop.")
    Traffic_Signal: bool = Field(False, description="Presenza di semaforo.")
    Turning_Loop: bool = Field(False, description="Presenza di anello di inversione.")

    class Config:
        json_schema_extra = {
            "example": {
                "Severity": "High",
                "Incident_Type": "Incendio Veicolo",
                "Road_Type": "motorway_trunk",
                "Daylight": True,
                "Bump": False,
                "Crossing": False,
                "Give_Way": False,
                "Junction": False,
                "Railway": False,
                "Roundabout": False,
                "Stop": False,
                "Traffic_Signal": False,
                "Turning_Loop": False
            }
        }


class ProtocolResponse(BaseModel):
    """Output della predizione: contiene le linee guida comportamentali da seguire."""
    
    guidelines: List[str] = Field(..., description="Lista ordinata di azioni da intraprendere in ordine di priorità.")


# --- Funzioni Core Logic ---

def load_model():
    """
    Scopo: Carica il modello ML addestrato e i suoi artefatti (encoders, feature columns).

    Parametri:
    - Nessuno

    Valore di ritorno:
    - None: Popola le variabili globali ML_MODEL, LABEL_ENCODERS, FEATURE_COLUMNS.

    Eccezioni:
    - FileNotFoundError: Se i file del modello non sono presenti.
    - RuntimeError: Se il caricamento del modello fallisce.
    """
    global ML_MODEL, LABEL_ENCODERS, FEATURE_COLUMNS
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'models')
    
    model_path = os.path.join(model_dir, 'best_model.pkl')
    encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
    features_path = os.path.join(model_dir, 'feature_columns.pkl')
    
    # Verifica esistenza file - ora obbligatorio
    missing_files = [p for p in [model_path, encoders_path, features_path] if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(
            f"File modello mancanti: {missing_files}. "
            f"Eseguire 'python src/train_compare.py' per addestrare il modello."
        )
    
    try:
        with open(model_path, 'rb') as f:
            ML_MODEL = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            LABEL_ENCODERS = pickle.load(f)
        with open(features_path, 'rb') as f:
            FEATURE_COLUMNS = pickle.load(f)
        
        print(f"Modello caricato con successo. Features: {FEATURE_COLUMNS}")
    except Exception as e:
        raise RuntimeError(f"Errore nel caricamento del modello: {e}")


def load_rules():
    """
    Scopo: Carica il database dei protocolli di sicurezza dal file JSON all'avvio.

    Parametri:
    - Nessuno

    Valore di ritorno:
    - None: Popola la variabile globale PROTOCOL_DB.

    Eccezioni:
    - Nessuna eccezione prevista (errori gestiti internamente con log).
    """
    global PROTOCOL_DB
    
    # Percorso relativo alla struttura del progetto: ../data/protocol_db.json
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rules_path = os.path.join(base_dir, 'data', 'protocol_db.json')
    
    if not os.path.exists(rules_path):
        print(f"WARNING: Database file not found at {rules_path}")
        return
    
    try:
        with open(rules_path, 'r', encoding='utf-8') as f:
            PROTOCOL_DB = json.load(f)
        print(f"Database protocolli caricato con successo: {list(PROTOCOL_DB.keys())}")
    except Exception as e:
        print(f"ERROR loading rules: {e}")


def prepare_features(data: IncidentInput) -> np.ndarray:
    """
    Scopo: Prepara le feature per la predizione ML, applicando encoding e conversioni.

    Parametri:
    - data (IncidentInput): Dati dell'incidente validati da Pydantic.

    Valore di ritorno:
    - np.ndarray: Vettore delle feature pronto per la predizione (1, n_features).

    Eccezioni:
    - KeyError: Se una colonna categorica ha un valore non visto durante il training.
    """
    features = []
    
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            # Ottieni il valore dal modello Pydantic
            value = getattr(data, col)
            if value is None:
                value = 'Unknown'
            
            # Normalizzazione case-sensitive per matching con il training set
            # Severity e Road_Type sono lowercase nel training
            # Incident_Type ha maiuscola iniziale (es. "Tamponamento")
            if col in ['Severity', 'Road_Type']:
                value_normalized = str(value).lower()
            else:
                # Per Incident_Type, capitalizza la prima lettera di ogni parola
                value_normalized = str(value).title()
            
            # Applica LabelEncoder
            if col in LABEL_ENCODERS:
                try:
                    encoded = LABEL_ENCODERS[col].transform([value_normalized])[0]
                except ValueError:
                    # Valore non visto durante training, usa valore più comune (0)
                    print(f"[WARNING] Valore '{value_normalized}' non riconosciuto per colonna '{col}', uso fallback")
                    encoded = 0
                features.append(encoded)
            else:
                features.append(0)
        
        elif col in BOOLEAN_COLUMNS:
            # Converti boolean in 0/1
            value = getattr(data, col)
            features.append(1 if value else 0)
    
    return np.array(features).reshape(1, -1)


def classify_incident_ml(data: IncidentInput) -> str:
    """
    Scopo: Classifica l'incidente usando il modello ML addestrato.

    Parametri:
    - data (IncidentInput): Dati dell'incidente validati da Pydantic.

    Valore di ritorno:
    - str: Etichetta della classe predetta dal modello.

    Eccezioni:
    - Exception: Eventuali errori propagati dalla predizione.
    """
    X = prepare_features(data)
    prediction = ML_MODEL.predict(X)
    return prediction[0]


def classify_incident(data: IncidentInput) -> str:
    """
    Scopo: Classifica l'incidente usando il modello ML addestrato.

    Parametri:
    - data (IncidentInput): Dati dell'incidente validati da Pydantic.

    Valore di ritorno:
    - str: Etichetta della classe di rischio predetta.

    Eccezioni:
    - RuntimeError: Se il modello non è stato caricato.
    - Exception: Eventuali errori propagati dalla predizione ML.
    """
    if ML_MODEL is None:
        raise RuntimeError("Modello ML non caricato. Eseguire load_model() all'avvio.")
    
    return classify_incident_ml(data)


# --- Endpoint API ---

@app.on_event("startup")
async def startup_event():
    """
    Scopo: Evento di inizializzazione eseguito all'avvio dell'applicazione.
           Carica il database dei protocolli in memoria.

    Parametri:
    - Nessuno

    Valore di ritorno:
    - None

    Eccezioni:
    - Nessuna eccezione prevista.
    """
    load_model()  # Carica il modello ML (se disponibile)
    load_rules()  # Carica il database dei protocolli


@app.get("/")
async def root():
    """
    Scopo: Endpoint di benvenuto con informazioni base sull'API.

    Parametri:
    - Nessuno

    Valore di ritorno:
    - dict: Messaggio di benvenuto, versione e stato.

    Eccezioni:
    - Nessuna eccezione prevista.
    """
    return {
        "message": "RG Behavioral Guidelines AI (ML-Based)",
        "version": "2.1.0",
        "status": "Active"
    }


@app.get("/health")
async def health_check():
    """
    Scopo: Endpoint di health check per monitoraggio e load balancer.

    Parametri:
    - Nessuno

    Valore di ritorno:
    - dict: Stato del servizio, flag caricamento modello e database.

    Eccezioni:
    - Nessuna eccezione prevista.
    """
    return {
        "status": "healthy",
        "model_loaded": ML_MODEL is not None,
        "db_loaded": len(PROTOCOL_DB) > 0
    }


@app.get("/protocols")
async def get_all_protocols():
    """
    Scopo: Restituisce l'intero database dei protocolli di sicurezza.

    Parametri:
    - Nessuno

    Valore di ritorno:
    - dict: Dizionario completo {classe: {guidelines: [...]}}.

    Eccezioni:
    - Nessuna eccezione prevista.
    """
    return PROTOCOL_DB


@app.post("/predict", response_model=ProtocolResponse)
async def predict_protocol(incident: IncidentInput):
    """
    Scopo: Endpoint principale per la predizione delle linee guida comportamentali.
           Riceve dati incidente, classifica il rischio e restituisce il protocollo.

    Parametri:
    - incident (IncidentInput): Dati dell'incidente validati da Pydantic.

    Valore di ritorno:
    - ProtocolResponse: Oggetto contenente la lista delle linee guida.

    Eccezioni:
    - HTTPException 503: Se il database protocolli non è caricato.
    - HTTPException 500: Se il protocollo di default non viene trovato.

    Flusso:
    1. Classificazione: Determina la classe di rischio (Logic Layer)
    2. Recupero: Estrae le linee guida dal database (Retrieval Layer)
    3. Fallback: Usa PRUDENZA_INTERSEZIONE se classe non trovata
    """
    if not PROTOCOL_DB:
        raise HTTPException(status_code=503, detail="Protocol Database not loaded")

    # Step 1: Classificazione - Determina la classe di rischio
    predicted_label = classify_incident(incident)
    
    # Step 2: Recupero - Estrae il protocollo dal database
    protocol_data = PROTOCOL_DB.get(predicted_label)

    # Step 3: Fallback - Usa protocollo di default se classe non trovata
    if not protocol_data:
        protocol_data = PROTOCOL_DB.get("PRUDENZA_INTERSEZIONE")
        if not protocol_data:
             raise HTTPException(status_code=500, detail="Default protocol not found in DB")

    return ProtocolResponse(guidelines=protocol_data['guidelines'])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)