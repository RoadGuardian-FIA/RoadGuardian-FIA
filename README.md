# RG-Linee_Guida_Comportamentali_AI

Sistema di intelligenza artificiale per la classificazione degli incidenti stradali e la raccomandazione di protocolli comportamentali appropriati.

## Descrizione

Questo progetto implementa un sistema basato su Machine Learning (scikit-learn) con API REST (FastAPI) per classificare incidenti stradali e fornire linee guida comportamentali appropriate. Il sistema addestra e confronta modelli DecisionTree e RandomForest, selezionando automaticamente il migliore in base a precisione e recall.

## Struttura del Progetto

```
RG-Linee_Guida_Comportamentali_AI/
├── data/
│   ├── train.csv          # Dataset di addestramento
│   └── rules.json         # Knowledge base con protocolli comportamentali
├── src/
│   ├── __init__.py
│   ├── model_factory.py   # Classi per DecisionTree e RandomForest
│   ├── train_compare.py   # Script di addestramento e confronto modelli
│   └── main.py            # API FastAPI
├── models/
│   ├── best_model.pkl     # Modello migliore salvato (generato)
│   └── label_encoders.pkl # Encoder per features categoriche (generato)
├── requirements.txt       # Dipendenze Python
├── .gitignore
└── README.md
```

## Caratteristiche

- **Addestramento Automatico**: Confronto tra DecisionTree e RandomForest
- **Selezione Modello**: Selezione automatica del modello migliore basata su Precision, Recall e F1-Score
- **API REST**: Interfaccia FastAPI per predizioni in tempo reale
- **Knowledge Base**: Protocolli comportamentali strutturati in JSON
- **Metriche**: Valutazione con Precision, Recall e F1-Score

## Installazione

### Prerequisiti

- Python 3.8 o superiore
- pip

### Setup

1. Clona il repository:
```bash
git clone https://github.com/RoadGuardian-FIA/RG-Linee_Guida_Comportamentali_AI.git
cd RG-Linee_Guida_Comportamentali_AI
```

2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

## Utilizzo

### Quick Start

Per un test rapido dell'intero sistema:

```bash
# 1. Installa le dipendenze
pip install -r requirements.txt

# 2. Addestra i modelli
python src/train_compare.py

# 3. Avvia l'API (in un terminale)
python src/main.py

# 4. Testa l'API (in un altro terminale)
python test_api.py
```

### 1. Addestramento dei Modelli

Prima di utilizzare l'API, è necessario addestrare i modelli:

```bash
python src/train_compare.py
```

Questo script:
- Carica i dati da `data/train.csv`
- Addestra un DecisionTree e un RandomForest
- Confronta i modelli usando Precision, Recall e F1-Score
- Salva il modello migliore in `models/best_model.pkl`
- Salva gli encoder in `models/label_encoders.pkl`

Output esempio:
```
TRAINING AND EVALUATION

DecisionTree:
----------------------------------------
Training...
Evaluating...
  Precision: 0.8750
  Recall:    0.8750
  F1-Score:  0.8750

RandomForest:
----------------------------------------
Training...
Evaluating...
  Precision: 0.9375
  Recall:    0.9375
  F1-Score:  0.9375

MODEL COMPARISON

DecisionTree:
  Combined Score (F1): 0.8750

RandomForest:
  Combined Score (F1): 0.9375

BEST MODEL: RandomForest
F1-Score: 0.9375
Precision: 0.9375
Recall: 0.9375

Best model saved to: /path/to/models/best_model.pkl
```

### 2. Avvio dell'API

Dopo aver addestrato i modelli, avvia il server FastAPI:

```bash
python src/main.py
```

Oppure usando uvicorn direttamente:
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

L'API sarà disponibile su `http://localhost:8000`

### 3. Utilizzo dell'API

#### Documentazione Interattiva

Accedi alla documentazione Swagger UI:
```
http://localhost:8000/docs
```

#### Endpoint Disponibili

**GET /** - Informazioni API
```bash
curl http://localhost:8000/
```

**GET /health** - Stato di salute dell'API
```bash
curl http://localhost:8000/health
```

**GET /protocols** - Lista di tutti i protocolli disponibili
```bash
curl http://localhost:8000/protocols
```

**POST /predict** - Predizione del protocollo per un incidente

Esempio di richiesta:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gravita": "high",
    "data_ora": "2024-01-15 08:30:00",
    "latitudine": 45.4642,
    "longitudine": 9.1900,
    "descrizione": "Tamponamento multiplo su autostrada",
    "categoria": "tamponamento"
  }'
```

Esempio di risposta:
```json
{
  "protocol_id": 3,
  "protocol_name": "High-Risk Incident Protocol",
  "description": "Protocollo per incidenti ad alto rischio",
  "guidelines": [
    "Evacuare l'area circostante se necessario",
    "Attivare immediatamente i servizi di emergenza",
    "Istituire un perimetro di sicurezza",
    "Coordinare con squadre specializzate (pompieri, ambulanze)",
    "Documentare accuratamente con reportistica dettagliata",
    "Avviare indagine approfondita entro 48 ore"
  ]
}
```

### Test Automatico

È disponibile uno script di test automatico che verifica tutti gli endpoint dell'API:

```bash
# Assicurati che l'API sia in esecuzione, poi:
python test_api.py
```

Lo script testa:
- Root endpoint
- Health check
- Lista protocolli
- Predizioni per diversi scenari
- Gestione errori

## Dataset

### train.csv

Il dataset di addestramento contiene le seguenti colonne:

- `gravita`: Gravità dell'incidente (low, medium, high)
- `data_ora`: Data e ora dell'incidente (formato: YYYY-MM-DD HH:MM:SS)
- `latitudine`: Latitudine GPS dell'incidente
- `longitudine`: Longitudine GPS dell'incidente
- `descrizione`: Descrizione testuale dell'incidente
- `categoria`: Categoria dell'incidente
  - `tamponamento`: Collisione tra veicoli in sequenza
  - `collisione_con_ostacolo`: Veicolo contro oggetto fisso
  - `veicoli_fuori_strada`: Veicolo uscito dalla carreggiata
  - `investimento`: Pedone o ciclista investito
  - `incendio_veicolo`: Veicolo in fiamme
- `protocol_id`: ID del protocollo appropriato (target)

### rules.json

Knowledge base con i protocolli comportamentali:

- **Protocol 1**: Protocollo Tamponamento - Gestione di collisioni tra veicoli
- **Protocol 2**: Protocollo Collisione con Ostacolo - Per impatti con ostacoli fissi
- **Protocol 3**: Protocollo Veicolo Fuori Strada - Per veicoli usciti dalla carreggiata
- **Protocol 4**: Protocollo Investimento - Per investimento di pedoni o ciclisti
- **Protocol 5**: Protocollo Incendio Veicolo - Per veicoli in fiamme

Ogni protocollo include:
- `protocol_name`: Nome del protocollo
- `description`: Descrizione
- `guidelines`: Lista di linee guida comportamentali specifiche

## Architettura

### Model Factory (`model_factory.py`)

Implementa una factory pattern per la creazione e gestione dei modelli:
- `ModelBase`: Classe base con funzionalità comuni
- `DecisionTreeModel`: Implementazione DecisionTree
- `RandomForestModel`: Implementazione RandomForest
- `get_model()`: Factory function per istanziare modelli

### Train Compare (`train_compare.py`)

Gestisce l'addestramento e il confronto dei modelli:
1. Carica e preprocessa i dati
2. Codifica features categoriche
3. Split train/test (80/20)
4. Addestra entrambi i modelli
5. Valuta con Precision, Recall, F1-Score
6. Salva il modello migliore

### API (`main.py`)

API FastAPI con i seguenti endpoint:
- `/`: Informazioni API
- `/health`: Health check
- `/protocols`: Lista protocolli
- `/predict`: Predizione protocollo

Gestisce:
- Caricamento modello e encoder
- Validazione input
- Encoding features
- Predizione
- Recupero protocollo da rules.json

## Tecnologie Utilizzate

- **FastAPI**: Web framework per API REST
- **scikit-learn**: Machine Learning (DecisionTree, RandomForest)
- **pandas**: Manipolazione dati
- **numpy**: Computazione numerica
- **uvicorn**: Server ASGI
- **pydantic**: Validazione dati

## Estensioni Future

- Aggiungere più modelli (SVM, XGBoost, Neural Networks)
- Implementare cross-validation
- Aggiungere logging e monitoring
- Implementare cache per predizioni
- Aggiungere autenticazione API
- Espandere il dataset
- Implementare feature importance analysis

## Contributi

Contributi sono benvenuti! Per favore:
1. Fork del progetto
2. Crea un branch per la feature (`git checkout -b feature/AmazingFeature`)
3. Commit delle modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## Licenza

Vedi file `LICENSE` per dettagli.

## Autori

RoadGuardian-FIA Team

## Supporto

Per problemi o domande, apri un issue su GitHub.
