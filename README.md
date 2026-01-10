# RG-Linee_Guida_Comportamentali_AI

Sistema intelligente per la classificazione di scenari stradali e l'erogazione di protocolli comportamentali di sicurezza basato su Machine Learning.

## Descrizione

Questo progetto implementa un sistema "Safety-First" per guidatori. A partire da dati telemetrici e segnalazioni di incidenti, il sistema:

1. **Classifica lo scenario** utilizzando un modello ML (RandomForest/DecisionTree) addestrato su dati etichettati.
2. **Assegna un protocollo di sicurezza** specifico (es. "Emergenza Critica", "Zona Residenziale").
3. **Recupera le linee guida operative** da un database JSON separato.

Il sistema è progettato per dare priorità alla sicurezza fisica immediata (es. fermarsi per un incendio) rispetto alle normali regole del codice della strada.

## Struttura del Progetto

```text
RG-Linee_Guida_Comportamentali_AI/
├── data/
│   ├── dataset.csv             # Dataset con feature e Target_Label
│   └── protocol_db.json        # Database testuale delle Linee Guida
├── models/
│   ├── best_model.pkl          # Modello ML addestrato (RandomForest/DecisionTree)
│   ├── label_encoders.pkl      # Encoder per feature categoriche
│   └── feature_columns.pkl     # Ordine delle colonne feature
├── src/
│   ├── __init__.py             # Package init
│   ├── main.py                 # API FastAPI per l'integrazione
│   ├── model_factory.py        # Factory pattern per modelli ML
│   └── train_compare.py        # Training e confronto modelli
├── test_api.py                 # Suite di test per l'API
├── verify_dataset_duckdb.py    # Verifica struttura dataset con DuckDB
├── requirements.txt            # Dipendenze Python
├── LICENSE                     # Licenza del progetto
└── README.md
```

## Le Classi (Protocolli di Sicurezza)

Il sistema gestisce 6 protocolli distinti, ordinati per priorità di intervento. Le linee guida sono definite in `data/protocol_db.json`.

| Priorità | Classe                          | Trigger                                             | Azione                                              |
| -------- | ------------------------------- | --------------------------------------------------- | --------------------------------------------------- |
| 1        | **EMERGENZA_CRITICA**           | Incendi, Investimenti gravi, Veicoli fuori strada   | Arresto immediato, chiamata soccorsi                |
| 2        | **ARRESTO_OBBLIGATORIO**        | Passaggi a livello, Stop, Semafori                  | Preparazione all'arresto fisico del mezzo           |
| 3        | **GESTIONE_SINISTRO**           | Tamponamenti lievi, ostacoli fissi                  | Rallentamento, 4 frecce, attenzione ai detriti      |
| 4        | **ZONA_RESIDENZIALE_CRITICA**   | Pedoni (Crossing), Dossi (Bump), Zone residenziali  | Protezione utenti deboli, passo d'uomo              |
| 5        | **GUIDA_NOTTURNA_EXTRAURBANA**  | Buio (Daylight=False) su strade veloci              | Visibilità aumentata, distanza di sicurezza         |
| 6        | **PRUDENZA_INTERSEZIONE**       | Incroci, Rotonde, Precedenze (Default)              | Gestione flussi di traffico e segnalazione manovre  |

## Dataset e Feature

Il classificatore lavora sulle seguenti feature in ingresso:

### Feature Categoriche

| Feature         | Valori ammessi                                                                                              |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| `Severity`      | `low`, `medium`, `high`, `Unknown`                                                                          |
| `Incident_Type` | `Incendio Veicolo`, `Investimento`, `Veicolo Fuori Strada`, `Tamponamento`, `Collisione con ostacolo`       |
| `Road_Type`     | `motorway_trunk`, `primary_secondary`, `residential`, `service`, `tertiary`, `living_street`, `unclassified`|

### Feature Booleane (True/False)

- `Daylight` - Luce diurna
- `Bump` - Dossi
- `Crossing` - Attraversamento pedonale
- `Give_Way` - Segnale dare precedenza
- `Junction` - Incrocio
- `Railway` - Passaggio a livello
- `Roundabout` - Rotatoria
- `Stop` - Segnale di stop
- `Traffic_Signal` - Semaforo
- `Turning_Loop` - Anello di inversione

## Installazione

1. **Clona il repository:**

   ```bash
   git clone https://github.com/RoadGuardian-FIA/RG-Linee_Guida_Comportamentali_AI.git
   cd RG-Linee_Guida_Comportamentali_AI
   ```

1. **Crea un ambiente virtuale (opzionale ma consigliato):**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

1. **Installa le dipendenze:**

```bash
pip install -r requirements.txt
```

## Utilizzo

### 1. Training del Modello ML

Addestra e confronta modelli DecisionTree e RandomForest, salvando il migliore:

```bash
python src/train_compare.py
```

**Output:**

- `models/best_model.pkl` - Modello serializzato
- `models/label_encoders.pkl` - Encoder per feature categoriche
- `models/feature_columns.pkl` - Ordine delle colonne

Il training utilizza:

- **Stratified K-Fold Cross-Validation** (5 fold) per valutazione robusta
- **F1-Score weighted** come metrica principale
- **Hold-out test set** (20%) per valutazione finale

### 2. Avvio dell'API

Avvia il server FastAPI:

```bash
python src/main.py
```

L'API sarà disponibile su `http://localhost:8000`.

### 3. Test dell'API

Esegui la suite di test:

```bash
python test_api.py
```

## API Endpoints

| Metodo | Endpoint     | Descrizione                              |
| ------ | ------------ | ---------------------------------------- |
| `GET`  | `/`          | Informazioni sull'API (versione, stato)  |
| `GET`  | `/health`    | Health check (stato modello e database)  |
| `GET`  | `/protocols` | Elenco completo dei protocolli           |
| `POST` | `/predict`   | Predizione linee guida per un incidente  |

### Esempio di Richiesta POST /predict

```json
{
  "Severity": "High",
  "Incident_Type": "Incendio Veicolo",
  "Road_Type": "motorway_trunk",
  "Daylight": true,
  "Bump": false,
  "Crossing": false,
  "Give_Way": false,
  "Junction": false,
  "Railway": false,
  "Roundabout": false,
  "Stop": false,
  "Traffic_Signal": false,
  "Turning_Loop": false
}
```

### Risposta

```json
{
  "guidelines": [
    "Arrestare il veicolo immediatamente in zona sicura",
    "Attivare le quattro frecce d'emergenza",
    "Non avvicinarsi al luogo dell'incidente",
    "Chiamare i servizi di soccorso (112)",
    "Attendere istruzioni dalle autorità",
    "Lasciare libero il corridoio di emergenza per le ambulanze"
  ]
}
```

## Documentazione API Interattiva

Con l'API in esecuzione, accedi a:

- **Swagger UI:** <http://localhost:8000/docs>
- **ReDoc:** <http://localhost:8000/redoc>

## Verifica Dataset

Per verificare la struttura del dataset con DuckDB:

```bash
python verify_dataset_duckdb.py
```

Questo script controlla:

- Presenza di tutte le colonne attese
- Validità dei valori categorici
- Allineamento classi target con `protocol_db.json`
- Presenza di valori NULL

## Dipendenze

```text
fastapi==0.104.1
uvicorn[standard]==0.24.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.26.2
pydantic==2.5.0
duckdb==0.8.1
```

## Architettura del Sistema

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Client/App     │────▶│  FastAPI (main)  │────▶│  ML Model       │
│  (JSON Request) │     │                  │     │  (RandomForest) │
└─────────────────┘     └────────┬─────────┘     └────────┬────────┘
                                 │                        │
                                 ▼                        ▼
                        ┌────────────────┐       ┌────────────────┐
                        │ protocol_db.json│       │ Predicted Label│
                        │ (Guidelines)   │◀──────│ (Target_Label) │
                        └────────────────┘       └────────────────┘
```

## Performance del Modello

Risultati tipici del training (su dataset bilanciato ~80k samples):

| Modello            | CV F1-Score  | Test F1-Score |
| ------------------ | ------------ | ------------- |
| DecisionTree       | 99.95%       | 99.94%        |
| **RandomForest**   | **99.97%**   | **99.96%**    |

## Licenza

Vedi il file [LICENSE](LICENSE) per i dettagli.

## Autori

RoadGuardian-FIA Team
