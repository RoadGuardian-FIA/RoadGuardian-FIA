#!/usr/bin/env python3
"""
Script di verifica struttura dataset con DuckDB.

Ottimizzato per file CSV di grandi dimensioni usando query SQL lazy.
Verifica allineamento tra dataset, modello e protocol_db.json.
"""

import duckdb
import json
import os

# Percorsi dei file di riferimento
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')
PROTOCOL_DB_PATH = os.path.join(BASE_DIR, 'data', 'protocol_db.json')

# Colonne categoriche attese nel dataset
EXPECTED_CATEGORICAL = ['Severity', 'Incident_Type', 'Road_Type']

# Colonne booleane attese (caratteristiche infrastrutturali)
EXPECTED_BOOLEAN = [
    'Daylight', 'Bump', 'Crossing', 'Give_Way', 'Junction',
    'Railway', 'Roundabout', 'Stop', 'Traffic_Signal', 'Turning_Loop'
]

# Nome della colonna target
TARGET_COLUMN = 'Target_Label'

# Valori ammessi per le colonne categoriche
EXPECTED_VALUES = {
    'Severity': ['low', 'medium', 'high', 'Unknown'],
    'Incident_Type': ['Incendio Veicolo', 'Investimento', 'Veicolo Fuori Strada', 'Tamponamento', 'Collisione con ostacolo'],
    'Road_Type': ['motorway_trunk', 'primary_secondary', 'service', 'tertiary', 'residential', 'living_street', 'unclassified']
}


def load_protocol_classes():
    """
    Scopo: Carica le classi di protocollo dal file protocol_db.json.

    Parametri:
    - Nessuno

    Valore di ritorno:
    - list: Lista delle chiavi (classi) presenti nel database protocolli.

    Eccezioni:
    - Nessuna eccezione prevista (file mancante gestito con warning).
    """
    if not os.path.exists(PROTOCOL_DB_PATH):
        print(f"⚠ File protocol_db.json non trovato: {PROTOCOL_DB_PATH}")
        return []
    
    with open(PROTOCOL_DB_PATH, 'r', encoding='utf-8') as f:
        protocol_db = json.load(f)
    return list(protocol_db.keys())


def verify_dataset():
    """
    Scopo: Verifica la struttura del dataset usando DuckDB per performance ottimali.

    Parametri:
    - Nessuno

    Valore di ritorno:
    - bool: True se il dataset è valido e allineato con protocol_db, False altrimenti.

    Eccezioni:
    - Nessuna eccezione prevista (errori gestiti internamente).

    Verifiche eseguite:
    - Esistenza e leggibilità file CSV
    - Presenza colonne attese
    - Distribuzione valori categorici
    - Formato colonne booleane
    - Allineamento classi target con protocol_db.json
    - Presenza valori NULL
    """
    
    print("=" * 70)
    print("VERIFICA STRUTTURA DATASET CON DUCKDB")
    print("=" * 70)
    
    # Verifica esistenza file
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ ERRORE: File non trovato: {DATA_PATH}")
        return False
    
    print(f"\n✓ File trovato: {DATA_PATH}")
    
    # Connessione DuckDB
    con = duckdb.connect(':memory:')
    
    # Crea una view sul CSV (lazy loading, non carica tutto in memoria)
    con.execute(f"CREATE VIEW dataset AS SELECT * FROM read_csv_auto('{DATA_PATH.replace(os.sep, '/')}')")
    
    # =========================================================================
    # VERIFICA COLONNE
    # =========================================================================
    print("\n" + "-" * 70)
    print("VERIFICA COLONNE")
    print("-" * 70)
    
    columns = con.execute("DESCRIBE dataset").fetchall()
    column_names = [col[0] for col in columns]
    
    print(f"\nColonne trovate ({len(column_names)}):")
    for col in columns:
        print(f"  - {col[0]} ({col[1]})")
    
    all_expected = EXPECTED_CATEGORICAL + EXPECTED_BOOLEAN + [TARGET_COLUMN]
    missing = [col for col in all_expected if col not in column_names]
    
    if missing:
        print(f"\n❌ COLONNE MANCANTI: {missing}")
    else:
        print(f"\n✓ Tutte le colonne attese sono presenti")
    
    # =========================================================================
    # STATISTICHE GENERALI
    # =========================================================================
    print("\n" + "-" * 70)
    print("STATISTICHE GENERALI")
    print("-" * 70)
    
    row_count = con.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
    print(f"Righe totali: {row_count:,}")
    print(f"Colonne totali: {len(column_names)}")
    
    # =========================================================================
    # VERIFICA VALORI CATEGORICI
    # =========================================================================
    print("\n" + "-" * 70)
    print("VERIFICA VALORI CATEGORICI")
    print("-" * 70)
    
    for col, expected_vals in EXPECTED_VALUES.items():
        if col not in column_names:
            print(f"\n⚠ {col}: colonna non presente")
            continue
        
        result = con.execute(f"""
            SELECT {col}, COUNT(*) as cnt 
            FROM dataset 
            GROUP BY {col} 
            ORDER BY cnt DESC
        """).fetchall()
        
        actual_vals = [r[0] for r in result]
        unexpected = [v for v in actual_vals if v not in expected_vals]
        
        print(f"\n{col}:")
        print(f"  Valori attesi: {expected_vals}")
        print(f"  Distribuzione nel dataset:")
        for val, cnt in result:
            pct = (cnt / row_count) * 100
            status = "✓" if val in expected_vals else "❌"
            print(f"    {status} {val}: {cnt:,} ({pct:.2f}%)")
        
        if unexpected:
            print(f"  ❌ Valori non attesi: {unexpected}")
    
    # =========================================================================
    # VERIFICA COLONNE BOOLEANE
    # =========================================================================
    print("\n" + "-" * 70)
    print("VERIFICA COLONNE BOOLEANE")
    print("-" * 70)
    
    for col in EXPECTED_BOOLEAN:
        if col not in column_names:
            print(f"⚠ {col}: colonna non presente")
            continue
        
        result = con.execute(f"""
            SELECT {col}, COUNT(*) as cnt 
            FROM dataset 
            GROUP BY {col}
            ORDER BY {col}
        """).fetchall()
        
        print(f"\n{col}:")
        for val, cnt in result:
            pct = (cnt / row_count) * 100
            print(f"    {val}: {cnt:,} ({pct:.2f}%)")
    
    # =========================================================================
    # VERIFICA TARGET VS PROTOCOL_DB
    # =========================================================================
    print("\n" + "-" * 70)
    print("VERIFICA TARGET vs PROTOCOL_DB")
    print("-" * 70)
    
    # Classi nel protocol_db.json
    protocol_classes = load_protocol_classes()
    print(f"\nClassi in protocol_db.json ({len(protocol_classes)}):")
    for cls in protocol_classes:
        print(f"  - {cls}")
    
    # Classi nel dataset
    if TARGET_COLUMN in column_names:
        result = con.execute(f"""
            SELECT {TARGET_COLUMN}, COUNT(*) as cnt 
            FROM dataset 
            GROUP BY {TARGET_COLUMN}
            ORDER BY cnt DESC
        """).fetchall()
        
        dataset_classes = [r[0] for r in result]
        
        print(f"\nClassi nel dataset ({len(dataset_classes)}):")
        for cls, cnt in result:
            pct = (cnt / row_count) * 100
            in_db = "✓" if cls in protocol_classes else "❌"
            print(f"  {in_db} {cls}: {cnt:,} ({pct:.2f}%)")
        
        # Classi mancanti nel dataset
        missing_in_dataset = [cls for cls in protocol_classes if cls not in dataset_classes]
        if missing_in_dataset:
            print(f"\n⚠ CLASSI IN PROTOCOL_DB MA NON NEL DATASET ({len(missing_in_dataset)}):")
            for cls in missing_in_dataset:
                print(f"  - {cls}")
        
        # Classi extra nel dataset
        extra_in_dataset = [cls for cls in dataset_classes if cls not in protocol_classes]
        if extra_in_dataset:
            print(f"\n❌ CLASSI NEL DATASET MA NON IN PROTOCOL_DB ({len(extra_in_dataset)}):")
            for cls in extra_in_dataset:
                print(f"  - {cls}")
        
        if not missing_in_dataset and not extra_in_dataset:
            print("\n✓ Le classi nel dataset corrispondono esattamente a protocol_db.json")
    else:
        print(f"\n❌ Colonna target '{TARGET_COLUMN}' non trovata!")
    
    # =========================================================================
    # VERIFICA NULL VALUES
    # =========================================================================
    print("\n" + "-" * 70)
    print("VERIFICA VALORI NULL")
    print("-" * 70)
    
    null_counts = []
    for col in column_names:
        null_count = con.execute(f"SELECT COUNT(*) FROM dataset WHERE {col} IS NULL").fetchone()[0]
        if null_count > 0:
            null_counts.append((col, null_count))
    
    if null_counts:
        print("\nColonne con valori NULL:")
        for col, cnt in null_counts:
            pct = (cnt / row_count) * 100
            print(f"  ❌ {col}: {cnt:,} ({pct:.2f}%)")
    else:
        print("\n✓ Nessun valore NULL trovato")
    
    # =========================================================================
    # RIEPILOGO
    # =========================================================================
    print("\n" + "=" * 70)
    print("RIEPILOGO VERIFICA")
    print("=" * 70)
    
    issues = []
    if missing:
        issues.append(f"Colonne mancanti: {missing}")
    if TARGET_COLUMN in column_names:
        if missing_in_dataset:
            issues.append(f"Classi mancanti nel dataset: {missing_in_dataset}")
        if extra_in_dataset:
            issues.append(f"Classi extra nel dataset: {extra_in_dataset}")
    if null_counts:
        issues.append(f"Valori NULL in: {[c[0] for c in null_counts]}")
    
    if issues:
        print("\n❌ PROBLEMI RILEVATI:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ Il dataset è compatibile con il modello e protocol_db.json!")
        return True


if __name__ == "__main__":
    verify_dataset()
