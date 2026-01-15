import duckdb
import time

start_time = time.time()

# 1. Configurazione
INPUT_CSV = 'US_Accidents_March23_Cleaned.csv'
OUTPUT_CSV = 'US_Accidents_March23_Labeled.csv'

# 2. Inizializziamo DuckDB
con = duckdb.connect(database=':memory:')

print(f"Caricamento dataset da '{INPUT_CSV}'...")

# 3. QUERY DI CLASSIFICAZIONE
sql_classify = f"""
SELECT 
    *,
    CASE 
        -- PRIORITÀ 1: EMERGENZA CRITICA
        WHEN (
            LOWER(Severity) = 'high' 
            OR Incident_Type IN ('Incendio Veicolo', 'Investimento', 'Veicolo Fuori Strada')
        ) THEN 'EMERGENZA_CRITICA'

        -- PRIORITÀ 2: ARRESTO OBBLIGATORIO
        WHEN (
            Railway = 'True' 
            OR Stop = 'True' 
            OR Traffic_Signal = 'True'
        ) THEN 'ARRESTO_OBBLIGATORIO'

        -- PRIORITÀ 3: ZONA RESIDENZIALE CRITICA
        WHEN (
            Crossing = 'True' 
            OR Bump = 'True' 
            OR Road_Type IN ('living_street', 'residential', 'service')
        ) THEN 'ZONA_RESIDENZIALE_CRITICA'

        -- PRIORITÀ 4: PRUDENZA INTERSEZIONE
        WHEN (
            Give_Way = 'True' 
            OR Roundabout = 'True' 
            OR Junction = 'True' 
            OR Turning_Loop = 'True'
        ) THEN 'PRUDENZA_INTERSEZIONE'

        -- PRIORITÀ 5: GUIDA NOTTURNA
        WHEN (
            Daylight = 'False'
        ) THEN 'GUIDA_NOTTURNA_EXTRAURBANA'

        -- PRIORITÀ 6: GESTIONE SINISTRO (incidenti rimanenti con tipo definito)
        WHEN (
            Incident_Type IS NOT NULL 
            AND Incident_Type != ''
        ) THEN 'GESTIONE_SINISTRO'

        -- DEFAULT: casi residui senza tipo incidente
        ELSE 'GUIDA_NOTTURNA_EXTRAURBANA'
    END AS Target_Label

FROM read_csv('{INPUT_CSV}', header=True)
"""

# 4. Esecuzione e Export
print("Esecuzione classificazione...")
con.execute(f"""
    COPY ({sql_classify}) TO '{OUTPUT_CSV}' (HEADER, DELIMITER ',');
""")

end_time = time.time()
execution_time = end_time - start_time

print(f"Completato! Il file '{OUTPUT_CSV}' è stato generato in {execution_time:.2f} secondi.")

# 5. Anteprima risultati
print("\n--- ANTEPRIMA DATASET ETICHETTATO ---")
preview = con.execute(f"SELECT * FROM read_csv('{OUTPUT_CSV}', header=True) LIMIT 10").fetchall()
columns = con.execute(f"SELECT * FROM read_csv('{OUTPUT_CSV}', header=True) LIMIT 1").description
col_names = [col[0] for col in columns]
print(", ".join(col_names))
for row in preview:
    print(row)

# 6. Distribuzione etichette
print("\n--- DISTRIBUZIONE TARGET_LABEL ---")
distribution = con.execute(f"""
    SELECT Target_Label, COUNT(*) as Count
    FROM read_csv('{OUTPUT_CSV}', header=True)
    GROUP BY Target_Label
    ORDER BY Count DESC
""").fetchall()
for row in distribution:
    print(f"{row[0]}: {row[1]}")
