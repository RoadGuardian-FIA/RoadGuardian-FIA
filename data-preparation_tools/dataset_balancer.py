import duckdb
import time

start_time = time.time()

# 1. Configurazione
INPUT_CSV = 'US_Accidents_March23_Labeled.csv'
OUTPUT_CSV = 'US_Accidents_Balanced.csv'

# 2. Inizializziamo DuckDB
con = duckdb.connect(database=':memory:')

print(f"Caricamento dataset da '{INPUT_CSV}'...")

# 3. Carica il dataset etichettato
con.execute(f"""
    CREATE TABLE labeled AS SELECT * FROM read_csv('{INPUT_CSV}', header=True)
""")

# 4. Trova la classe con meno elementi per il bilanciamento
min_count = con.execute("""
    SELECT MIN(cnt) FROM (
        SELECT COUNT(*) as cnt FROM labeled GROUP BY Target_Label
    )
""").fetchone()[0]

print(f"Bilanciamento classi: {min_count} record per classe")

# 5. Query per bilanciare con undersampling casuale
sql_balanced = f"""
CREATE TABLE balanced AS
SELECT * FROM (
    SELECT * FROM labeled WHERE Target_Label = 'EMERGENZA_CRITICA' ORDER BY RANDOM() LIMIT {min_count}
) t1
UNION ALL
SELECT * FROM (
    SELECT * FROM labeled WHERE Target_Label = 'ARRESTO_OBBLIGATORIO' ORDER BY RANDOM() LIMIT {min_count}
) t2
UNION ALL
SELECT * FROM (
    SELECT * FROM labeled WHERE Target_Label = 'ZONA_RESIDENZIALE_CRITICA' ORDER BY RANDOM() LIMIT {min_count}
) t3
UNION ALL
SELECT * FROM (
    SELECT * FROM labeled WHERE Target_Label = 'PRUDENZA_INTERSEZIONE' ORDER BY RANDOM() LIMIT {min_count}
) t4
UNION ALL
SELECT * FROM (
    SELECT * FROM labeled WHERE Target_Label = 'GUIDA_NOTTURNA_EXTRAURBANA' ORDER BY RANDOM() LIMIT {min_count}
) t5
UNION ALL
SELECT * FROM (
    SELECT * FROM labeled WHERE Target_Label = 'GESTIONE_SINISTRO' ORDER BY RANDOM() LIMIT {min_count}
) t6
"""

# 6. Esecuzione e Export
print("Esecuzione bilanciamento...")
con.execute(sql_balanced)
con.execute(f"""
    COPY (SELECT * FROM balanced ORDER BY RANDOM()) TO '{OUTPUT_CSV}' (HEADER, DELIMITER ',');
""")

end_time = time.time()
execution_time = end_time - start_time

print(f"Completato! Il file '{OUTPUT_CSV}' è stato generato in {execution_time:.2f} secondi.")

# 7. Anteprima risultati
print("\n--- ANTEPRIMA DATASET BILANCIATO ---")
preview = con.execute(f"SELECT * FROM read_csv('{OUTPUT_CSV}', header=True) LIMIT 10").fetchall()
columns = con.execute(f"SELECT * FROM read_csv('{OUTPUT_CSV}', header=True) LIMIT 1").description
col_names = [col[0] for col in columns]
print(", ".join(col_names))
for row in preview:
    print(row)

# 8. Distribuzione etichette
print("\n--- DISTRIBUZIONE TARGET_LABEL ---")
distribution = con.execute(f"""
    SELECT Target_Label, COUNT(*) as Count
    FROM read_csv('{OUTPUT_CSV}', header=True)
    GROUP BY Target_Label
    ORDER BY Count DESC
""").fetchall()
for row in distribution:
    print(f"{row[0]}: {row[1]}")
