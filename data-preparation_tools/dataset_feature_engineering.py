import duckdb
import time

start_time = time.time()

con = duckdb.connect(database='accidents.db')

csv_path = 'US_Accidents_March23.csv'

print("Inizio processamento dataset filtrato...")
print("NOTA: Calcolo 'Luce' basato sulla colonna 'Sunrise_Sunset' esistente (ottimizzato).")

# Utilizziamo read_csv con parametri espliciti per massima stabilità
con.execute(f"""
    COPY (
        WITH RawData AS (
            SELECT * FROM read_csv('{csv_path}', all_varchar=True)
            WHERE Start_Time IS NOT NULL
        ),
        -- Primo passaggio: Calcoliamo la Tipologia Incidente per poter FILTRARE SUBITO
        Intermediate AS (
            SELECT 
                *,
                CASE 
                    -----------------------------------------------------------
                    -- LIVELLO 1: CERTEZZA (Basato sul testo esplicito INGLESE)
                    -----------------------------------------------------------
                    WHEN Description ILIKE '%FIRE%' OR Description ILIKE '%ABLAZE%' OR Description ILIKE '%SMOKE%' 
                        THEN 'Incendio Veicolo'
                    
                    WHEN Description ILIKE '%PEDESTRIAN%' OR Description ILIKE '%CYCLIST%' OR Description ILIKE '%STRUCK BY PERSON%' 
                        THEN 'Investimento'

                    WHEN Description ILIKE '%OVERTURN%' OR Description ILIKE '%ROLLOVER%' OR Description ILIKE '%DITCH%' OR Description ILIKE '%OFF ROAD%' OR Description ILIKE '%JACKKNIFE%'
                        THEN 'Veicolo Fuori Strada'
                    
                    WHEN Description ILIKE '%REAR END%'
                        THEN 'Tamponamento'

                    WHEN Description ILIKE '%DEBRIS%' OR Description ILIKE '%TREE%' OR Description ILIKE '%POLE%' OR Description ILIKE '%GUARDRAIL%' OR Description ILIKE '%BARRIER%' OR Description ILIKE '%DEER%' OR Description ILIKE '%ANIMAL%'
                        THEN 'Collisione con ostacolo'

                    -----------------------------------------------------------
                    -- LIVELLO 2: DEDUZIONE CONTESTUALE
                    -----------------------------------------------------------

                    -- A. INVESTIMENTO
                    WHEN Crossing = 'True' OR Station = 'True' 
                        THEN 'Investimento'

                    -- B. VEICOLO FUORI STRADA (Inferito da Meteo + Geometria)
                    WHEN (Weather_Condition ILIKE '%Snow%' OR Weather_Condition ILIKE '%Ice%' OR Weather_Condition ILIKE '%Rain%' OR Weather_Condition ILIKE '%Wind%' OR Weather_Condition ILIKE '%Thunder%')
                        AND (Traffic_Signal = 'False' AND Stop = 'False')
                        THEN 'Veicolo Fuori Strada'

                    -- C. COLLISIONE CON OSTACOLO
                    WHEN Railway = 'True' OR No_Exit = 'True'
                        THEN 'Collisione con ostacolo'

                    -- D. TAMPONAMENTO
                    WHEN Traffic_Signal = 'True' OR Stop = 'True' OR Junction = 'True' OR Bump = 'True' OR Roundabout = 'True'
                        THEN 'Tamponamento'

                    -----------------------------------------------------------
                    -- LIVELLO 3: RESIDUALE
                    -----------------------------------------------------------
                    ELSE 'Incidente Generico'
                END AS Incident_Type_Calc
            FROM RawData
        )
        -- Query Finale: Applica filtri e colonne calcolate SOLO sulle righe superstiti
        SELECT 
            -- Mappatura Gravità (1-4) a Etichette
            CASE 
                WHEN Severity = 1 OR Severity = 2 THEN 'low'
                WHEN Severity = 3 THEN 'medium'
                WHEN Severity = 4 THEN 'high'
                ELSE 'Unknown'
            END AS Severity,
            
            -- Usiamo la colonna calcolata nella CTE
            Incident_Type_Calc AS Incident_Type,

            -- Tipologia Strada (Mappatura approssimativa a tag OSM)
            CASE 
                -- motorway / trunk
                WHEN Street ILIKE 'I-%' OR Street ILIKE 'Interstate %' 
                     OR Street ILIKE '% Fwy' OR Street ILIKE '% Freeway' 
                     OR Street ILIKE '% Expy' OR Street ILIKE '% Expressway'
                     OR Street ILIKE '% Tpke' OR Street ILIKE '% Turnpike' OR Street ILIKE '% Tollway'
                     OR Street ILIKE '% Pkwy' OR Street ILIKE '% Parkway'
                     OR Street ILIKE '% Beltway%' OR Street ILIKE '% Bypass%'
                     OR Description ILIKE '%Interstate%'
                    THEN 'motorway_trunk'

                -- primary / secondary
                WHEN Street ILIKE 'US-%' OR Street ILIKE 'U.S. %' 
                     OR Description ILIKE '%U.S. Hwy%' OR Description ILIKE '%US-%'
                     OR Street ILIKE 'SR-%' OR Street ILIKE 'State Route %' 
                     OR Street ILIKE 'CR-%' OR Street ILIKE 'County Road %'
                     OR Street ILIKE 'Rte %' OR Street ILIKE 'Route %' 
                     OR Street ILIKE 'Hwy %' OR Street ILIKE '% Highway%'
                     OR Street ILIKE 'FM-%' OR Street ILIKE 'Farm to Market %'
                    THEN 'primary_secondary'
                
                -- service
                WHEN Street ILIKE '% Service Rd' OR Street ILIKE '% Frontage Rd'
                     OR Street ILIKE '% Access Rd' OR Street ILIKE '% Parking%'
                    THEN 'service'

                -- tertiary
                WHEN Street ILIKE '% Rd' OR Street ILIKE '% Road'
                     OR Street ILIKE '% Blvd' OR Street ILIKE '% Boulevard'
                    THEN 'tertiary'

                -- residential
                WHEN Street ILIKE '% Ave' OR Street ILIKE '% Avenue'
                     OR Street ILIKE '% St' OR Street ILIKE '% Street'
                     OR Street ILIKE '% Dr' OR Street ILIKE '% Drive'
                     OR Street ILIKE '% Ln' OR Street ILIKE '% Lane'
                     OR Street ILIKE '% Way' OR Street ILIKE '% Pl' OR Street ILIKE '% Place'
                     OR Street ILIKE '% Ct' OR Street ILIKE '% Court'
                     OR Street ILIKE '% Cir' OR Street ILIKE '% Circle'
                     OR Street ILIKE '% Ter' OR Street ILIKE '% Terrace'
                    THEN 'residential'
                
                -- living_street
                WHEN Street ILIKE '% Plaza' OR Street ILIKE '% Walk%' OR Street ILIKE '% Mall%'
                    THEN 'living_street'

                ELSE 'unclassified'
            END AS Road_Type,

            -- Nuova colonna Momento Giornata basata su Sunrise_Sunset esistente (Day=Luce, Night=Buio)
            CASE 
                WHEN Sunrise_Sunset = 'Day' THEN 'True'
                ELSE 'False'
            END AS Daylight,
            
            -- Colonne Infrastruttura e Tempo mantenute
            Bump,
            Crossing,
            Give_Way,
            Junction,
            Railway,
            Roundabout,
            Stop,
            Traffic_Signal,
            Turning_Loop
        FROM Intermediate
        WHERE Incident_Type_Calc != 'Incidente Generico'
    ) TO 'US_Accidents_March23_Cleaned.csv' (HEADER, DELIMITER ',');
""")

end_time = time.time()
execution_time = end_time - start_time

print(f"Completato! Il file 'US_Accidents_March23_Cleaned.csv' è stato generato in {execution_time:.2f} secondi.")