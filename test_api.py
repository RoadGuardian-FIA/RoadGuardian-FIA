#!/usr/bin/env python3
"""
Test suite per l'API RG Linee guida comportamentali AI.

Esegue una suite di test pytest per verificare il corretto funzionamento
dell'API FastAPI, inclusi endpoint di health check e predizione.

Utilizzo:
    pytest test_api.py -v
    pytest test_api.py -v --tb=short
"""

import pytest
import requests
import time

# URL base dell'API da testare
API_URL = "http://localhost:8001"


@pytest.fixture(scope="module")
def api_available():
    """
    Scopo: Fixture pytest che verifica la disponibilità dell'API.

    Parametri:
    - Nessuno

    Valore di ritorno:
    - bool: True se l'API è disponibile.

    Eccezioni:
    - pytest.skip: Se l'API non risponde entro il timeout.
    """
    timeout = 5
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    pytest.skip(
        "API non disponibile. Avviala con: python src/main.py"
    )


class TestRootEndpoint:
    """Test per l'endpoint root (/)."""

    def test_root_returns_200(self, api_available):
        """
        Scopo: Verifica che l'endpoint root restituisca status 200.

        Valore atteso:
        - Status code 200.
        """
        response = requests.get(f"{API_URL}/")
        assert response.status_code == 200

    def test_root_contains_required_fields(self, api_available):
        """
        Scopo: Verifica che la risposta contenga i campi obbligatori.

        Valore atteso:
        - Campi 'message', 'version', 'status' presenti.
        """
        response = requests.get(f"{API_URL}/")
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data

    def test_root_status_is_active(self, api_available):
        """
        Scopo: Verifica che lo status dell'API sia 'Active'.

        Valore atteso:
        - Campo 'status' uguale a 'Active'.
        """
        response = requests.get(f"{API_URL}/")
        data = response.json()
        assert data["status"] == "Active"


class TestHealthEndpoint:
    """Test per l'endpoint health (/health)."""

    def test_health_returns_200(self, api_available):
        """
        Scopo: Verifica che l'endpoint health restituisca status 200.

        Valore atteso:
        - Status code 200.
        """
        response = requests.get(f"{API_URL}/health")
        assert response.status_code == 200

    def test_health_contains_required_fields(self, api_available):
        """
        Scopo: Verifica che la risposta contenga i campi obbligatori.

        Valore atteso:
        - Campi 'status', 'model_loaded', 'db_loaded' presenti.
        """
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "db_loaded" in data

    def test_health_model_is_loaded(self, api_available):
        """
        Scopo: Verifica che il modello ML sia caricato.

        Valore atteso:
        - Campo 'model_loaded' uguale a True.
        """
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        assert data["model_loaded"] is True

    def test_health_db_is_loaded(self, api_available):
        """
        Scopo: Verifica che il database protocolli sia caricato.

        Valore atteso:
        - Campo 'db_loaded' uguale a True.
        """
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        assert data["db_loaded"] is True


class TestProtocolsEndpoint:
    """Test per l'endpoint protocols (/protocols)."""

    def test_protocols_returns_200(self, api_available):
        """
        Scopo: Verifica che l'endpoint protocols restituisca status 200.

        Valore atteso:
        - Status code 200.
        """
        response = requests.get(f"{API_URL}/protocols")
        assert response.status_code == 200

    def test_protocols_returns_dict(self, api_available):
        """
        Scopo: Verifica che la risposta sia un dizionario.

        Valore atteso:
        - Risposta di tipo dict.
        """
        response = requests.get(f"{API_URL}/protocols")
        data = response.json()
        assert isinstance(data, dict)

    def test_protocols_contains_expected_classes(self, api_available):
        """
        Scopo: Verifica che siano presenti le classi di protocollo attese.

        Valore atteso:
        - Presenza delle 6 classi di protocollo.
        """
        expected_classes = [
            "EMERGENZA_CRITICA",
            "ARRESTO_OBBLIGATORIO",
            "GESTIONE_SINISTRO",
            "ZONA_RESIDENZIALE_CRITICA",
            "GUIDA_NOTTURNA_EXTRAURBANA",
            "PRUDENZA_INTERSEZIONE",
        ]
        response = requests.get(f"{API_URL}/protocols")
        data = response.json()
        for cls in expected_classes:
            assert cls in data, f"Classe mancante: {cls}"


class TestPredictEndpoint:
    """Test per l'endpoint predict (/predict)."""

    @pytest.fixture
    def base_incident_data(self):
        """
        Scopo: Fornisce dati base per un incidente.

        Valore di ritorno:
        - dict: Dati di un incidente con valori di default.
        """
        return {
            "Severity": "Medium",
            "Incident_Type": "Tamponamento",
            "Road_Type": "primary_secondary",
            "Daylight": True,
            "Bump": False,
            "Crossing": False,
            "Give_Way": False,
            "Junction": False,
            "Railway": False,
            "Roundabout": False,
            "Stop": False,
            "Traffic_Signal": False,
            "Turning_Loop": False,
        }

    def test_predict_returns_200(self, api_available, base_incident_data):
        """
        Scopo: Verifica che l'endpoint predict restituisca status 200.

        Valore atteso:
        - Status code 200.
        """
        response = requests.post(f"{API_URL}/predict", json=base_incident_data)
        assert response.status_code == 200

    def test_predict_returns_guidelines(self, api_available, base_incident_data):
        """
        Scopo: Verifica che la risposta contenga le linee guida.

        Valore atteso:
        - Campo 'guidelines' presente e non vuoto.
        """
        response = requests.post(f"{API_URL}/predict", json=base_incident_data)
        data = response.json()
        assert "guidelines" in data
        assert isinstance(data["guidelines"], list)
        assert len(data["guidelines"]) > 0

    @pytest.mark.parametrize(
        "scenario_name,incident_data",
        [
            (
                "Incendio Veicolo - Alta gravità su autostrada",
                {
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
                    "Turning_Loop": False,
                },
            ),
            (
                "Tamponamento - Media gravità con semaforo",
                {
                    "Severity": "Medium",
                    "Incident_Type": "Tamponamento",
                    "Road_Type": "primary_secondary",
                    "Daylight": True,
                    "Bump": False,
                    "Crossing": False,
                    "Give_Way": False,
                    "Junction": True,
                    "Railway": False,
                    "Roundabout": False,
                    "Stop": False,
                    "Traffic_Signal": True,
                    "Turning_Loop": False,
                },
            ),
            (
                "Investimento - Alta gravità zona residenziale",
                {
                    "Severity": "High",
                    "Incident_Type": "Investimento",
                    "Road_Type": "residential",
                    "Daylight": True,
                    "Bump": True,
                    "Crossing": True,
                    "Give_Way": False,
                    "Junction": False,
                    "Railway": False,
                    "Roundabout": False,
                    "Stop": False,
                    "Traffic_Signal": False,
                    "Turning_Loop": False,
                },
            ),
            (
                "Veicolo Fuori Strada - Notte su strada extraurbana",
                {
                    "Severity": "Medium",
                    "Incident_Type": "Veicolo Fuori Strada",
                    "Road_Type": "tertiary",
                    "Daylight": False,
                    "Bump": False,
                    "Crossing": False,
                    "Give_Way": False,
                    "Junction": False,
                    "Railway": False,
                    "Roundabout": False,
                    "Stop": False,
                    "Traffic_Signal": False,
                    "Turning_Loop": False,
                },
            ),
            (
                "Collisione con ostacolo - Bassa gravità con stop",
                {
                    "Severity": "Low",
                    "Incident_Type": "Collisione con ostacolo",
                    "Road_Type": "service",
                    "Daylight": True,
                    "Bump": False,
                    "Crossing": False,
                    "Give_Way": True,
                    "Junction": True,
                    "Railway": False,
                    "Roundabout": False,
                    "Stop": True,
                    "Traffic_Signal": False,
                    "Turning_Loop": False,
                },
            ),
            (
                "Passaggio a livello - Gravità sconosciuta",
                {
                    "Severity": "Unknown",
                    "Incident_Type": None,
                    "Road_Type": "unclassified",
                    "Daylight": True,
                    "Bump": False,
                    "Crossing": False,
                    "Give_Way": False,
                    "Junction": False,
                    "Railway": True,
                    "Roundabout": False,
                    "Stop": False,
                    "Traffic_Signal": False,
                    "Turning_Loop": False,
                },
            ),
        ],
    )
    def test_predict_various_scenarios(
        self, api_available, scenario_name, incident_data
    ):
        """
        Scopo: Verifica predizioni per vari scenari di incidente.

        Parametri:
        - scenario_name (str): Nome descrittivo dello scenario.
        - incident_data (dict): Dati dell'incidente.

        Valore atteso:
        - Status code 200 e guidelines non vuote.
        """
        response = requests.post(f"{API_URL}/predict", json=incident_data)
        assert response.status_code == 200, f"Fallito: {scenario_name}"
        data = response.json()
        assert "guidelines" in data, f"Guidelines mancanti: {scenario_name}"
        assert len(data["guidelines"]) > 0, f"Guidelines vuote: {scenario_name}"


class TestPredictValidation:
    """Test di validazione input per l'endpoint predict."""

    def test_predict_missing_required_field(self, api_available):
        """
        Scopo: Verifica che manchi un campo obbligatorio restituisca errore.

        Valore atteso:
        - Status code 422 (Unprocessable Entity).
        """
        incomplete_data = {
            "Severity": "High",
            # Mancano altri campi obbligatori
        }
        response = requests.post(f"{API_URL}/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_predict_invalid_boolean_type(self, api_available):
        """
        Scopo: Verifica che un tipo invalido per campo booleano restituisca errore.

        Valore atteso:
        - Status code 422 (Unprocessable Entity).
        """
        invalid_data = {
            "Severity": "High",
            "Incident_Type": "Tamponamento",
            "Road_Type": "primary_secondary",
            "Daylight": "not_a_boolean",  # Tipo invalido
            "Bump": False,
            "Crossing": False,
            "Give_Way": False,
            "Junction": False,
            "Railway": False,
            "Roundabout": False,
            "Stop": False,
            "Traffic_Signal": False,
            "Turning_Loop": False,
        }
        response = requests.post(f"{API_URL}/predict", json=invalid_data)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])