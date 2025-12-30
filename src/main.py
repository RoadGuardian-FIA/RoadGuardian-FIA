"""
FastAPI main application
Provides endpoints for incident classification and protocol retrieval
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import pickle
import json
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize FastAPI app
app = FastAPI(
    title="AI Behavioral Guidelines API",
    description="API per la classificazione degli incidenti e il recupero dei protocolli comportamentali",
    version="1.0.0"
)

# Global variables for model and data
model = None
label_encoders = None
rules = None


class IncidentInput(BaseModel):
    """Input model for incident data"""
    incident_type: str = Field(..., description="Tipo di incidente (accident, violation)")
    severity: str = Field(..., description="Gravità (low, medium, high)")
    location_type: str = Field(..., description="Tipo di località (highway, urban, rural)")
    weather: str = Field(..., description="Condizioni meteo (clear, rain, snow, fog)")
    time_of_day: str = Field(..., description="Momento della giornata (day, night)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "incident_type": "accident",
                "severity": "high",
                "location_type": "highway",
                "weather": "rain",
                "time_of_day": "night"
            }
        }


class ProtocolResponse(BaseModel):
    """Response model for protocol information"""
    protocol_id: int = Field(..., description="ID del protocollo predetto")
    protocol_name: str = Field(..., description="Nome del protocollo")
    description: str = Field(..., description="Descrizione del protocollo")
    guidelines: List[str] = Field(..., description="Linee guida comportamentali")


def load_model():
    """Load the trained model and label encoders"""
    global model, label_encoders
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    encoders_path = os.path.join(base_dir, 'models', 'label_encoders.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please run train_compare.py first to train and save the model."
        )
    
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(
            f"Label encoders file not found: {encoders_path}. "
            "Please run train_compare.py first to save the label encoders."
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)
    
    print("Model and label encoders loaded successfully")


def load_rules():
    """Load the rules/protocols from JSON file"""
    global rules
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rules_path = os.path.join(base_dir, 'data', 'rules.json')
    
    if not os.path.exists(rules_path):
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    
    print(f"Rules loaded: {len(rules)} protocols")


@app.on_event("startup")
async def startup_event():
    """
    Initialize the application on startup
    Note: Using @app.on_event("startup") for broader compatibility.
    For FastAPI 0.109+, consider migrating to lifespan context manager.
    """
    try:
        load_model()
        load_rules()
        print("API ready to serve requests")
    except Exception as e:
        print(f"Error during startup: {e}")
        print("API will start but may not function correctly until models are trained")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Behavioral Guidelines API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - POST endpoint to classify incidents",
            "protocols": "/protocols - GET endpoint to list all protocols",
            "health": "/health - GET endpoint to check API health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "rules_loaded": rules is not None,
        "label_encoders_loaded": label_encoders is not None
    }


@app.get("/protocols", response_model=Dict[str, Dict])
async def get_protocols():
    """Get all available protocols"""
    if rules is None:
        raise HTTPException(status_code=500, detail="Rules not loaded")
    
    return rules


@app.post("/predict", response_model=ProtocolResponse)
async def predict_protocol(incident: IncidentInput):
    """
    Predict the appropriate protocol for a given incident
    
    Takes incident details and returns the predicted protocol with guidelines
    """
    if model is None or label_encoders is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please train the model first by running train_compare.py"
        )
    
    if rules is None:
        raise HTTPException(status_code=500, detail="Rules not loaded")
    
    try:
        # Prepare input data in the same order as training
        features = ['incident_type', 'severity', 'location_type', 'weather', 'time_of_day']
        input_data = []
        
        incident_dict = incident.dict()
        
        for feature in features:
            value = incident_dict[feature]
            
            # Encode using the saved label encoder
            if feature in label_encoders:
                encoder = label_encoders[feature]
                # Handle unknown categories
                if value not in encoder.classes_:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown value '{value}' for feature '{feature}'. "
                               f"Valid values: {list(encoder.classes_)}"
                    )
                encoded_value = encoder.transform([value])[0]
                input_data.append(encoded_value)
            else:
                input_data.append(value)
        
        # Make prediction
        X = np.array([input_data])
        protocol_id = int(model.predict(X)[0])
        
        # Get protocol information from rules
        protocol_key = str(protocol_id)
        if protocol_key not in rules:
            raise HTTPException(
                status_code=404,
                detail=f"Protocol ID {protocol_id} not found in rules"
            )
        
        protocol_info = rules[protocol_key]
        
        return ProtocolResponse(
            protocol_id=protocol_id,
            protocol_name=protocol_info['protocol_name'],
            description=protocol_info['description'],
            guidelines=protocol_info['guidelines']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
