import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Text Classification API",
    description="API for classifying text using facebook/bart-large-mnli model",
    version="1.0.0"
)

# Global variable to store candidate labels
CANDIDATE_LABELS = []

# Request model
class ClassificationRequest(BaseModel):
    text: str

# Response model
class ClassificationResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float

def load_config():
    """Load candidate labels from config file"""
    try:
        config_path = Path("classifiers.config")
        if not config_path.exists():
            raise FileNotFoundError("classifiers.config file not found")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        if not isinstance(config.get('candidate_labels'), list):
            raise ValueError("Config file must contain 'candidate_labels' as a list")
            
        return config['candidate_labels']
    
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing config file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading config: {str(e)}")

# Startup event to initialize the model and load config
@app.on_event("startup")
async def startup_event():
    global CANDIDATE_LABELS
    try:
        # Load candidate labels from config
        CANDIDATE_LABELS = load_config()
        print(f"Loaded {len(CANDIDATE_LABELS)} candidate labels")
        
        # Initialize the classifier
        global classifier
        classifier = pipeline(model="facebook/bart-large-mnli")
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Startup error: {str(e)}")
        raise

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    if not CANDIDATE_LABELS:
        raise HTTPException(
            status_code=500,
            detail="Candidate labels not loaded. Check configuration."
        )
        
    try:
        # Run classification
        result = classifier(
            request.text,
            candidate_labels=CANDIDATE_LABELS
        )
        
        # Get the highest scoring label and its score
        top_prediction_idx = result['scores'].index(max(result['scores']))
        print("XXX"*10)
        print(top_prediction_idx)
        print("XXX"*10)
        # Format response
        response = ClassificationResponse(
            text=result["sequence"],
            predicted_label=result["labels"][top_prediction_idx],
            confidence=result["scores"][top_prediction_idx]
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "config_loaded": bool(CANDIDATE_LABELS),
        "num_labels": len(CANDIDATE_LABELS)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
