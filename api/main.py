"""
Line Check API
FastAPI application for insulator damage detection
"""

import io
import os
import sys
import traceback
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from fastai.vision.all import load_learner, PILImage
from torchvision import transforms

# Initialize FastAPI app
app = FastAPI(
    title="Line Check API",
    description="API for detecting damaged insulators in power transmission lines",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables
MODEL_PATH = os.path.join(os.getcwd(), "insulator_classifier.pkl")
model = None
model_load_error = None

# Image preprocessing
def preprocess_image(img: Image.Image) -> PILImage:
    """Preprocess image for model input."""
    # Ensure image is RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model's expected size (224x224 for ResNet)
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    return PILImage.create(img)

@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model, model_load_error
    try:
        if not os.path.exists(MODEL_PATH):
            model_load_error = f"Model file not found at {MODEL_PATH}"
            print(f"Error: {model_load_error}")
            return
            
        print(f"Loading model from {MODEL_PATH}")
        model = load_learner(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
        model_load_error = None
    except Exception as e:
        model_load_error = f"Error loading model: {str(e)}"
        print(model_load_error)
        print(traceback.format_exc())

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Line Check API",
        "version": "1.0.0",
        "description": "AI-powered insulator damage detection"
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(0.5, description="Minimum confidence threshold (0-1)")
):
    """
    Predict whether an insulator is damaged from an uploaded image.
    
    Args:
        file: Uploaded image file
        confidence_threshold: Minimum confidence threshold for prediction (0-1)
    
    Returns:
        JSON with prediction and confidence score
    """
    # Check if model is loaded
    if model is None:
        error_msg = model_load_error or "Model not loaded. Please check server logs."
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

    # Validate file exists
    if not file:
        raise HTTPException(
            status_code=422,
            detail="No file uploaded. Please provide an image file."
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file."
        )
    
    # Validate confidence threshold
    threshold = 0.5 if confidence_threshold is None else confidence_threshold
    if not 0 <= threshold <= 1:
        raise HTTPException(
            status_code=422,
            detail="Confidence threshold must be between 0 and 1"
        )
    
    try:
        # Read and convert image
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=422,
                detail="Empty file uploaded. Please provide a valid image."
            )
            
        try:
            # Open image and preprocess
            img = Image.open(io.BytesIO(contents))
            img = preprocess_image(img)
        except Exception as e:
            print(f"Image processing error: {e}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=422,
                detail=f"Failed to process image: {str(e)}"
            )
        
        # Make prediction
        try:
            pred, pred_idx, probs = model.predict(img)
            confidence = float(probs[pred_idx])
            
            # Apply confidence threshold
            if confidence < threshold:
                prediction = "Uncertain"
            else:
                prediction = str(pred)
            
            return JSONResponse({
                "filename": file.filename,
                "prediction": prediction,
                "confidence": confidence,
                "confidence_threshold": threshold
            })
            
        except Exception as e:
            print(f"Prediction error: {e}")
            print(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Model prediction failed: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if model_load_error is None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "error": model_load_error,
        "cwd": os.getcwd()
    } 