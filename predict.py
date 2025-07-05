#!/usr/bin/env python3
"""
Insulator Vision - Quick Inference Script
Loads a trained FastAI model and makes predictions on new images.
"""

import sys
from pathlib import Path
from typing import Union, Tuple, List
import argparse

from fastai.vision.all import load_learner, PILImage
import torch
import numpy as np


def load_model(model_path: str = "models/checkpoints/insulator_classifier.pkl") -> object:
    """Load the trained FastAI model."""
    try:
        return load_learner(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def predict_image(
    image_path: Union[str, Path],
    model_path: str = "models/checkpoints/insulator_classifier.pkl",
    confidence_threshold: float = 0.5
) -> Tuple[str, float]:
    """
    Predict whether an insulator is damaged or normal.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
        confidence_threshold: Minimum confidence threshold for prediction
    
    Returns:
        Tuple of (prediction, confidence)
    """
    try:
        # Load image
        img = PILImage.create(image_path)
        
        # Load model and predict
        learn = load_model(model_path)
        pred, pred_idx, probs = learn.predict(img)
        confidence = float(probs[pred_idx])
        
        # Apply confidence threshold
        if confidence < confidence_threshold:
            return "Uncertain", confidence
        
        return str(pred), confidence
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0


def predict_batch(
    image_paths: List[Union[str, Path]],
    model_path: str = "models/checkpoints/insulator_classifier.pkl",
    confidence_threshold: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Predict multiple images at once.
    
    Args:
        image_paths: List of paths to image files
        model_path: Path to the trained model
        confidence_threshold: Minimum confidence threshold for prediction
    
    Returns:
        List of (prediction, confidence) tuples
    """
    learn = load_model(model_path)
    results = []
    
    for path in image_paths:
        pred, conf = predict_image(path, model_path, confidence_threshold)
        results.append((path, pred, conf))
    
    return results


def main():
    """Command-line interface for the prediction script."""
    parser = argparse.ArgumentParser(
        description="Predict insulator damage from images using trained FastAI model."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to image file(s) to predict"
    )
    parser.add_argument(
        "--model",
        default="models/checkpoints/insulator_classifier.pkl",
        help="Path to trained model file (default: models/checkpoints/insulator_classifier.pkl)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for prediction (default: 0.5)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process images in batch mode"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        results = predict_batch(args.images, args.model, args.threshold)
        print("\nBatch Prediction Results:")
        print("-" * 60)
        for path, pred, conf in results:
            print(f"Image: {path}")
            print(f"Prediction: {pred} (Confidence: {conf:.2%})")
            print("-" * 60)
    else:
        for image_path in args.images:
            pred, conf = predict_image(image_path, args.model, args.threshold)
            print(f"\nImage: {image_path}")
            print(f"Prediction: {pred}")
            print(f"Confidence: {conf:.2%}")


if __name__ == "__main__":
    main() 