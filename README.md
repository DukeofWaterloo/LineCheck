# 🔍⚡ Line Check: Deep Learning for Power Infrastructure Maintenance

![Line Check Banner](https://dummyimage.com/1200x300/222/FFFF00&text=Line+Check:+Empowering+Power+Maintenance)


## 📋 Overview
Line Check is a deep learning project that automatically detects and classifies damaged electrical insulators in power transmission systems. Using state-of-the-art computer vision techniques with FastAI and PyTorch, this model helps maintain critical power infrastructure by identifying potential failures before they occur.

## 🚀 Technical Highlights
- **Architecture**: Transfer learning with ResNet34
- **Framework**: FastAI/PyTorch
- **Dataset**: 1.6k labeled images of transmission insulators
- **Categories**: Normal and Damaged insulator classification
- **Training**: Fine-tuning approach with progressive resizing
- **Deployment**: Docker containerization for easy deployment

## 📁 Project Structure
```
Line Check/
├── notebooks/              # Jupyter notebooks
│   ├── insulator_classification.ipynb  # Main training notebook
│   └── archive/           # Previous versions
├── models/                # Trained models
│   └── checkpoints/      # Model checkpoints
├── api/                  # FastAPI application
├── Train/                 # Training data
│   ├── Images/           # Image dataset
│   └── labels_v1.2.json  # Image annotations
├── Dockerfile            # Docker configuration
├── .dockerignore         # Docker build exclusions
├── requirements.txt       # Python dependencies
└── README.md
```

## 🏁 Getting Started

### 🐳 Using Docker (Recommended)
1. Build the Docker image:
```bash
docker build -t line-check .
```

2. Run the container:
```bash
docker run -d -p 8000:8000 --name line-check-api line-check
```

3. The API will be available at `http://localhost:8000`

### 💻 Manual Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 🔧 Running the Model
1. Clone the repository
2. Install dependencies as above
3. Open `notebooks/insulator_classification.ipynb` in Jupyter
4. Run all cells to train the model

### 🌐 Using the API
1. Start the FastAPI server:
```bash
# If using Docker (recommended):
docker run -d -p 8000:8000 --name line-check-api line-check

# If running locally:
uvicorn api.main:app --reload
```
2. Visit `http://localhost:8000/docs` for interactive API documentation
3. Use the endpoints as described below

#### 📡 API Endpoints

##### 1. Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns API information and status
- **Example**:
```bash
curl http://localhost:8000/
```
```json
{
    "name": "Line Check API",
    "version": "1.0.0",
    "description": "AI-powered insulator damage detection"
}
```

##### 2. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Checks API and model health
- **Example**:
```bash
curl http://localhost:8000/health
```
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_path": "insulator_classifier.pkl"
}
```

##### 3. Predict Endpoint
- **URL**: `/predict`
- **Method**: `POST`
- **Description**: Analyzes an insulator image and returns damage prediction
- **Parameters**:
  - `file`: Image file (required)
  - `confidence_threshold`: Float between 0-1 (default: 0.5)
- **Example**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg" \
  -F "confidence_threshold=0.5"
```
```json
{
    "filename": "image.jpg",
    "prediction": "damaged",
    "confidence": 0.95,
    "confidence_threshold": 0.5
}
```

## 📊 Model Performance
The model achieves strong performance in identifying damaged insulators:
- High accuracy in distinguishing between normal and damaged insulators
- Robust performance across different lighting conditions and angles
- Fast inference time suitable for real-world applications

## 🔮 Future Improvements
- Implement real-time detection
- Add multi-class classification for different types of damage
- Add CI/CD pipeline
- Edge device deployment for field inspections
- Kubernetes deployment configuration

## 👥 Contributing
Feel free to open issues or submit pull requests. All contributions are welcome!

## 🙏 Acknowledgments
- FastAI community for the excellent deep learning framework
