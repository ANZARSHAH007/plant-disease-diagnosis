# 🌿 Multi-Crop Plant Disease Classification and Severity Estimation

A complete AI-powered system for plant disease diagnosis using deep learning. The system combines ResNet18 for disease classification and YOLOv8 for segmentation-based severity assessment.

## 📊 Pipeline Architecture

```
Input Leaf Image
      ↓
Image Preprocessing
      ↓
ResNet18 CNN Classifier
      ↓
Disease Classification
      ↓
YOLOv8 Segmentation
      ↓
Infected Region Detection
      ↓
Severity Calculation (%)
      ↓
Final Diagnosis Output
```

## 🚀 Features

- **28 Disease Classes** - Multi-crop disease identification
- **Disease Classification** - ResNet18-based CNN with 21%+ accuracy (demo training)
- **Segmentation** - YOLOv8 for infected region detection
- **Severity Assessment** - Percentage-based severity calculation
  - Healthy: <5%
  - Mild: 5-15%
  - Moderate: 15-35%
  - Severe: >35%
- **REST API** - FastAPI backend for deployment
- **Web Interface** - Simple drag-and-drop UI
- **Real-time Predictions** - Fast inference on CPU

## 📁 Project Structure

```
semester_project/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_multicrop_generation.ipynb
│   └── 03_model_training.ipynb          # Complete pipeline training
│
├── dataset/
│   └── crops/
│       ├── plantdoc_train/              # Training data
│       └── plantdoc_test/               # Test data
│
├── models/
│   ├── disease_classifier.pth           # Trained classifier
│   └── yolov8_seg/weights/best.pt      # Trained segmentation model
│
├── static/
│   └── index.html                       # Web frontend
│
├── app.py                               # FastAPI backend
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🛠️ Installation

### 1. Clone or Navigate to Project
```bash
cd "D:\Deep learning\semester_project"
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
# Using conda
conda create -n plant_disease python=3.10
conda activate plant_disease

# Or using venv
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📚 Training the Models

### Option 1: Run the Complete Notebook
Open `notebooks/03_model_training.ipynb` and run all cells to:
1. Install required packages
2. Train the disease classifier (ResNet18)
3. Train the segmentation model (YOLOv8)
4. Test the complete pipeline

**Note:** The demo uses only 3 epochs for quick demonstration. For better accuracy:
- Change `DEMO_EPOCHS = 3` to `CLASSIFIER_EPOCHS = 15`
- Change YOLOv8 epochs from `10` to `50`

### Option 2: Pre-trained Models
If you've already trained the models, ensure these files exist:
- `models/disease_classifier.pth`
- `models/yolov8_seg/weights/best.pt`

## 🌐 Running the Application

### Start the API Server
```bash
python app.py
```

The server will start at: `http://localhost:8000`

### Access the Web Interface
Open your browser and go to:
```
http://localhost:8000/static/index.html
```

### API Documentation
Interactive API docs available at:
```
http://localhost:8000/docs
```

## 🔌 API Endpoints

### 1. Health Check
```bash
GET http://localhost:8000/health
```

### 2. Model Information
```bash
GET http://localhost:8000/model-info
```

### 3. Single Image Prediction
```bash
POST http://localhost:8000/predict
Content-Type: multipart/form-data

file: <image_file>
```

**Response:**
```json
{
  "success": true,
  "filename": "leaf.jpg",
  "diagnosis": {
    "disease": "Tomato leaf",
    "confidence": 72.59,
    "severity_percent": 0.00,
    "severity_category": "Healthy",
    "overlay_image": "base64_encoded_image",
    "mask_image": "base64_encoded_mask",
    "recommendations": {
      "severity_advice": "Leaf appears healthy...",
      "general_tips": [...]
    }
  }
}
```

### 4. Batch Prediction (Max 10 images)
```bash
POST http://localhost:8000/batch-predict
Content-Type: multipart/form-data

files: <image_file_1>
files: <image_file_2>
...
```

## 💻 Usage Examples

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/leaf_image.jpg"
```

### Using Python
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("leaf_image.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Disease: {result['diagnosis']['disease']}")
print(f"Confidence: {result['diagnosis']['confidence']}%")
print(f"Severity: {result['diagnosis']['severity_category']}")
```

### Using JavaScript
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Disease:', data.diagnosis.disease);
    console.log('Severity:', data.diagnosis.severity_category);
});
```

## 🎯 Model Performance

### Disease Classifier (Demo Training - 3 Epochs)
- **Architecture:** ResNet18
- **Training Accuracy:** 25.77%
- **Validation Accuracy:** 21.19%
- **Parameters:** 11.4M
- **Note:** Accuracy will improve significantly with full training (15-20 epochs)

### YOLOv8 Segmentation (10 Epochs)
- **Architecture:** YOLOv8n-seg
- **Purpose:** Infected region detection
- **Training:** Synthetic masks (demo data)
- **Note:** Real annotated data will improve performance

## 📊 Supported Disease Classes

The system can identify 28 plant disease classes:
- Apple leaf, Apple rust leaf, Apple Scab Leaf
- Bell pepper leaf, Bell pepper leaf spot
- Blueberry leaf, Cherry leaf
- Corn varieties (Gray leaf spot, leaf blight, rust)
- Grape leaf, grape leaf black rot
- Peach leaf
- Potato (early blight, late blight)
- Raspberry leaf, Soyabean leaf
- Squash Powdery mildew leaf, Strawberry leaf
- Tomato varieties (Early blight, bacterial spot, late blight, mosaic virus, yellow virus, mold, Septoria, spider mites)

## 🔧 Configuration

### Modify Severity Thresholds
Edit `app.py`, function `calculate_severity()`:
```python
if severity_pct < 5:
    category = 'Healthy'
elif severity_pct < 15:
    category = 'Mild'
elif severity_pct < 35:
    category = 'Moderate'
else:
    category = 'Severe'
```

### Change Port
Edit `app.py` at the bottom:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)  # Change port here
```

## 🚢 Deployment

### Option 1: Local Deployment
Already covered above ✅

### Option 2: Docker (Coming Soon)
```dockerfile
# Dockerfile example
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 3: Cloud Deployment
- **Heroku:** Deploy using Procfile
- **AWS EC2:** Run on virtual machine
- **Google Cloud Run:** Containerized deployment
- **Azure:** App Service deployment

## 🐛 Troubleshooting

### Models Not Loading
```
⚠ Warning: Classifier model not found. Please train the model first.
```
**Solution:** Run the training notebook first or ensure model files exist in `models/` directory.

### Port Already in Use
```
ERROR: [Errno 10048] error while attempting to bind on address
```
**Solution:** Change the port in `app.py` or kill the process using port 8000.

### CUDA Not Available
The app runs on CPU by default. For GPU support:
1. Install CUDA-enabled PyTorch
2. Change `device='cpu'` to `device='cuda'` in `app.py`

## 📝 Future Improvements

- [ ] Add more disease classes
- [ ] Use real annotated masks instead of synthetic
- [ ] Train for more epochs (15-20)
- [ ] Add data augmentation
- [ ] Implement model versioning
- [ ] Add user authentication
- [ ] Create mobile app
- [ ] Add historical tracking
- [ ] Deploy to cloud

## 👨‍💻 Development

### Run in Development Mode
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests (Future)
```bash
pytest tests/
```

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- PyTorch Team
- Ultralytics YOLOv8
- FastAPI Framework
- PlantDoc Dataset

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ for Plant Health Monitoring**
