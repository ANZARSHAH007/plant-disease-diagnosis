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

## 🎓 For Someone Training from Scratch

**Your friend will follow these steps to train the models:**

### Step 1: Clone the Repository
```bash
git clone https://github.com/ANZARSHAH007/plant-disease-diagnosis.git
cd plant-disease-diagnosis
```

### Step 2: Get the Dataset
**IMPORTANT:** You need to provide or help them download the plant disease datasets:

1. **PlantDoc Dataset:** https://github.com/pratikkayal/PlantDoc-Dataset
2. **PlantVillage Dataset:** https://www.kaggle.com/datasets/emmarex/plantdisease

After downloading, organize like this:
```
plant-disease-diagnosis/
└── dataset/
    └── crops/
        ├── plantdoc_train/     # Training images (28 classes)
        │   ├── Apple leaf/
        │   ├── Apple rust leaf/
        │   ├── Tomato leaf/
        │   └── ... (25 more classes)
        └── plantdoc_test/      # Testing images (28 classes)
            └── (same structure)
```

**Dataset Stats:**
- Training samples: ~7,008 images
- Testing samples: ~708 images
- Total classes: 28 plant diseases

### Step 3: Set Up Python Environment
```bash
# Create conda environment (recommended)
conda create -n agri_dl python=3.10
conda activate agri_dl

# Install all dependencies
pip install -r requirements.txt
```

**Dependencies installed:**
- PyTorch (CPU version)
- Torchvision  
- Ultralytics (YOLOv8)
- FastAPI & Uvicorn
- OpenCV, Pillow, Pandas, Matplotlib, etc.

### Step 4: Train the Models

**Option A: Use the Training Script (Easiest)**
```bash
python retrain.py
```
- Trains for 15 epochs
- Takes ~45-60 minutes on CPU
- Saves models automatically to `models/` folder
- Expected accuracy: 60-70%

**Option B: Use Jupyter Notebook (Recommended for Learning)**
1. Open `notebooks/03_model_training.ipynb`
2. Run each cell in order
3. You'll see:
   - Dataset loading and exploration
   - Model architecture
   - Training progress with visualizations
   - Test results

**Training Configuration:**
```python
DEMO_EPOCHS = 15        # For 60-70% accuracy (45 mins)
# DEMO_EPOCHS = 30      # For 80%+ accuracy (90 mins)

SEG_EPOCHS = 30         # YOLOv8 segmentation epochs
```

### Step 5: Run the Web Application
```bash
python app.py
```
Then open: `http://localhost:8000/static/index.html`

---

## 🛠️ Troubleshooting for Your Friend

### Issue: "Dataset not found"
**Solution:** Make sure the dataset folder structure matches exactly:
```
dataset/crops/plantdoc_train/Apple leaf/image001.jpg
dataset/crops/plantdoc_test/Apple leaf/image002.jpg
```

### Issue: "CUDA out of memory" or slow training
**Solution:** The code uses CPU by default. Training 15 epochs takes ~45 minutes. This is normal.

### Issue: "ModuleNotFoundError"
**Solution:** 
```bash
conda activate agri_dl
pip install -r requirements.txt
```

### Issue: "Models not loading in app.py"
**Solution:** After training, verify files exist:
```bash
# Windows
dir models\disease_classifier.pth
dir models\yolov8_seg\weights\best.pt

# Linux/Mac  
ls models/disease_classifier.pth
ls models/yolov8_seg/weights/best.pt
```

### Issue: Low accuracy (<30%)
**Solution:** Increase epochs in the notebook or retrain.py:
- Change `DEMO_EPOCHS = 3` to `DEMO_EPOCHS = 15` or higher
- More epochs = better accuracy but longer training

---

## 📝 What to Tell Your Friend

**Send them this message:**
> "Hey! I've created a plant disease diagnosis system. Here's how to run it:
> 
> 1. Clone: `git clone https://github.com/ANZARSHAH007/plant-disease-diagnosis.git`
> 2. Download the PlantDoc dataset (links in README)
> 3. Put images in `dataset/crops/plantdoc_train/` and `plantdoc_test/`
> 4. Create environment: `conda create -n agri_dl python=3.10`
> 5. Install: `conda activate agri_dl` then `pip install -r requirements.txt`
> 6. Train: Open `notebooks/03_model_training.ipynb` and run all cells (~1 hour)
> 7. Run app: `python app.py` and visit http://localhost:8000/static/index.html
>
> The notebook explains everything step-by-step. Reach out if you get stuck!"

---

## ⚡ Alternative: Use Pre-trained Models (Skip Training)

If you share your trained models with them:
1. Share the `models/` folder (50 MB total)
2. They place it in the project root
3. Run `python app.py` directly
4. No training needed!

---

## 🛠️ Troubleshooting for Your Friend

### 1. Clone Repository
```bash
git clone https://github.com/ANZARSHAH007/plant-disease-diagnosis.git
cd plant-disease-diagnosis
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n plant_disease python=3.10
conda activate plant_disease

# Or using venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
You'll need the PlantDoc and PlantVillage datasets:
- Place training images in `dataset/crops/plantdoc_train/`
- Place test images in `dataset/crops/plantdoc_test/`

## 📚 Training the Models

### Quick Training Script (Recommended)
```bash
python retrain.py
```
This will train for 15 epochs (~45 minutes on CPU) and achieve 60-80% accuracy.

### Notebook Training (For Learning/Experimentation)
Open `notebooks/03_model_training.ipynb` and run all cells to:
1. Install required packages
2. Train the disease classifier (ResNet18)
3. Train the segmentation model (YOLOv8)
4. Test the complete pipeline

**Training Tips:**
- **Demo (fast, low accuracy):** `DEMO_EPOCHS = 3` → ~21% accuracy in 10 minutes
- **Production (recommended):** `DEMO_EPOCHS = 15` → ~60-70% accuracy in 45 minutes  
- **High quality:** `DEMO_EPOCHS = 30` → ~80%+ accuracy in 90 minutes
- YOLOv8 segmentation: Use 30-50 epochs for best results

### Sharing Models with Friends
After training, share these files:
```bash
# Zip the models folder
tar -czf models.tar.gz models/

# Or use cloud storage (Google Drive, Dropbox, etc.)
# Upload: models/disease_classifier.pth (~44 MB)
# Upload: models/yolov8_seg/weights/best.pt (~6 MB)
```

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
