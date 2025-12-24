"""
Plant Disease Diagnosis API
FastAPI backend for the complete diagnosis pipeline
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import cv2
import numpy as np
import io
import base64
from pathlib import Path
import json

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Diagnosis API",
    description="Multi-Crop Plant Disease Classification and Severity Estimation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Disease Classifier Model
class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class PlantDiseaseDiagnosisSystem:
    def __init__(self, classifier_path, seg_model_path, device='cpu'):
        self.device = device
        
        # Load classifier
        checkpoint = torch.load(classifier_path, map_location=device)
        self.class_names = checkpoint['class_names']
        self.classifier = DiseaseClassifier(len(self.class_names))
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.to(device)
        self.classifier.eval()
        
        # Load YOLO segmentation
        self.seg_model = YOLO(seg_model_path)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Diagnosis system loaded with {len(self.class_names)} classes")
    
    def predict_disease(self, image_pil):
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, 1)
        return self.class_names[idx.item()], conf.item() * 100
    
    def segment(self, image_np):
        results = self.seg_model(image_np, conf=0.25, verbose=False)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            combined = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
            for mask in masks:
                resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
                combined = np.maximum(combined, (resized * 255).astype(np.uint8))
            return combined
        return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    
    def calculate_severity(self, mask):
        if mask is None or mask.size == 0:
            return 0.0, 'Healthy'
        
        total_pixels = mask.shape[0] * mask.shape[1]
        infected_pixels = np.sum(mask > 0)
        severity_pct = (infected_pixels / total_pixels) * 100
        
        if severity_pct < 5:
            category = 'Healthy'
        elif severity_pct < 15:
            category = 'Mild'
        elif severity_pct < 35:
            category = 'Moderate'
        else:
            category = 'Severe'
        
        return severity_pct, category
    
    def diagnose(self, image_bytes):
        # Read image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Step 1: Disease classification
        disease, confidence = self.predict_disease(image)
        
        # Step 2: Segmentation
        mask = self.segment(image_np)
        
        # Step 3: Severity calculation
        severity_pct, severity_cat = self.calculate_severity(mask)
        
        # Create overlay visualization
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_np, 0.6, mask_colored, 0.4, 0)
        
        # Convert images to base64
        _, buffer = cv2.imencode('.jpg', overlay)
        overlay_b64 = base64.b64encode(buffer).decode('utf-8')
        
        _, buffer = cv2.imencode('.jpg', mask)
        mask_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'disease': disease,
            'confidence': round(confidence, 2),
            'severity_percent': round(severity_pct, 2),
            'severity_category': severity_cat,
            'overlay_image': overlay_b64,
            'mask_image': mask_b64,
            'recommendations': self._get_recommendations(disease, severity_cat)
        }
    
    def _get_recommendations(self, disease, severity):
        """Get treatment recommendations based on diagnosis"""
        base_rec = {
            'Healthy': "Leaf appears healthy. Continue regular monitoring.",
            'Mild': "Early signs detected. Monitor closely and consider preventive measures.",
            'Moderate': "Moderate infection. Apply appropriate fungicide/pesticide treatment.",
            'Severe': "Severe infection. Immediate treatment required. Consider removing affected parts."
        }
        
        return {
            'severity_advice': base_rec[severity],
            'general_tips': [
                "Ensure proper spacing between plants",
                "Avoid overhead watering",
                "Remove and destroy infected plant debris",
                "Apply organic or chemical treatments as needed"
            ]
        }


# Global model instance
diagnosis_system = None


def load_models():
    """Load models on startup"""
    global diagnosis_system
    try:
        classifier_path = Path("models/disease_classifier.pth")
        seg_model_path = Path("models/yolov8_seg/weights/best.pt")
        
        if not classifier_path.exists():
            print("⚠ Warning: Classifier model not found. Please train the model first.")
            return
        
        if not seg_model_path.exists():
            print("⚠ Warning: Segmentation model not found. Please train the model first.")
            return
        
        diagnosis_system = PlantDiseaseDiagnosisSystem(
            classifier_path=str(classifier_path),
            seg_model_path=str(seg_model_path),
            device='cpu'
        )
        print("✅ Models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Plant Disease Diagnosis API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "API documentation",
            "/predict": "Diagnose plant disease (POST)",
            "/health": "Health check",
            "/model-info": "Model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": diagnosis_system is not None
    }


@app.get("/model-info")
async def model_info():
    """Get model information"""
    if diagnosis_system is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "num_classes": len(diagnosis_system.class_names),
        "classes": diagnosis_system.class_names,
        "severity_levels": ["Healthy", "Mild", "Moderate", "Severe"],
        "pipeline": [
            "Image Preprocessing",
            "ResNet18 Disease Classification",
            "YOLOv8 Segmentation",
            "Severity Calculation",
            "Diagnosis Output"
        ]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Diagnose plant disease from uploaded image
    
    Returns:
    - disease: Predicted disease name
    - confidence: Prediction confidence (%)
    - severity_percent: Infected area percentage
    - severity_category: Health status (Healthy/Mild/Moderate/Severe)
    - overlay_image: Base64 encoded overlay visualization
    - mask_image: Base64 encoded segmentation mask
    - recommendations: Treatment recommendations
    """
    if diagnosis_system is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please ensure models are trained and available."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Run diagnosis
        result = diagnosis_system.diagnose(image_bytes)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "diagnosis": result
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Diagnose multiple images at once"""
    if diagnosis_system is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
    
    results = []
    for file in files:
        try:
            image_bytes = await file.read()
            result = diagnosis_system.diagnose(image_bytes)
            results.append({
                "filename": file.filename,
                "success": True,
                "diagnosis": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})


if __name__ == "__main__":
    import uvicorn
    load_models()  # Load models before starting
    uvicorn.run(app, host="0.0.0.0", port=8000)
