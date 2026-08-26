import torch
import joblib
from pathlib import Path
from model_architecture import AttentionLSTM

# Относительный путь к папке models (относительно этого файла)
BASE_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = BASE_DIR / "models"

class ModelManager:
    def __init__(self):
        self.detector_model = None
        self.segmenter_model = None
        self.clf_model = None
        self.detector_scaler = None
        self.segmenter_scaler = None
        self.clf_scaler = None
        self.feature_bounds = None
        
    def load_detector(self, model_path: str = None, scaler_path: str = None):
        try:
            model_path = str(WEIGHTS_DIR / "detector.pth") if model_path is None else model_path
            scaler_path = str(WEIGHTS_DIR / "detector_scaler.pkl") if scaler_path is None else scaler_path
            
            checkpoint = torch.load(model_path, map_location='cpu')
            self.detector_model = AttentionLSTM(input_dim=5, n_classes=2, hidden_dim=64, num_layers=2)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.detector_model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.detector_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.detector_model.load_state_dict(checkpoint)
            
            self.detector_model.eval()
            self.detector_scaler = joblib.load(scaler_path)
            return True, "Детектор загружен успешно"
        except Exception as e:
            return False, f"Ошибка загрузки детектора: {str(e)}"
    
    def load_segmenter(self, model_path: str = None, scaler_path: str = None):
        try:
            model_path = str(WEIGHTS_DIR / "segmenter.pth") if model_path is None else model_path
            scaler_path = str(WEIGHTS_DIR / "segmenter_scaler.pkl") if scaler_path is None else scaler_path
            
            checkpoint = torch.load(model_path, map_location='cpu')
            self.segmenter_model = AttentionLSTM(input_dim=5, n_classes=2, hidden_dim=64, num_layers=2)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.segmenter_model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.segmenter_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.segmenter_model.load_state_dict(checkpoint)
            
            self.segmenter_model.eval()
            self.segmenter_scaler = joblib.load(scaler_path)
            return True, "Сегментатор загружен успешно"
        except Exception as e:
            return False, f"Ошибка загрузки сегментатора: {str(e)}"
    
    def load_classifier(self, model_path: str = None, scaler_path: str = None):
        try:
            model_path = str(WEIGHTS_DIR / "classifier.pkl") if model_path is None else model_path
            scaler_path = str(WEIGHTS_DIR / "classifier_scaler.pkl") if scaler_path is None else scaler_path
            
            self.clf_model = joblib.load(model_path)
            self.clf_scaler = joblib.load(scaler_path)
            return True, "Классификатор загружен успешно"
        except Exception as e:
            return False, f"Ошибка загрузки классификатора: {str(e)}"
    
    def load_feature_bounds(self, bounds_path: str = None):
        try:
            bounds_path = str(WEIGHTS_DIR / "feature_bounds.json") if bounds_path is None else bounds_path
            import json
            with open(bounds_path, 'r', encoding='utf-8') as f:
                self.feature_bounds = json.load(f)
            return True, "Границы нормы загружены"
        except Exception as e:
            return False, f"Ошибка загрузки границ нормы: {str(e)}"
    
    def is_ready(self) -> bool:
        return all([
            self.detector_model is not None, self.segmenter_model is not None,
            self.clf_model is not None, self.detector_scaler is not None,
            self.segmenter_scaler is not None, self.clf_scaler is not None,
            self.feature_bounds is not None
        ])

model_manager = ModelManager()

def load_all_models():
    print("Загрузка моделей из папки:", WEIGHTS_DIR)
    print("Детектор:", model_manager.load_detector()[1])
    print("Сегментатор:", model_manager.load_segmenter()[1])
    print("Классификатор:", model_manager.load_classifier()[1])
    print("Границы нормы:", model_manager.load_feature_bounds()[1])
    print("✅ Все модели успешно загружены!" if model_manager.is_ready() else "⚠️ Ошибка загрузки!")

load_all_models()