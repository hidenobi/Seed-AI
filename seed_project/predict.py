import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import sys

_PREDICT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_PREDICT_MODEL = None
_CLASS_NAMES = None
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_prediction_assets():
    global _PREDICT_MODEL, _CLASS_NAMES

    if _PREDICT_MODEL is not None and _CLASS_NAMES is not None:
        return _PREDICT_DEVICE, _PREDICT_MODEL, _CLASS_NAMES

    dataset_path = os.path.join('dataset', 'train')
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load('best_seed_model.pth', map_location=_PREDICT_DEVICE, weights_only=True))
    model.eval().to(_PREDICT_DEVICE)

    _PREDICT_MODEL = model
    _CLASS_NAMES = class_names
    return _PREDICT_DEVICE, _PREDICT_MODEL, _CLASS_NAMES


def predict_single_image(image_path, verbose=True):
    device, model, class_names = load_prediction_assets()
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Lỗi không mở được ảnh: {e}")
        return None

    image_tensor = _TRANSFORM(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    label = class_names[predicted_idx.item()]
    score = confidence.item() * 100

    result = {
        'image_path': image_path,
        'label': label,
        'confidence': score,
    }

    if verbose:
        print("=" * 40)
        print(f"ẢNH: {image_path}")
        print(f"KẾT QUẢ DỰ ĐOÁN: {label}")
        print(f"ĐỘ TIN CẬY (CONFIDENCE): {score:.2f}%")
        print("=" * 40)

    return result

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predict_single_image(img_path)
    else:
        print("Vui lòng cung cấp đường dẫn ảnh. Ví dụ: python predict.py test_seed.jpg")
