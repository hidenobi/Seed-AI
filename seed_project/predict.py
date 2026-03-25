import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import sys

def predict_single_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tự động lấy số lượng và tên class
    dataset_path = os.path.join('dataset', 'train')
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    num_classes = len(class_names)
    
    # Khởi tạo mô hình
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('best_seed_model.pth', map_location=device, weights_only=True))
    model.eval().to(device)
    
    # Tiền xử lý
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Lỗi không mở được ảnh: {e}")
        return

    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    label = class_names[predicted_idx.item()]
    score = confidence.item() * 100
    
    print("="*40)
    print(f"ẢNH: {image_path}")
    print(f"KẾT QUẢ DỰ ĐOÁN: {label}")
    print(f"ĐỘ TIN CẬY (CONFIDENCE): {score:.2f}%")
    print("="*40)

if __name__ == '__main__':
    # Cách chạy: python predict.py ten_anh.jpg
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predict_single_image(img_path)
    else:
        print("Vui lòng cung cấp đường dẫn ảnh. Ví dụ: python predict.py test_seed.jpg")