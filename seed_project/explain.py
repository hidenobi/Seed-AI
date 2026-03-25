import os
import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = os.path.join('dataset', 'train')
class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
num_classes = len(class_names)
print(f"Hệ thống nhận diện được {num_classes} loại hạt từ tập dữ liệu.")

# Khởi tạo mô hình
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('best_seed_model.pth', map_location=device, weights_only=True))
model.eval().to(device)

# 2. Chọn layer cuối cùng của mạng CNN để nội suy Grad-CAM
target_layers = [model.layer4[-1]] 

# 3. Tiền xử lý ảnh gốc
image_path = "seed_test.jpg" # ĐƯỜNG DẪN ẢNH TEST CỦA BẠN
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] # Đọc bằng OpenCV và chuyển BGR sang RGB
rgb_img = cv2.resize(rgb_img, (224, 224))
rgb_img_float = np.float32(rgb_img) / 255 # Chuẩn hóa về [0, 1] để vẽ Heatmap

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = transform(rgb_img).unsqueeze(0).to(device)

# 4. Chạy Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

# Đè bản đồ nhiệt (Heatmap) lên ảnh gốc
cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

# 5. Lưu kết quả
cv2.imwrite("gradcam_result.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))