import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Cấu hình Data giống tập Val
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(os.path.join('dataset', 'val'), val_transforms)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
class_names = val_dataset.classes

# 2. Khởi tạo lại cấu trúc mô hình và nạp trọng số đã lưu
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('best_seed_model.pth', weights_only=True))
model = model.to(device)
model.eval()

# 3. Quét qua tập dữ liệu đánh giá
all_preds = []
all_labels = []

print("Đang chạy đánh giá trên tập Val...")
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 4. Vẽ Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Dự đoán của Model')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn - Phân loại hạt giống')
plt.show()