import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
data_dir = "dataset"
model_path = "best_seed_model.pth"
output_dir = "evaluation_outputs"


def load_validation_data():
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_dataset, val_loader


def load_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model


def collect_predictions(model, dataloader):
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds


def save_confusion_matrix(cm, class_names):
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def save_per_class_accuracy(cm, class_names):
    per_class_accuracy = []
    for idx, class_name in enumerate(class_names):
        total = cm[idx].sum()
        correct = cm[idx, idx]
        score = float(correct / total) if total else 0.0
        per_class_accuracy.append((class_name, score))

    labels = [item[0] for item in per_class_accuracy]
    scores = [item[1] * 100 for item in per_class_accuracy]

    plt.figure(figsize=(14, 8))
    plt.bar(labels, scores, color="#2b6cb0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-class Accuracy on Validation Set")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_accuracy.png"), dpi=300)
    plt.close()


def main():
    os.makedirs(output_dir, exist_ok=True)

    val_dataset, val_loader = load_validation_data()
    class_names = val_dataset.classes
    model = load_model(len(class_names))

    print("Dang danh gia model tren tap validation...")
    all_labels, all_preds = collect_predictions(model, val_loader)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    report_text = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    save_confusion_matrix(cm, class_names)
    save_per_class_accuracy(cm, class_names)

    metrics_summary = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }

    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(os.path.join(output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n=== TONG HOP CHI SO ===")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision weighted : {precision:.4f}")
    print(f"Recall weighted    : {recall:.4f}")
    print(f"F1-score weighted  : {f1:.4f}")
    print("\n=== CLASSIFICATION REPORT ===")
    print(report_text)
    print(f"Da luu confusion matrix tai: {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"Da luu bieu do theo lop tai: {os.path.join(output_dir, 'per_class_accuracy.png')}")
    print(f"Da luu bao cao text tai    : {os.path.join(output_dir, 'classification_report.txt')}")
    print(f"Da luu bao cao json tai    : {os.path.join(output_dir, 'classification_report.json')}")
    print(f"Da luu tong hop chi so tai : {os.path.join(output_dir, 'metrics_summary.json')}")


if __name__ == "__main__":
    main()
