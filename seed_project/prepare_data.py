import os
import shutil
import random

def split_dataset(raw_dir='raw_images', base_dir='dataset', split_ratio=0.8):
    # Tạo thư mục đích (dataset/train và dataset/val)
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    for folder in [train_dir, val_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Kiểm tra xem thư mục ảnh gốc có tồn tại không
    if not os.path.exists(raw_dir):
        print(f"Lỗi: Không tìm thấy thư mục '{raw_dir}'. Hãy đảm bảo bạn đã giải nén dữ liệu Kaggle vào đây.")
        return

    # Lấy danh sách các loại hạt (tên các thư mục con)
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    print(f"Tìm thấy {len(classes)} loại hạt: {classes}")
    
    for cls in classes:
        # Tạo thư mục cho từng class trong train và val
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        
        # Lấy toàn bộ danh sách file ảnh của class đó
        cls_dir = os.path.join(raw_dir, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Xáo trộn ngẫu nhiên ảnh để đảm bảo tính khách quan
        random.shuffle(images)
        
        # Cắt mảng theo tỉ lệ (80% train, 20% val)
        train_size = int(len(images) * split_ratio)
        train_imgs = images[:train_size]
        val_imgs = images[train_size:]
        
        # Tiến hành copy file
        for img in train_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(train_dir, cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(cls_dir, img), os.path.join(val_dir, cls, img))
            
        print(f" - {cls}: Đã copy {len(train_imgs)} ảnh vào Train, {len(val_imgs)} ảnh vào Val.")
        
    print(f"\nHoàn tất! Cấu trúc dữ liệu đã sẵn sàng tại thư mục '{base_dir}'.")

# Khởi chạy hàm
if __name__ == '__main__':
    split_dataset()