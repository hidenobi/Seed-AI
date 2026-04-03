# Tài liệu dự án và kiểm tra hiện trạng

## 1. Mục tiêu dự án

Dự án này xây dựng một hệ thống phân loại ảnh cho 20 lớp hạt/quả bằng học sâu, dựa trên PyTorch và torchvision. Luồng xử lý chính gồm:

1. Chuẩn bị dữ liệu từ thư mục ảnh gốc.
2. Gắn nhãn tự động theo tên thư mục.
3. Huấn luyện mô hình baseline CNN và mô hình chính ResNet18 fine-tune từ pretrain.
4. Dự đoán ảnh mới.
5. Đánh giá mô hình bằng accuracy, precision, recall, F1-score, confusion matrix.
6. Giải thích vùng ảnh quan trọng bằng Grad-CAM.

Các file chính:

- `prepare_data.py`: tách dữ liệu train/val.
- `train_baseline.py`: huấn luyện mô hình CNN đơn giản làm baseline.
- `train.py`: huấn luyện mô hình chính ResNet18 fine-tune.
- `predict.py`: suy luận ảnh đơn.
- `demo_predict_console.py`: giao diện console để demo dự đoán.
- `analyze_model.py`: đánh giá chi tiết và lưu báo cáo.
- `evaluate.py`: đánh giá cơ bản bằng confusion matrix.
- `explain.py`: sinh heatmap Grad-CAM.

## 2. Cơ sở lý thuyết

### 2.1. Bài toán học có giám sát

Đây là bài toán phân loại ảnh nhiều lớp. Mỗi ảnh đầu vào có đúng 1 nhãn thật. Mô hình học ánh xạ:

`ảnh RGB -> vector đặc trưng -> logits -> xác suất theo lớp`

Trong huấn luyện, mô hình tối ưu sao cho lớp dự đoán gần với nhãn thật nhất.

### 2.2. CNN và embedding ảnh

CNN học đặc trưng ảnh theo tầng:

1. Tầng đầu học cạnh, góc, texture.
2. Tầng giữa học mẫu hình cục bộ như hình dạng, hoa văn, bề mặt.
3. Tầng sâu học đặc trưng ngữ nghĩa hơn, ví dụ hình dáng tổng thể của từng loại quả/hạt.

“Embedding ảnh” trong dự án này không được lưu riêng thành file vector, nhưng về mặt mô hình nó chính là biểu diễn đặc trưng sâu ở phần cuối backbone trước lớp phân loại. Với ResNet18, ảnh sau khi đi qua các block tích chập và residual sẽ được gom thông tin bằng average pooling, tạo ra vector đặc trưng 512 chiều trước lớp `fc`.

Vector này là phần “nén thông tin” quan trọng nhất của ảnh, dùng làm đầu vào cho lớp phân loại cuối cùng.

### 2.3. Transfer learning và pre-train

Trong `train.py`, mô hình chính dùng:

`models.resnet18(weights=models.ResNet18_Weights.DEFAULT)`

Điều này có nghĩa là backbone ResNet18 được khởi tạo bằng trọng số pretrain trên ImageNet. Ý tưởng:

- Mạng đã học sẵn đặc trưng thị giác phổ quát từ tập dữ liệu lớn.
- Ta không cần huấn luyện từ đầu toàn bộ mạng.
- Ta chỉ fine-tune một phần mạng để thích nghi với bộ dữ liệu 20 lớp hiện tại.

Đây là cơ chế giúp mô hình hội tụ nhanh hơn, tận dụng tri thức có sẵn và thường tốt hơn baseline CNN nhỏ.

### 2.4. Fine-tuning

Theo `train.py`, chỉ `layer4` và `fc` được mở khóa để cập nhật gradient, còn các tầng còn lại bị freeze.

Ý nghĩa:

- Các tầng đầu giữ lại tri thức phổ quát đã học từ ImageNet.
- Các tầng cuối được điều chỉnh để phân biệt 20 lớp cụ thể trong bài toán.
- Đây là chiến lược cân bằng giữa hiệu quả học và tránh overfitting khi dữ liệu không quá lớn.

### 2.5. Hàm mất mát và dự đoán

Dự án dùng `CrossEntropyLoss`, phù hợp cho phân loại nhiều lớp một nhãn.

Trong suy luận:

1. Mô hình xuất ra `logits`.
2. `softmax` biến logits thành xác suất.
3. `argmax` chọn lớp có xác suất lớn nhất.

## 3. Cơ chế dữ liệu và gắn nhãn

### 3.1. Tách train/validation

Trong [`prepare_data.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/prepare_data.py#L5), hàm `split_dataset()`:

- đọc các thư mục con trong `raw_images/`,
- xem mỗi thư mục như một lớp,
- xáo trộn danh sách ảnh bằng `random.shuffle`,
- tách theo tỷ lệ `0.8 / 0.2`,
- copy sang `dataset/train/<class>` và `dataset/val/<class>`.

### 3.2. Gắn nhãn

Trong [`train.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/train.py#L29), [`train_baseline.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/train_baseline.py#L36) và [`analyze_model.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/analyze_model.py#L35), dự án dùng `torchvision.datasets.ImageFolder`.

Cơ chế của `ImageFolder`:

- tên thư mục chính là nhãn lớp,
- thứ tự lớp được sắp xếp theo alphabet,
- mỗi ảnh trong thư mục đó sẽ tự động mang nhãn số tương ứng.

Với dữ liệu hiện tại, kiểm tra thực tế cho thấy:

- số lớp: 20,
- train: 1440 ảnh,
- validation: 360 ảnh,
- mỗi lớp có 72 ảnh train và 18 ảnh val.

Đây là tập validation cân bằng hoàn toàn, nên accuracy tổng thể phản ánh khá tốt trung bình giữa các lớp.

## 4. Thuật toán và mô hình đang dùng

### 4.1. Baseline CNN

Trong [`train_baseline.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/train_baseline.py#L11), mô hình baseline gồm:

- 2 lớp tích chập,
- ReLU,
- MaxPool,
- flatten,
- 2 fully connected layers.

Đây là mô hình train từ đầu, không pretrain. Mục đích chính là làm mốc so sánh.

Ưu điểm:

- dễ hiểu,
- nhẹ,
- phù hợp minh họa cơ bản.

Hạn chế:

- ít tầng, khả năng trích xuất đặc trưng hạn chế,
- không tận dụng tri thức pretrain,
- không có đánh giá validation đi kèm trong file baseline.

### 4.2. Mô hình chính ResNet18 fine-tune

Trong [`train.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/train.py#L35), mô hình chính là ResNet18.

Pipeline huấn luyện:

1. Resize ảnh về `224x224`.
2. Augmentation ở train:
   - random horizontal flip,
   - random rotation 10 độ.
3. Chuẩn hóa ảnh theo mean/std của ImageNet.
4. Nạp ResNet18 pretrained.
5. Freeze các tầng ngoài `layer4` và `fc`.
6. Thay lớp `fc` cuối để số đầu ra bằng số lớp hiện có.
7. Tối ưu bằng Adam với learning rate `0.001`.
8. Theo dõi loss/accuracy train và val.
9. Lưu `best_seed_model.pth` theo val accuracy tốt nhất.

### 4.3. Cơ chế embedding ảnh trong ResNet18

Từ góc nhìn khái niệm, đường đi của embedding là:

`input image -> conv1 -> residual blocks layer1..layer4 -> global average pooling -> vector 512 chiều -> fc -> logits`

Ở đây:

- residual blocks giúp lan truyền gradient tốt hơn mạng CNN thường,
- các skip connection giữ thông tin ổn định khi mạng sâu hơn,
- vector 512 chiều là biểu diễn cô đọng của ảnh,
- lớp `fc` biến embedding thành điểm số cho 20 lớp.

### 4.4. Cơ chế giải thích bằng Grad-CAM

Trong [`explain.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/explain.py#L24), Grad-CAM lấy `model.layer4[-1]` làm tầng mục tiêu.

Nguyên lý:

1. Tính gradient của lớp dự đoán theo feature maps cuối.
2. Dùng gradient làm trọng số cho các kênh đặc trưng.
3. Nội suy thành heatmap trên ảnh gốc.

Kết quả được lưu tại `gradcam_result.jpg`, giúp biết mô hình tập trung vào vùng nào khi ra quyết định.

## 5. Dự đoán ảnh mới hoạt động thế nào

File hoạt động đúng với mô hình hiện tại là [`predict.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/predict.py#L19).

Luồng dự đoán:

1. Đọc danh sách lớp từ `dataset/train`.
2. Dựng lại ResNet18 với số lớp đúng theo dataset.
3. Nạp checkpoint `best_seed_model.pth`.
4. Resize và normalize ảnh đầu vào.
5. Chạy forward pass.
6. Dùng `softmax` lấy xác suất.
7. Chọn lớp có xác suất cao nhất.
8. Trả về nhãn và confidence.

`demo_predict_console.py` chỉ là giao diện console gọi lại `predict_single_image()`.

## 6. Kiểm tra output đánh giá thuật toán

### 6.1. Output hiện có trong dự án

Thư mục `evaluation_outputs/` hiện chứa:

- `metrics_summary.json`
- `classification_report.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `per_class_accuracy.png`

Từ các file output và phép kiểm tra lại trực tiếp bằng script đọc model, các chỉ số khớp nhau:

- Accuracy: `0.8944`
- Precision weighted: `0.9078`
- Recall weighted: `0.8944`
- F1 weighted: `0.8950`

Vì tập validation cân bằng 20 lớp x 18 ảnh, các chỉ số weighted và macro gần nhau là hợp lý.

### 6.2. Lớp mạnh và lớp yếu

Các lớp có kết quả rất tốt:

- `Palm`: recall 1.00, đúng 18/18.
- `Pumpkin`: recall 1.00, đúng 18/18.
- `Amla`, `Black_Plum`, `Mango`: F1 khoảng 0.97.

Các lớp còn khó:

- `Orange`: recall 0.6111, đúng 11/18.
- `Avocado`: F1 0.7568.
- `Olive`: precision 0.68, nghĩa là model khá hay đoán nhầm ảnh lớp khác thành `Olive`.

### 6.3. Các nhầm lẫn nổi bật

Kiểm tra lại confusion thực tế cho thấy:

- `Orange` hay bị nhầm sang `Pumpkin` (4 ảnh) và `Olive` (3 ảnh).
- `Avocado` hay bị nhầm sang `Litchi` (2 ảnh).
- `Date` bị phân tán sang `Avocado`, `Chinese_Date`, `Olive`.
- `Guava` hay bị nhầm sang `Olive` (2 ảnh).
- `Tamarind` có 2 ảnh nhầm sang `Sapodilla`.

Nhận xét:

- Đây là kiểu nhầm lẫn khá hợp lý về mặt thị giác nếu các lớp có màu hoặc hình dạng gần nhau.
- Accuracy gần 90% là tốt với bài toán 20 lớp, nhưng còn dư địa cải thiện ở các lớp có bề ngoài tương tự.

### 6.4. Kiểm tra suy luận mẫu

Chạy `predict.py` với `seed_test.jpg` cho kết quả:

- nhãn dự đoán: `Cashew`
- confidence: `100.00%`

Con số này cho thấy model rất chắc với ảnh thử này, nhưng không nên xem đây là bằng chứng chất lượng tổng quát; chất lượng tổng quát phải đọc từ validation report.

## 7. Kiểm tra lại code hiện trạng

### 7.1. Điểm đúng và nhất quán

- Pipeline train chính trong [`train.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/train.py#L14) phù hợp với transfer learning tiêu chuẩn.
- `analyze_model.py` là script đánh giá đáng tin cậy nhất hiện tại vì:
  - nạp đúng số lớp từ `ImageFolder`,
  - tính đủ accuracy/precision/recall/F1,
  - lưu cả text, json và hình ảnh.
- `predict.py` dựng số lớp động từ dataset nên tương thích với checkpoint hiện tại.

### 7.2. Vấn đề phát hiện được

1. [`inference.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/inference.py#L8) đang hard-code 3 lớp `Canadian/Kama/Rosa`, trong khi checkpoint hiện tại là model 20 lớp.
   Hệ quả: chạy file này bị lỗi `size mismatch for fc.weight/fc.bias` khi `load_state_dict`.

2. [`evaluate.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/evaluate.py#L39) chỉ vẽ `plt.show()`.
   Hệ quả: script này phù hợp khi chạy có giao diện, nhưng không phải công cụ lưu báo cáo chuẩn. Trong thực tế, `analyze_model.py` mới là nguồn output đánh giá đầy đủ hơn.

3. [`prepare_data.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/prepare_data.py#L33) không cố định random seed.
   Hệ quả: mỗi lần tách lại dữ liệu có thể cho train/val split khác nhau, nên kết quả khó tái lập tuyệt đối.

4. [`prepare_data.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/prepare_data.py#L41) copy ảnh sang thư mục đích nhưng không làm sạch dữ liệu cũ trước.
   Hệ quả: nếu thay đổi source và chạy lại nhiều lần, cần cẩn thận tránh trạng thái dữ liệu cũ còn sót.

5. [`train_baseline.py`](/Volumes/Extreme_SSD/PyCharmData/Projects/tgmt/seed_project/train_baseline.py#L44) chỉ in loss train, chưa có validation metrics.
   Hệ quả: baseline hiện chỉ mang tính tham khảo mô hình, chưa đủ dữ liệu để so sánh công bằng với ResNet18.

## 8. Kết luận

Về mặt thuật toán, dự án đang vận hành theo hướng đúng:

- gắn nhãn bằng cấu trúc thư mục,
- tiền xử lý chuẩn theo ImageNet,
- dùng transfer learning với ResNet18 pretrained,
- fine-tune tầng cuối để thích nghi với dữ liệu chuyên biệt,
- đánh giá bằng bộ chỉ số chuẩn cho phân loại nhiều lớp,
- có thêm Grad-CAM để giải thích dự đoán.

Về output đánh giá, các file trong `evaluation_outputs/` là nhất quán với kết quả kiểm tra lại:

- mô hình đạt khoảng `89.44%` accuracy trên validation,
- có vài lớp đạt rất cao,
- điểm yếu tập trung ở một số cặp lớp gần nhau về hình thái hoặc màu sắc,
- `predict.py` phù hợp với checkpoint hiện tại,
- `inference.py` không còn phù hợp với model đang lưu.

Nếu cần bước tiếp theo, nên ưu tiên:

1. thống nhất chỉ giữ một đường suy luận chính,
2. cố định seed để tái lập,
3. tăng augmentation hoặc số epoch cho các lớp đang nhầm nhiều,
4. lưu lịch sử train dưới dạng số liệu thô ngoài ảnh biểu đồ để tiện phân tích sâu hơn.
