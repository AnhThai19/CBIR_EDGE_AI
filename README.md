# 📱 CBIR_EDGE_AI
Hệ Thống Tìm Kiếm Hình Ảnh Bằng Deep Learning (CBIR) Triển Khai Trên Thiết Bị Di Động

Dự án xây dựng hệ thống tìm kiếm hình ảnh dựa trên nội dung (Content-Based Image Retrieval - CBIR) cho các thiết bị công nghệ.

Thay vì tìm kiếm bằng văn bản, hệ thống cho phép người dùng tìm các sản phẩm tương tự bằng chính hình ảnh.

Mô hình Deep Learning được huấn luyện để trích xuất vector đặc trưng (embedding) từ ảnh, sau đó sử dụng **Cosine Similarity** để tìm các ảnh có nội dung tương tự.

Điểm nổi bật của dự án là triển khai AI trực tiếp trên thiết bị di động (**Edge AI**) bằng TensorFlow Lite, giúp hệ thống hoạt động offline và không cần server.

---

## 🚀 Tính Năng Chính
- 🔍 Tìm kiếm hình ảnh dựa trên độ tương đồng nội dung  
- 🧠 Trích xuất đặc trưng bằng Deep Learning (MobileNetV3)  
- 📱 Chạy AI trực tiếp trên thiết bị di động  
- ⚡ Tìm kiếm nhanh bằng Cosine Similarity  
- 🌐 Hoạt động offline không cần Internet  
- 📦 Lưu trữ embedding dataset dưới dạng `vectors.json`  

---

## 🧠 Kiến Trúc Hệ Thống
```
Ảnh từ người dùng (Camera / Gallery)
│
▼
Tiền xử lý ảnh
Resize → CenterCrop → Normalize
│
▼
Mô hình MobileNetV3 (.tflite)
│
▼
Vector đặc trưng (Embedding 256 chiều)
│
▼
So khớp Cosine Similarity
│
▼
Top-K ảnh tương tự
```

---

## 🛠 Công Nghệ Sử Dụng
- **Machine Learning**: PyTorch, MobileNetV3-Small, Triplet Loss  
- **Chuyển đổi mô hình**: ONNX → TensorFlow → TensorFlow Lite  
- **Mobile Application**: React Native (Expo), `react-native-fast-tflite`  
- **Xử lý dữ liệu**: Python, NumPy, PIL  

---

## 📊 Dataset
Dataset gồm các sản phẩm công nghệ như:
- Smartphone  
- Laptop  
- Keyboard  
- Speaker  
- Monitor  
- Server Rack  
- Smartwatch  
- Các thiết bị điện tử khác  

**Thống kê dataset**  
- Số lớp: ~25  
- Số lượng ảnh: ~4000  

**Cấu trúc dataset : **
```
dataset/
- train/
- val/
- test/
```

---

## 🧪 Huấn Luyện Mô Hình
- **Backbone**: MobileNetV3-Small (pretrained ImageNet)  
- **Loss**: Triplet Margin Loss  
- **Optimizer**: AdamW  
- **Input size**: 224×224  
- **Embedding size**: 256  

**Data Augmentation**  
- Resize(256)  
- CenterCrop(224)  
- RandomHorizontalFlip  
- ColorJitter  
- RandomRotation  
- Normalize(ImageNet)  

---

## 🔄 Pipeline Chuyển Đổi Mô Hình
```
PyTorch (.pth)
│
▼
ONNX (.onnx)
│
▼
TensorFlow
│
▼
TensorFlow Lite (.tflite)
```

---

## 📱 Ứng Dụng Di Động
Ứng dụng mobile cho phép:
- Chụp ảnh bằng camera  
- Chọn ảnh từ thư viện  
- Chạy inference trực tiếp trên thiết bị  
- Hiển thị các ảnh tương tự nhất  

**Pipeline trên Mobile**
```
Ảnh đầu vào
│
Resize → CenterCrop → Normalize
│
TFLite Inference
│
Embedding Vector
│
Cosine Similarity
│
Top-K ảnh tương tự
```

Toàn bộ quá trình xử lý được thực hiện trực tiếp trên thiết bị, giúp:
- Tốc độ xử lý nhanh
- Bảo mật dữ liệu tốt hơn
- Hoạt động hoàn toàn offline


📁 Cấu Trúc Project
```
CBIR_ON_EDGE_DEVICE
│
├── notebooks
│   ├── mobilenetv3_small.ipynb     # huấn luyện mô hình
│   ├── convert_file.ipynb          # chuyển đổi model
│   └── preprocessing.ipynb         # trích xuất embedding dataset
│
├── model
│   ├── best_model.pth
│   ├── model.onnx
│   └── model.tflite
│
├── mobile_app
│   ├── App.tsx
│   ├── components/
│   └── assets/
│
├── dataset
│   ├── train
│   ├── val
│   └── test
│
├── vectors.json                    # database embedding
│
└── README.md
```
## 📷 Ảnh Minh Họa !
[Kiến trúc hệ thống](mobile_app/assets/picture1.jpg)
