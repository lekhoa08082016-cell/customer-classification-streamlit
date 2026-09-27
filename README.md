# Phân loại Khách hàng (Customer Classification)

Đây là mã nguồn và tài liệu cho **Đề bài tập (Python – Classification)** - **Đề 1: Phân loại khách hàng (Customer Segmentation / Churn)**. 
Mục tiêu của dự án là áp dụng các thuật toán classification phổ biến trên tập dữ liệu thực tế nhằm dự đoán hành vi khách hàng, từ đó rút ra các Insight kinh doanh (E-commerce / Marketing).

## 📊 Dataset: Bank Marketing (UCI)

Tập dữ liệu được sử dụng là **Bank Marketing** từ [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing), với tính chất tương tự như bài toán dự đoán hành vi mua hàng trong E-commerce (khách hàng có đồng ý đăng ký dịch vụ/mua hàng hay không).

- **Số lượng records:** 41,188
- **Số lượng features:** 20 (10 numerical + 10 categorical)
- **Target:** `y` (yes/no — khách hàng có đăng ký sử dụng dịch vụ/mua hàng không).

---

## 🎯 Hoàn thành các Yêu cầu của Đề bài

Dự án này đã thực hiện đầy đủ 6 yêu cầu từ đề bài:

### 1️⃣ Yêu cầu 1: Load dữ liệu, làm sạch
- Đọc dữ liệu từ file `.csv` và loại bỏ các dữ liệu trùng lặp (duplicates).
- **Xử lý Missing values:** Thay thế các giá trị `"unknown"` bằng giá trị phổ biến nhất (Mode) thông qua `SimpleImputer`.
- **Encoding & Scaling:** Sử dụng `OneHotEncoder` cho biến phân loại (categorical) và `StandardScaler` cho biến liên tục (numerical).
- Tất cả được đóng gói gọn gàng bằng `ColumnTransformer` để tránh data leakage.

### 2️⃣ Yêu cầu 2: Tạo target
- Chuyển đổi cột mục tiêu `y` (yes/no) thành định dạng nhị phân `1` (có mua hàng/đăng ký) và `0` (không mua hàng/từ chối) để đưa vào mô hình học máy.

### 3️⃣ Yêu cầu 3: Chia tập Train/Test & Cross Validation
- Sử dụng `train_test_split` để chia tập dữ liệu theo tỷ lệ **70/30**.
- Đảm bảo tỷ lệ phân phối nhãn (Class distribution) đồng đều bằng tham số `stratify=y`.
- Áp dụng **StratifiedKFold (K=5)** trong quá trình Cross-validation để đánh giá mô hình khách quan nhất.

### 4️⃣ Yêu cầu 4: Huấn luyện ít nhất 4 thuật toán
Dự án đã sử dụng `Pipeline` để huấn luyện 4 thuật toán classification phổ biến:
1. **Logistic Regression** (baseline có trọng số cân bằng)
2. **Decision Tree**
3. **Random Forest** (mô hình ensemble với 300 cây)
4. **Gradient Boosting** (boosting tree method)

### 5️⃣ Yêu cầu 5: Đánh giá mô hình
Tất cả các mô hình được đánh giá toàn diện qua các chỉ số:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**.
- Trực quan hóa bằng **Confusion Matrix** và đường cong **ROC Curve**.

### 6️⃣ Yêu cầu 6: Phân tích Feature Importance & Insight E-commerce
- Trích xuất mức độ quan trọng của các đặc trưng (Feature Importance) từ mô hình mạnh nhất (Random Forest / Gradient Boosting).
- Viết báo cáo giải thích các yếu tố cốt lõi ảnh hưởng đến quyết định mua hàng (ví dụ: thời lượng tư vấn, tuổi tác, biến động kinh tế vĩ mô...).
- *Chi tiết được trình bày trong file báo cáo (`report/report.md`).*

---

## 📁 Cấu trúc thư mục (Deliverables)

```text
customer_classification/
│
├── data/                          # Chứa file dữ liệu dataset
│   └── bank-additional-full.csv   
│
├── notebooks/                     # Chứa Notebook theo yêu cầu
│   └── customer_classification.ipynb  # File Notebook thực hiện toàn bộ YC 1->6
│
├── report/                        # Báo cáo kết quả
│   └── report.md                  # Báo cáo ngắn (1-2 trang) giải thích Insight E-commerce
│
├── outputs/                       # Thư mục chứa kết quả xuất ra
│   ├── figures/                   # Biểu đồ đánh giá, Confusion Matrix, Feature Importance
│   ├── models/                    # File mô hình đã được huấn luyện (.pkl)
│   └── results/                   # Bảng kết quả metrics (.csv)
│
├── src/                           # Source code (nếu cần xem chi tiết cách cấu trúc)
│   ├── data_preprocessing.py      
│   ├── models.py                  
│   └── evaluation.py              
│
├── app.py                         # Ứng dụng Web Streamlit để test model trực quan
├── save_models.py                 # File python để lưu model
├── requirements.txt               # Thư viện cần thiết
└── README.md                      # File hướng dẫn (là file này)
```

---

## 🚀 Hướng dẫn chạy dự án

### 1. Cài đặt thư viện
Hãy chắc chắn rằng máy bạn đã cài đặt Python 3.9 trở lên. Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

### 2. Chạy Notebook (Đề bài yêu cầu)
Bạn có thể mở và chạy toàn bộ mã nguồn trong file `notebooks/customer_classification.ipynb` bằng VS Code hoặc Jupyter Notebook:
```bash
cd notebooks
jupyter notebook customer_classification.ipynb
```

### 3. Xem Demo Giao diện Web (Mở rộng thêm)
Để thấy mô hình hoạt động trên giao diện thực tế (cực kỳ trực quan cho Marketing), hãy chạy Streamlit:
```bash
streamlit run app.py
```
Sau đó truy cập đường link trình duyệt hiện ra (thường là `http://localhost:8501`).

---

## 📝 License & References
- Bộ dữ liệu được cấp bởi **UCI Machine Learning Repository**.
- Phục vụ hoàn toàn cho mục đích học tập và nộp bài môn Học Máy.
