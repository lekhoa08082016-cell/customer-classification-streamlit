# Báo Cáo Phân Loại Khách Hàng – Bank Marketing Dataset

**Môn học:** Machine Learning / Data Science  
**Đề tài:** Đề 1 – Phân loại khách hàng (Customer Segmentation / Churn)  
**Phương pháp:** Supervised Binary Classification

---

## 1. Giới Thiệu Bài Toán

Bài toán đặt ra là xây dựng mô hình Machine Learning để dự đoán khách hàng có phản hồi tích cực với chiến dịch marketing trực tiếp của ngân hàng hay không (đăng ký sản phẩm tiết kiệm có kỳ hạn). Đây là bài toán **phân loại nhị phân có giám sát** (supervised binary classification) với biến mục tiêu `y` (yes/no).

**Ứng dụng trong E-commerce:** Doanh nghiệp thương mại điện tử có thể áp dụng phương pháp tương tự để xác định nhóm khách hàng có xác suất chuyển đổi cao, từ đó ưu tiên phân bổ ngân sách quảng cáo, voucher và chiến dịch remarketing.

## 2. Dataset và Tiền Xử Lý

**Nguồn dữ liệu:** UCI Machine Learning Repository – Bank Marketing Dataset  
**Quy mô:** 41,188 bản ghi × 21 cột (20 features + 1 target)

### Đặc điểm dữ liệu:
- **10 biến số (numerical):** `age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
- **10 biến phân loại (categorical):** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`
- **Target:** `y` → mã hóa: yes=1, no=0

### Tiền xử lý:
- Loại bỏ bản ghi trùng lặp
- Xử lý giá trị `"unknown"`:
  - Cột có tỷ lệ unknown < 20%: thay bằng NaN → impute bằng most_frequent
  - Cột có tỷ lệ unknown ≥ 20%: giữ "unknown" như một category hợp lệ
- Feature engineering: tạo `has_previous_contact`, `campaign_intensity`, `age_group`
- Sử dụng `Pipeline` + `ColumnTransformer` để đảm bảo không có data leakage

### Mất cân bằng lớp:
- **No:** ~88.7% | **Yes:** ~11.3%
- Sử dụng `class_weight="balanced"` và ưu tiên F1/ROC-AUC thay vì Accuracy

## 3. Phương Pháp Machine Learning

### Các mô hình sử dụng:

| # | Model | Đặc điểm |
|---|-------|----------|
| 1 | Logistic Regression | Mô hình tuyến tính, class_weight="balanced" |
| 2 | Decision Tree | Cây quyết định, dễ diễn giải |
| 3 | Random Forest | Ensemble 300 cây, class_weight="balanced" |
| 4 | Gradient Boosting | Boosting tuần tự, mặc định |

### Quy trình đánh giá:
1. **Train/Test Split:** 70/30, stratified
2. **Cross-Validation:** Stratified 5-Fold trên training set
3. **Final Evaluation:** Fit trên toàn bộ training set → predict trên test set
4. **Kiểm tra Overfitting:** So sánh train vs test performance

## 4. Kết Quả

### Cross-Validation (Training Data):

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC |
|-------|------------|-------------|----------|-------|-----------|
| Logistic Regression | 0.8604 | 0.4404 | 0.8827 | 0.5876 | 0.9371 |
| Decision Tree | 0.8911 | 0.5179 | 0.4962 | 0.5067 | 0.7187 |
| Random Forest | 0.9064 | 0.5677 | 0.7114 | 0.6315 | 0.9449 |
| Gradient Boosting | 0.9158 | 0.6613 | 0.5174 | 0.5804 | 0.9466 |

### Test Set (Final Evaluation):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------|----------|--------|-----|---------|
| Logistic Regression | 0.8642 | 0.4484 | 0.8937 | 0.5972 | 0.9383 |
| Decision Tree | 0.8901 | 0.5129 | 0.4871 | 0.4996 | 0.7142 |
| Random Forest | 0.9063 | 0.5669 | 0.7119 | 0.6312 | 0.9430 |
| Gradient Boosting | 0.9169 | 0.6583 | 0.5453 | 0.5965 | 0.9472 |

**Nhận xét:**
- **Gradient Boosting** đạt ROC-AUC cao nhất (0.9472) và Accuracy cao nhất (0.9169)
- **Random Forest** đạt F1 cao nhất (0.6312), cân bằng tốt nhất giữa Precision và Recall
- **Logistic Regression** có Recall rất cao (0.8937) nhưng Precision thấp (0.4484)
- **Decision Tree** cho kết quả kém nhất cả về F1 và ROC-AUC

## 5. Feature Importance và Insight

### Top features (thường xuất hiện):
1. **`duration`** – Thời gian cuộc gọi (lưu ý: chỉ biết sau khi gọi)
2. **`euribor3m`** – Lãi suất Euribor 3 tháng (chỉ số kinh tế vĩ mô)
3. **`nr.employed`** – Số lượng nhân viên (chỉ số kinh tế)
4. **`poutcome_success`** – Kết quả chiến dịch trước đó = thành công
5. **`age`** – Tuổi khách hàng

### Business Insights:
1. **Lịch sử tương tác** (`poutcome`, `previous`) có mức đóng góp cao → khách hàng từng phản hồi tích cực nên được ưu tiên remarketing.
2. **Yếu tố kinh tế vĩ mô** ảnh hưởng đến quyết định → nên điều chỉnh chiến dịch theo chu kỳ kinh tế.
3. **Cường độ liên hệ** quá cao có thể phản tác dụng → nên giới hạn số lần tiếp cận.
4. **Nhân khẩu học** hỗ trợ phân nhóm nhưng không phải yếu tố quyết định duy nhất.

> **Quan trọng:** Feature importance thể hiện mức độ đóng góp trong mô hình dự đoán, KHÔNG chứng minh quan hệ nhân quả.

## 6. Ứng Dụng Thực Tế Trong E-commerce

| Ứng dụng | Mô tả |
|----------|-------|
| Customer Targeting | Ưu tiên quảng cáo cho nhóm khách hàng có xác suất chuyển đổi cao |
| Personalized Promotion | Phân bổ voucher theo mức xác suất chuyển đổi |
| Remarketing | Ưu tiên khách hàng có lịch sử tương tác tốt |
| Budget Allocation | Phân bổ ngân sách marketing theo dự đoán thay vì đại trà |
| Customer Segmentation | Phân nhóm High/Medium/Low propensity dựa trên predicted probability |

**Mô hình là công cụ hỗ trợ quyết định, không thay thế hoàn toàn chiến lược marketing.**

## 7. Hạn Chế

1. Dataset đến từ ngân hàng Bồ Đào Nha, không hoàn toàn đại diện cho mọi mô hình e-commerce
2. Biến `duration` chỉ biết sau khi liên hệ, không thể dùng cho targeting trước
3. Feature importance ≠ quan hệ nhân quả
4. Mô hình cần retrain định kỳ khi hành vi khách hàng thay đổi
5. Cần kiểm chứng bằng A/B testing trên dữ liệu thực tế trước khi triển khai
6. Kết quả có thể không generalize tốt sang phân khúc khách hàng hoặc thị trường khác

## 8. Kết Luận

Bài toán phân loại khách hàng sử dụng Bank Marketing Dataset đã được thực hiện thành công với 4 thuật toán classification. Kết quả cho thấy các mô hình ensemble (Random Forest, Gradient Boosting) thường đạt hiệu suất tốt hơn về ROC-AUC và F1.

Các feature quan trọng nhất liên quan đến lịch sử chiến dịch, điều kiện kinh tế vĩ mô và đặc điểm tương tác — cung cấp insight hữu ích cho chiến lược marketing. Mô hình có thể hỗ trợ doanh nghiệp tối ưu hóa phân bổ nguồn lực marketing, nhưng cần được kiểm chứng trên dữ liệu thực tế và không nên thay thế hoàn toàn cho chuyên môn marketing.

---

**Ghi chú:** Tất cả kết quả và biểu đồ trong báo cáo được lấy trực tiếp từ notebook chạy thực tế. `random_state=42` được sử dụng cho tính tái lập.
