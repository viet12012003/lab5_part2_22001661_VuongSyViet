# Báo cáo Lab 5 - Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)

## 1) Các bước hiện thực

### 1.1. Chuẩn bị dữ liệu
- Sử dụng bộ dữ liệu HWU gồm các câu hỏi tiếng Anh được gán nhãn ý định (intent)
- Tiền xử lý dữ liệu:
  - Chuẩn hóa văn bản (lowercase, bỏ dấu câu, ...)
  - Token hóa và chuẩn hóa từ
  - Mã hóa nhãn thành số nguyên

### 1.2. Xây dựng các mô hình
Triển khai 4 mô hình khác nhau:

1. **TF-IDF + Logistic Regression**
   - Sử dụng TfidfVectorizer để trích xuất đặc trưng
   - Huấn luyện mô hình Logistic Regression với class_weight='balanced'

2. **Word2Vec + Dense**
   - Sử dụng Word2Vec để tạo word embeddings
   - Xây dựng mạng Dense đơn giản với 2 lớp ẩn
   - Sử dụng Dropout để tránh overfitting

3. **LSTM (Pre-trained)**
   - Sử dụng pre-trained word embeddings
   - Kiến trúc LSTM 1 lớp với 128 units
   - Dropout 0.5 giữa các lớp

4. **LSTM (Scratch)**
   - Khởi tạo embeddings ngẫu nhiên
   - Kiến trúc tương tự LSTM pre-trained
   - Huấn luyện từ đầu trên tập dữ liệu

### 1.3. Đánh giá mô hình
- Sử dụng các độ đo: Accuracy, F1-score, Confusion Matrix
- Đánh giá trên tập test và phân tích các trường hợp khó

## 2) Hướng dẫn chạy mã

### 2.1. Cài đặt môi trường
```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2.2. Chạy huấn luyện
```bash
# Huấn luyện tất cả các mô hình
python -m src.train
```

### 2.3. Đánh giá mô hình
```bash
# Đánh giá tất cả các mô hình và tạo báo cáo
python -m src.evaluate
```

## 3) Phân tích kết quả

### 3.1. So sánh hiệu suất giữa các mô hình

#### Bảng so sánh các chỉ số đánh giá

| Mô hình | F1-Score (Macro) | Test Loss |
|---------|-----------------|-----------|
| TF-IDF + Logistic Regression | 0.778 | N/A |
| Word2Vec + Dense | 0.747 | 0.813 |
| LSTM (Pre-trained) | 0.578 | 2.214 |
| LSTM (Scratch) | 0.131 | 3.106 |

### 3.2. Đánh giá chi tiết từng mô hình

#### 3.2.1. TF-IDF + Logistic Regression
- **Ưu điểm nổi bật**:
  - Đạt F1-Score cao nhất (0.778) trong số các mô hình
  - Thời gian huấn luyện nhanh chóng
  - Hiệu quả với dữ liệu ít và vừa phải
- **Hạn chế**:
  - Không nắm bắt được ngữ nghĩa phức tạp
  - Không tận dụng được thông tin thứ tự từ trong câu
- **Phù hợp cho**: Ứng dụng cần triển khai nhanh với tài nguyên hạn chế

#### 3.2.2. Word2Vec + Dense
- **Hiệu suất**:
  - F1-Score: 0.747 (thấp hơn TF-IDF 3.1%)
  - Test Loss: 0.813 (tốt nhất trong các mô hình)
- **Đặc điểm**:
  - Tốc độ huấn luyện tương đối nhanh
  - Ổn định trong quá trình huấn luyện
- **Hạn chế**:
  - Gặp khó khăn với các câu phức tạp
  - Phụ thuộc vào chất lượng của Word2Vec

#### 3.2.3. LSTM (Pre-trained)
- **Hiệu suất**:
  - F1-Score: 0.578 (thấp hơn mong đợi)
  - Test Loss: 2.214 (khá cao)
- **Nguyên nhân có thể**:
  - Chưa tối ưu được hyperparameters
  - Cần thêm thời gian huấn luyện
- **Ưu điểm**:
  - Có tiềm năng cải thiện khi điều chỉnh tham số
  - Tận dụng được thông tin thứ tự từ

#### 3.2.4. LSTM (Scratch)
- **Hiệu suất thấp**:
  - F1-Score chỉ đạt 0.131
  - Test Loss cao (3.106)
- **Nguyên nhân**:
  - Thiếu dữ liệu huấn luyện
  - Chưa tối ưu được kiến trúc mô hình
  - Cần điều chỉnh learning rate và các tham số khác

### 3.3. Phân tích sâu hơn

#### 3.3.1. Về F1-Score
- TF-IDF + Logistic Regression cho kết quả tốt nhất, chứng tỏ tính hiệu quả của phương pháp truyền thống khi dữ liệu không quá lớn
- Mô hình dựa trên Word2Vec cho kết quả tương đối tốt, phù hợp với các tác vụ phân loại đơn giản
- Các mô hình LSTM cần được điều chỉnh thêm để phát huy hiệu quả

#### 3.3.2. Về Test Loss
- Word2Vec + Dense có test loss thấp nhất (0.813), cho thấy mô hình này ổn định và phù hợp với dữ liệu
- Các mô hình LSTM có test loss cao, cần xem xét lại kiến trúc và tham số

#### 3.3.3. Thời gian huấn luyện
- TF-IDF nhanh nhất do chỉ cần huấn luyện mô hình đơn giản
- Các mô hình neural network cần nhiều thời gian hơn, đặc biệt khi chạy trên CPU
- LSTM (Pre-trained) mất thời gian lâu nhất do phải tải và xử lý pre-trained embeddings

### 3.4. Hạn chế và thách thức

#### 3.4.1. Hạn chế về hiệu suất
- **Phần cứng**: Chạy trên CPU dẫn đến thời gian huấn luyện lâu
- **Tài nguyên tính toán**: Không thể mở rộng quy mô mô hình lớn
- **Thời gian huấn luyện**: Giới hạn số epoch do thời gian chạy (chỉ training trên 30 epoch)

#### 3.4.2. Các trường hợp khó
Các mô hình thường gặp khó khăn với:
- **Câu phức tạp**: Cấu trúc ngữ pháp phức tạp, câu dài
- **Từ vựng đa nghĩa**: Từ đồng âm khác nghĩa
- **Từ ngoại lai**: Từ viết tắt, từ lóng, tên riêng
- **Ngữ cảnh**: Các câu yêu cầu hiểu ngữ cảnh dài hạn

#### 3.4.3. Hướng cải thiện
- Tối ưu hóa tham số mô hình
- Tăng cường dữ liệu huấn luyện
- Sử dụng các kỹ thuật tiền xử lý nâng cao
- Thử nghiệm với các kiến trúc mạng khác (GRU, Transformer)

## 4) Khó khăn và cách khắc phục

### 4.1. Xử lý dữ liệu
- **Khó khăn**: Dữ liệu không cân bằng giữa các lớp
- **Giải pháp**: Sử dụng class_weight trong quá trình huấn luyện

### 4.2. Vấn đề về hiệu suất
- **Khó khăn**: Thời gian huấn luyện quá lâu trên CPU
- **Giải pháp**: 
  - Giảm kích thước batch
  - Giảm số lượng epochs
  - Tối ưu hóa mã nguồn

### 4.3. Xử lý lỗi
- **Lỗi**: Word2Vec trả về confidence score không chính xác
- **Giải pháp**: Thêm kiểm tra và chuẩn hóa đầu ra của mô hình

## 5) Tài liệu tham khảo
1. Tài liệu chính thức của thư viện:
   - Scikit-learn: https://scikit-learn.org/stable/
   - TensorFlow/Keras: https://www.tensorflow.org/api_docs
   - Gensim: https://radimrehurek.com/gensim/

2. Tài liệu hướng dẫn:
   - TensorFlow Tutorials
   - Keras Documentation
   - Scikit-learn User Guide