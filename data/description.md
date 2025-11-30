# Mô tả bộ dữ liệu HWU

Thư mục `data/hwu` chứa 3 file CSV chính dùng cho bài toán phân loại câu (text classification) trên tập dữ liệu HWU:

- `train.csv`: dữ liệu dùng để huấn luyện mô hình.
- `val.csv`: dữ liệu dùng để điều chỉnh siêu tham số và chọn mô hình (validation).
- `test.csv`: dữ liệu chỉ dùng để đánh giá cuối cùng (test), không dùng để huấn luyện.

## Cấu trúc chung các file CSV

Cả 3 file `train.csv`, `val.csv`, `test.csv` đều có cùng cấu trúc cột:

- **`text`**
  - Kiểu dữ liệu: chuỗi (string).
  - Ý nghĩa: câu truy vấn/hội thoại tiếng Anh của người dùng, ví dụ:
    - `"what is the weather today"`
    - `"play some jazz music"`
    - `"set an alarm for 7 am"`.
  - Đây là đầu vào cho mô hình NLP.

- **`category`**
  - Kiểu dữ liệu: chuỗi (string) – nhãn lớp.
  - Ý nghĩa: **nhãn intent / chủ đề** của câu trong cột `text`, ví dụ:
    - `weather_query`
    - `music_play`
    - `alarm_set`
    - `transport_traffic`
    - v.v.
  - Trong mã nguồn, cột này được mã hóa thành số (integer) bằng `LabelEncoder` để mô hình học.

## Mục đích sử dụng

- Các file trong `data/hwu` được lớp `DataLoader` (trong `src/data/data_loader.py`) đọc vào,
  sau đó:
  - Cột `text` được tiền xử lý (lowercase, bỏ ký tự đặc biệt, bỏ stopword, lemmatize, …).
  - Cột `category` được mã hóa thành các chỉ số lớp (label id).

- Bộ dữ liệu này được dùng để:
  - Huấn luyện và so sánh nhiều mô hình phân loại văn bản:
    - TF-IDF + Logistic Regression
    - Word2Vec + Dense
    - LSTM với embedding pretrained
    - LSTM train từ đầu.