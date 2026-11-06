# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603468
- **Họ và tên:** Dương Văn Công
- **SEED**: 757

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |


## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
|---|---:|---:|---:|---:|
| BoW + LogReg | 0.909 | 0.919 | 0.066 | 0 |
| BoW + MLP | 0.965 | 1.000 | 0.197 | 0 |
| MiniLM + kNN | 0.860 | 1.000 | 21.557 | 0 |
| MiniLM + LogReg | 0.939 | 1.000 | 22.361 | 0 |
| MiniLM + MLP | 0.944 | 1.000 | 27.163 | 0 |
| LLM nhỏ zero-shot | 0.000 | 0.000 | 1262.132 | 0.028 |
| LLM lớn zero-shot | 0.000 | 0.000 | 1544.026 | 0.285 |
| LLM nhỏ few-shot | 0.000 | 0.000 | 1240.589 | 0.111 |

BoW + MLP có F1 test cao nhất và độ trễ rất thấp. Các mô hình MiniLM cũng cho kết quả khá tốt nhưng xử lý chậm hơn. Kết quả LLM chưa đáng tin cậy do có vấn đề khi đọc nhãn đầu ra.

## 3. Dự đoán và thực tế

1. Mình dự đoán F1 của Logistic Regression khoảng 0.85. Thực tế đạt 0.909, cao hơn dự đoán vì các từ khóa trong dữ liệu giúp mô hình phân biệt spam khá tốt.

2. Mình dự đoán learning rate = 50 sẽ khiến loss dao động hoặc không hội tụ. Nhưng thực tế loss giảm nhanh xuống 0.005, cho thấy learning rate lớn chưa chắc gây mất ổn định trong mọi trường hợp.

3. Mình dự đoán few-shot sẽ giúp LLM tăng F1 nhờ có thêm ví dụ. Tuy nhiên F1 vẫn bằng 0 và có 40 nhãn không hợp lệ, nên chưa thể kết luận few-shot có cải thiện hay không.

## 4. Phân tích 50 case khó

BoW + MLP tụt nhiều nhất, từ F1 = 1.000 trên SUB 40 xuống 0.368 trên 50 case khó. Nhóm spam giống tin nhắn tự nhiên là nhóm khó nhất, vì nội dung không chứa nhiều từ khóa spam quen thuộc.

Hai case đáng chú ý:

- Case #14: Tin quảng cáo kiếm tiền tại nhà nhưng viết giống một cuộc trò chuyện bình thường, khiến mô hình dễ bỏ sót.
- Case #22: Tin giả mạo ngân hàng yêu cầu xác minh tài khoản. Mô hình không nhận ra dấu hiệu lừa đảo dù nội dung có tính khẩn cấp.

MiniLM + kNN đạt kết quả tốt nhất trên 50 case khó với F1 = 0.727 và accuracy = 70%. Điều này cho thấy F1 cao trên tập test thường chưa chắc phản ánh đúng khả năng xử lý dữ liệu khó.

## 5. Thử thách cá nhân

Mình nhận đề #2: Thay đổi kích thước vocab thành 100 và 5000 từ, sau đó train lại Logistic Regression.

Cách làm: Dùng build_vocab với max_size tương ứng, vectorize lại dữ liệu, train mô hình và tính F1 trên tập test.

Kết quả:
- Vocab 100 từ: F1 = 0.855
- Vocab 5000 từ: F1 = 0.909

Khi tăng vocab, mô hình nhận diện được nhiều từ đặc trưng của spam hơn nên F1 tăng. Tuy nhiên, vocab lớn cũng làm tăng số chiều vector và tốn thêm tài nguyên xử lý.

## 6. Phương án cho sản phẩm thật

Mình chọn BoW + MLP vì F1 test đạt 0.965, p95 chỉ 0.197 ms/tin, đáp ứng yêu cầu dưới 50 ms. Tốc độ đo được khoảng 55.943 tin/giây, cao hơn mức trung bình 116 tin/giây của hệ thống.

Tuy nhiên, mô hình vẫn dễ bỏ sót spam viết tự nhiên. Vì vậy mình sẽ kết hợp MiniLM để kiểm tra thêm các tin có xác suất không rõ ràng (0.3–0.7).

Cách này giúp tiết kiệm chi phí API và tăng khả năng xử lý tin khó, nhưng phải đánh đổi thêm tài nguyên tính toán. Cần kiểm tra lại độ trễ và chất lượng trên dữ liệu thực tế trước khi triển khai.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI hỗ trợ gì | Đã sửa / kiểm chứng gì |
|---|---|---|---|
| 1 | Giải thích BoW, vocab và Logistic Regression | Giải thích cách vector hóa và train | Đọc code, chạy thử và kiểm tra kết quả |
| 2 | Giải thích learning rate và overfitting | Phân tích đường loss | Đối chiếu với biểu đồ thực tế |
| 4 | Hướng dẫn gọi OpenAI API | Gợi ý code gọi API, tạo prompt và parse nhãn | Chạy thử API và kiểm tra output |
| 5 | Phân tích F1, Pareto và stress test | Giải thích kết quả các mô hình | Đối chiếu với bảng kết quả |
| Bonus | Thay đổi kích thước vocab | Gợi ý code thử nghiệm | Chạy lại, F1 đạt 0.855 và 0.909 |
| Cuối bài | Sửa lỗi UnicodeEncodeError | Hướng dẫn dùng UTF-8 | Sửa code xuất JSON |

Qua bài lab, mình hiểu rõ hơn cách hoạt động của BoW, MLP, Embedding và LLM. Điều mình bất ngờ nhất là mô hình có F1 rất cao trên tập test thường vẫn có thể sai nhiều khi gặp dữ liệu khó. Mình nhận ra rằng khi đánh giá mô hình AI không nên chỉ nhìn vào F1 mà còn cần quan tâm đến độ trễ, chi phí và khả năng xử lý dữ liệu thực tế.

