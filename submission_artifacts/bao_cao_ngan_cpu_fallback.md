# Báo cáo ngắn: CPU Fallback (Lab 16)

Phương án CPU trên `r5.2xlarge` cho thời gian huấn luyện rất nhanh: `training_seconds = 1.7592s` (load data `2.2208s`).
Chất lượng mô hình đạt mức tốt với `AUC-ROC = 0.95867`, cho thấy khả năng phân biệt gian lận hiệu quả.
Các chỉ số phân lớp thực tế ổn định: `F1 = 0.727273`, `Precision = 0.655738`, `Recall = 0.816327`.
Tốc độ suy luận cao trên CPU: `inference_latency_1row_ms = 2.1776` và `throughput = 545473.69 rows/s`.
So với hướng GPU ban đầu (phục vụ LLM), workload CPU + LightGBM vẫn chứng minh tốt pipeline training/inference.
Lý do chuyển sang CPU là tài khoản mới thường bị khóa quota GPU G/VT (mặc định 0 vCPU), yêu cầu tăng quota có thể bị chậm hoặc từ chối.
Vì vậy, CPU fallback là lựa chọn khả thi để hoàn thành đầy đủ Terraform IaC, benchmark và kiểm tra billing đúng hạn nộp bài.
