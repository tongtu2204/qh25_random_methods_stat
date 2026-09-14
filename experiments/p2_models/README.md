# Phần 2 — Mô hình dự đoán pool và từng giải

Phần 2 được tách thành hai bài toán độc lập nhưng dùng chung dữ liệu đầy đủ
27 kết quả mỗi ngày trong `data/raw/kqxsmb_all_prizes_2007_2026.csv`.

## P2A — dự đoán pool chữ số của cả ngày

## Đơn vị đánh giá

Mỗi ngày có 27 kết quả từ toàn bộ các nhóm giải. Với mỗi vị trí trong số có
5 chữ số, target là vector nhị phân 10 chiều:

```text
target[position, digit] = 1
```

nếu chữ số đó xuất hiện ít nhất một lần ở vị trí tương ứng trong bất kỳ kết
quả nào của ngày. Các giải ngắn được căn phải: giải 4 chữ số chỉ đóng góp vào
4 vị trí cuối, không tạo chữ số `0` giả ở đầu. File chuẩn bị dữ liệu cũng lưu
`eligible_count`, số lần xuất hiện của từng chữ số và toàn bộ `pool_numbers`
để P3 sinh tổ hợp.

## Mô hình

- `uniform_27_iid`: baseline Uniform ở cấp độ ngày, dùng
  `1 - (1 - 0,1)^27` cho xác suất xuất hiện ít nhất một lần trong 27 kết quả;
- `expanding_frequency`: tần suất xuất hiện tích lũy theo ngày;
- `rolling_frequency_w30`, `w90`, `w365`: tần suất trong cửa sổ gần nhất;
- `markov_presence`: xác suất xuất hiện hôm nay phụ thuộc trạng thái xuất hiện
  hoặc không xuất hiện của ngày trước;
- `random_forest`: học từ các vector target trễ 1, 2, 3, 7, 14 và 30 ngày.

## P2B — dự đoán đích danh từng giải

Mỗi target là một slot cụ thể `prize + prize_index` (ví dụ `Giải ba_4`),
không chỉ là giải đặc biệt. Mô hình dự đoán phân phối chữ số theo vị trí của
đúng slot đó, sinh danh sách Top-k số và chấm hit theo đúng độ dài của giải.
Không dùng ký tự phụ đặc biệt; các pool/độ dài được lấy trực tiếp từ dữ liệu
full-prize.

Protocol của Phần 2:

- lịch sử huấn luyện: từ 2007 đến hết 2022;
- validation: 2023–2024;
- 2025–2026 không dùng trong Phần 2, được khóa riêng để đánh giá chiến lược
  thực tế ở Phần 3.

## Chạy lại

```bash
python experiments/p2_models/00_prepare_daily_targets.py
python experiments/p2_models/01_daily_digit_models.py
python experiments/p2_models/02_daily_model_evaluation.py
python experiments/p2_models/03_boosted_daily_models.py
python experiments/p2_models/04_prize_target_models.py
python experiments/p2_models/05_boosted_prize_target_models.py
```

Kết quả nằm tại:

```text
data/processed/daily_digit_targets.csv
artifacts/p2_models/model_summary.csv
artifacts/p2_models/daily_scores.csv.gz
artifacts/p2_models/actual_digit_ranks.csv.gz
artifacts/p2_models/rank_summary.csv
artifacts/p2_models/calibration.csv
artifacts/p2_models/boosted/boosted_summary.csv
artifacts/p2_models/boosted/boosted_calibration.csv
artifacts/p2_models/figures/
artifacts/p2_models/prize_target/prize_target_predictions.csv.gz
artifacts/p2_models/prize_target/prize_target_summary.csv
artifacts/p2_models/prize_target_boosted/boosted_prize_target_predictions.csv.gz
artifacts/p2_models/prize_target_boosted/boosted_prize_target_summary.csv
```


## Model boosting

- `03_boosted_daily_models.py`: XGBoost và CatBoost, mỗi model gồm 50 classifier nhị phân cho 5 vị trí × 10 chữ số.
- Cài dependency bằng `python -m pip install -r requirements.txt`.
- Kết quả được đánh giá cùng protocol và metric của `02_daily_model_evaluation.py`.

## P2C — dự đoán riêng giải Đặc biệt

P2C chỉ sử dụng đúng một quan sát mỗi ngày: chuỗi 5 chữ số của giải
`Đặc biệt`. Các số có chữ số 0 ở đầu được giữ nguyên dưới dạng chuỗi, ví dụ
`01234`; không trộn dữ liệu của các giải ngắn vào target này.

Các model gồm:

- `uniform_random`;
- tần suất chữ số expanding;
- tần suất chữ số rolling với cửa sổ 30, 90, 180, 365 và 730 ngày;
- Markov theo từng vị trí;
- CatBoost/XGBoost theo từng vị trí.

Xác suất của một số 5 chữ số được tính từ tích xác suất của năm vị trí, sau đó
xếp hạng toàn bộ 100.000 số từ `00000` đến `99999`. Kết quả chỉ lưu Top-100
mỗi ngày nhưng metric được tính trên toàn bộ không gian số.

Chạy:

```bash
python experiments/p2_models/06_special_prize_models.py
python experiments/p3_strategies/03_special_prize_backtest.py
```

Kết quả P2C:

```text
artifacts/p2_models/special_prize/special_prize_predictions.csv.gz
artifacts/p2_models/special_prize/special_prize_model_summary.csv
```

Kết quả P3C:

```text
artifacts/p3_strategies/special_prize/special_prize_daily_results.csv.gz
artifacts/p3_strategies/special_prize/special_prize_prize_hits.csv.gz
artifacts/p3_strategies/special_prize/special_prize_strategy_summary.csv
artifacts/p3_strategies/special_prize/special_prize_by_prize_summary.csv
```

`special_prize_strategy_summary.csv` là kết quả chính của việc săn Đặc biệt.
`special_prize_by_prize_summary.csv` thống kê thêm tỷ lệ trúng từng giải khác
bằng chính các vé 5 chữ số đã chọn để săn Đặc biệt. Hai chế độ lợi nhuận được
tách riêng: `db_only_profit` chỉ tính giải Đặc biệt và `all_prizes_profit`
cộng cả các giải khác.
