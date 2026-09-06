# Meta-Judge Zh→Vi — repo chạy Colab

Thư mục này là một repo độc lập chứa notebook, source tính metric và toàn bộ
resource đầu vào. `main-experiment.ipynb` có thể Run All trên local hoặc clone
repo trên Colab; không cần upload ZIP hay file dữ liệu thủ công.

## Cấu trúc

- `main-experiment.ipynb`: pipeline đầy đủ, mặc định dùng damage đã sinh sẵn.
- `meta-judge/`: source metric và dependency đã chốt cùng revision notebook.
- `resources/source.csv`: 300 câu nguồn, reference và bản dịch A/B.
- `resources/scores.xlsx`: điểm người chấm.
- `resources/human_all.jsonl`: 600 hàng human dùng cho kiểm tra độc lập.
- `resources/zero_shot.jsonl`, `few_shot.jsonl`: mỗi nhánh 1.800 hàng.
- `cheat-runtime/`: notebook chuyên dụng cho generation hoặc chấm từng nhánh.

## Thiết lập Git một lần

```bash
cd work
git remote add origin https://github.com/<tai-khoan>/<repo>.git
git add .
git commit -m "Portable Meta-Judge Colab workspace"
git push -u origin main
```

Sau khi tạo remote, thay `WORK_REPO_URL` trong cell cấu hình của
`main-experiment.ipynb` bằng URL HTTPS đó, commit và push lần nữa. Mở notebook
từ GitHub bằng Colab rồi chọn **Runtime → Run all**.

## Hành vi của Run All

- Tự clone repo nếu runtime chưa có source.
- Đọc resource bằng đường dẫn tương đối trong repo.
- Không gọi Gemini và không dùng API key khi `USE_PREGENERATED_DAMAGE=True`.
- Probe NumPy/Pandas, BERTScore, COMET và BLEURT trước khi chạy full.
- Smoke test một config mỗi họ trên input nhỏ; lỗi dừng sớm và chỉ rõ log.
- Full metric ghép human/B1/zero/few để mỗi model nặng chỉ nạp một lần.
- Checkpoint tự resume trong cùng `RUN_NAME`.

Lần chạy đầu vẫn phải tải package và model metric từ Internet. Đây là cache
runtime, không phải file cần upload. Dùng Python 3.12 trở xuống và bật GPU khi
`RUN_HEAVY_METRICS=True`.
