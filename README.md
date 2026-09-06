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

## Repo Git

```bash
git clone https://github.com/thanhnghi-do-2k3/llm-as-judge.git
```

`WORK_REPO_URL` trong `main-experiment.ipynb` đã trỏ tới repo này. Mở notebook
trực tiếp từ GitHub bằng Colab rồi chọn **Runtime → Run all**; notebook sẽ tự
clone source và resource vào runtime khi cần.

## Cấu hình trong notebook

Toàn bộ tùy chọn cần chỉnh nằm trong cell `setup` đầu tiên:

| Tùy chọn | Mặc định | Ý nghĩa |
|---|---:|---|
| `REGENERATE_TRANSLATIONS` | `False` | Dùng A/B trong Git; bật `True` để sinh/resume A bằng Gemini và B bằng Google Translate. |
| `REGENERATE_DAMAGE` | `False` | Dùng zero/few-shot trong Git; bật `True` để sinh/resume damage mới. |
| `RUN_METRIC_SMOKE_TEST` | `True` | Test một cấu hình của từng họ metric trên 24 hàng. |
| `RUN_FULL_METRICS` | `False` | Không chấm full cho tới khi smoke test đã pass. |

Gemini key được phân thành hai mảng nhưng chạy qua cùng một scheduler:

- `STANDARD_GEMINI_API_KEYS`: quota thường, dùng chung một worker và khoảng cách
  4,2 giây như rule an toàn cũ.
- `HIGH_QUOTA_GEMINI_API_KEYS`: mỗi key có pool riêng, mặc định bốn worker và
  khoảng cách 0,25 giây. Chỉ dùng cho project đã kiểm tra RPM/RPD cao; không có
  key nào thực sự không giới hạn. Nhiều key thuộc cùng một project vẫn chia sẻ
  quota của project đó.

Nếu hai cờ regenerate đều `False`, notebook không gọi Gemini kể cả khi mảng key
có giá trị. Khi sinh A/B lại, notebook dùng ngay file mới ở các bước sau và
không ghép `scores.xlsx` cũ, vì điểm đó thuộc các bản dịch A/B trong Git.

## Hành vi của Run All

- Tự clone repo nếu runtime chưa có source.
- Đọc resource bằng đường dẫn tương đối trong repo.
- Mặc định `REGENERATE_TRANSLATIONS=False` và `REGENERATE_DAMAGE=False`: dùng
  toàn bộ artifact trong Git, không gọi Gemini và không tốn quota.
- Probe NumPy/Pandas, BERTScore, COMET và BLEURT trước khi chạy full.
- Smoke test một config mỗi họ trên input nhỏ; lỗi dừng sớm và chỉ rõ log.
- Mặc định `RUN_FULL_METRICS=False`: Run All chỉ chạy heavy smoke test; đổi thành
  `True` sau khi smoke pass để chấm toàn bộ.
- Full metric ghép human/B1/zero/few để mỗi model nặng chỉ nạp một lần.
- Checkpoint tự resume trong cùng `RUN_NAME`.

Lần chạy đầu vẫn phải tải package và model metric từ Internet. Đây là cache
runtime, không phải file cần upload. Dùng Python 3.12 trở xuống và bật GPU khi
`RUN_HEAVY_METRICS=True`.

Nếu runtime từng chạy một bản notebook cũ đã thay NumPy/Pandas, chọn
**Runtime → Disconnect and delete runtime** trước khi chạy bản mới. Bootstrap
hiện tại giữ nguyên NumPy/Pandas/SciPy của runtime và sẽ dừng với thông báo rõ
nếu phát hiện phiên bản trong RAM khác phiên bản trên disk.
