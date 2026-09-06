# Pipeline Meta-Judge tiếng Việt

> Với repo Colab độc lập, dùng `../README.md` và
> `../main-experiment.ipynb`. Source đã được vendor trực tiếp; các đoạn mô tả
> bundle ZIP/virtualenv bên dưới là tài liệu lịch sử của pipeline gốc.

Tài liệu này là note vận hành cho kế hoạch mới trong `documents/plans/plans.pdf`.
Phạm vi của pipeline:

- **Generation:** prompt tiếng Việt, Gemini sinh damage 0–5 ở hai nhánh zero-shot/few-shot, baseline mất dấu và audit output.
- **Metrics:** chạy 7 họ metric, 28 cấu hình paper, hai kiểu tokenization cho nhánh cần so sánh, B1/B2 và các correlation.

Pipeline này không huấn luyện model. Gemini chỉ sinh dữ liệu hư hại; BLEU/chrF/ROUGE/METEOR/BERTScore/COMET/BLEURT mới là metric được kiểm định.

## 1. Kiến trúc

```text
vn_meta_judge/
├── data/loaders.py              Google Sheet XLSX → schema chuẩn → JSONL
├── preprocessing/vietnamese.py  NFC, whitespace, âm tiết, underthesea, bỏ dấu
├── generation/prompts.py        prompt VI + 6 damage levels
├── generation/api_runner.py     checkpoint/resume dùng chung cho backend API
├── generation/env.py            nạp riêng `src/meta-judge/.env`
├── generation/gemini.py         Gemini API, temperature=0, retry, resume, cache
├── generation/deepseek.py       DeepSeek ablation tùy chọn, output tách riêng
├── generation/rule_based.py     B1: nhiễu cơ học, không gọi API
├── metrics/specs.py              7 × 4 cấu hình, vi thay cho cs ở BERTScore
├── metrics/runner.py             adapter lazy tới source metric của tác giả
├── evaluation/correlations.py   Eq. 1 r_hum, Eq. 2 r_syn, Eq. 3 MC
└── cli.py                        các lệnh export/generate/metric/correlation
```

Ranh giới dữ liệu là bắt buộc:

```text
D (human)   = source ZH + reference VI + translation A/B + human_score
D_syn       = reference VI + damaged VI + damage_level
r_hum       = correlation(metric(D), human_score)
r_syn       = correlation(metric(D_syn), -damage_level)
MC          = correlation(r_hum_vector, r_syn_vector)
```

`D` và `D_syn` không được dùng chung bản dịch hệ A/B. Hệ A/B là output thật để tính human validation; Gemini/B1 là output hư để tính synthetic validation.

## 2. Source tác giả dùng lại được gì?

| Thành phần source tác giả | Quyết định | Cách dùng trong pipeline mới |
|---|---|---|
| `src/human_feedback_datasets/processed_dataset.py` | Dùng được | Schema JSONL của tác giả là tham chiếu; loader mới dùng schema rõ hơn cho VI và giữ trường metadata/quality flags. |
| `src/generation/generate.py` | Dùng được có điều kiện | Có local model, Gemini/OpenAI API, retry và checkpoint. Phù hợp tái hiện dataset cũ; pipeline mới tách prompt/data VI ra để không sửa `DATASET_CONFIGS` global. |
| `src/metrics/hf_metrics.py` | Dùng lại | `vn_meta_judge.metrics.runner` import lazy `METRIC_FNS`; metric implementation không bị viết lại. Cấu hình BERTScore VI nằm ở `vn_meta_judge/config.py`. |
| `src/metrics/eval_dataset.py` | Tham chiếu/adaptor | Logic batch/resume hữu ích nhưng phụ thuộc trực tiếp schema tác giả; runner mới ghi envelope có `scores/errors/timings/config`. |
| `src/metrics/correlation.py` | Tham chiếu | Có human/metric correlation, nhưng runner mới ghi rõ hướng dấu `-damage_level` theo Equation 2. |
| `src/metrics/meta_correlation.py` | Dùng cho B2 | Chạy đối chiếu source/correlation cũ; không dùng để thay cho report chính của pipeline VI vì source cũ có quy ước đảo dấu riêng. |

### Quy tắc backend và ý nghĩa thí nghiệm

Kế hoạch chính dùng **Gemini 2.5 Flash Lite** làm generator vì đây là model đã
được chốt trong kế hoạch nhóm. Gemini chỉ tạo `D_syn`; nó không chấm A/B và
không thay human judgment. `r_hum` vẫn phải tính từ điểm người chấm trong
Sheet, còn `r_syn` tính từ mức damage được yêu cầu/audit.

DeepSeek được setup như **ablation tùy chọn**, không phải fallback tự động:

- Không gộp row Gemini và DeepSeek vào cùng một `D_syn` hoặc cùng một vector
  score.
- Mỗi backend có file output, audit và correlation report riêng; metadata mỗi
  row ghi `backend` và `model_name`.
- Nếu Gemini lỗi giữa chừng, resume lại Gemini; không đổi sang DeepSeek rồi
  coi đó là cùng một run.
- Nếu hai backend cho kết quả rất khác nhau, đó là kết quả về độ nhạy với
  generator, không được che bằng cách lấy trung bình hoặc trộn dữ liệu.

## 3. Chuẩn bị môi trường

```bash
cd src/meta-judge
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
# chỉ người 3 cần bước nặng này:
python -m pip install -r requirements-metrics.txt
# COMET và BLEURT cần hai venv riêng vì ràng buộc Protobuf không tương thích:
# python -m pip install -r requirements-metric-comet.txt
# python -m pip install -r requirements-metric-bleurt.txt
export PYTHONPATH="$PWD/src"
```

Repo đã có file [`.env`](.env) local bị ignore và file mẫu [`.env.example`](.env.example). Điền key vào `.env`:

```dotenv
GEMINI_API_KEY=AIza...
# Hoặc pool JSON/phân cách dấu phẩy nếu nhóm sở hữu nhiều key hợp lệ:
GEMINI_API_KEYS=key_1,key_2
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_MAX_OUTPUT_TOKENS=400
GEMINI_WORKERS=2
```

CLI tự nạp `.env`; biến đã export trong shell được ưu tiên. Pipeline coursework
dùng `GEMINI_API_KEY` hoặc pool `GEMINI_API_KEYS` cho primary run và không có fallback âm thầm sang backend
khác. Pool phân phối round-robin, không in key và không được dùng để né quota. Nếu cần ablation DeepSeek, điền riêng `DEEPSEEK_API_KEY`; key không được
commit hoặc gửi trong chat.

Nếu chỉ muốn kiểm tra loader, prompt và B1 thì chưa cần cài bộ metric nặng.

## 4. Dữ liệu nhóm phải chuẩn bị

### Đã có thể dùng ngay

Google Sheet nhóm cung cấp có thể tải bằng **File → Download → Microsoft Excel (.xlsx)**. Lưu thành một file local, ví dụ:

```text
data/incoming/Cham_diem_DeTai46.xlsx
```

Loader đọc `Cham_diem`; nếu workbook có `Du_lieu_goc` thì dùng sheet đó để kiểm tra căn chỉnh, còn Google Sheet export một-sheet thì lấy source/reference ngay trong `Cham_diem`. Không lấy số đã tính sẵn trong `Tong_hop` làm nguồn tính toán.

Khi tải lại link Google Sheet ngày 2026-09-05 thành
`documents/Cham_diem_DeTai46_latest.xlsx` và chạy loader, thấy:

- 300 câu nguồn/tham chiếu;
- 2 hệ dịch và 600 row A/B; đủ 300 điểm chính và 300 điểm final cho mỗi hệ;
- 120 cross-score row cho mỗi hệ (240 raw rows), trong đó 234 row còn lại sau
  khi loại các dòng có reference-alignment flag;
- 9 ID bị gắn cờ reference-alignment (18 row A/B):
  `0040, 0058, 0102, 0123, 0148, 0184, 0207, 0231, 0268`;
- 9/9 final-score override của A và 9/9 của B đều có ghi chú giải thích lỗi
  reference; không có arithmetic mismatch nào không được chú thích;
- final-score mean/SD là A `84.23/9.38`, B `78.68/13.59`.

Loader xuất `human_benchmark_all.jsonl`, `human_benchmark_clean.jsonl`,
`human_benchmark_scored.jsonl` và `human_cross_ratings.jsonl`. Bản `scored` hiện
là 582 row clean có điểm; 9 ID bị loại có chủ ý, không phải missing score. Vì
reference chưa được quyết định giữ/sửa/loại, `ready_for_r_hum` vẫn là `false`;
có thể chạy provisional correlation trên `human_benchmark_clean.jsonl` nhưng
phải ghi rõ đây là kết quả sau exclusion.

### Kết quả audit runtime mới nhất

- B1 rule-based: 1.800 row (`300 × 6`), đủ mọi level, không duplicate run key,
  không API error, không empty output và không meta-text.
- 16 cấu hình metric nhẹ đã chạy cho cả syllable và underthesea; không có metric
  failure trong các file score.
- Provisional trên 582 row clean: syllable có MC Spearman `0.4471`, Kendall
  `0.2500`; underthesea có MC Spearman `0.7029`, Kendall `0.5833`.
- Đây chưa phải kết luận cuối: 9 reference lỗi cần quyết định, B1 là baseline cơ
  học, và 12 cấu hình BERTScore/COMET/BLEURT còn thiếu model cache.
- Gemini và DeepSeek chưa gọi vì `.env` chưa có key.

### Chưa cần bạn gửi thêm ở bước code

Hiện tại chưa cần thêm data nếu file XLSX trên vẫn truy cập được. Để chạy full experiment, người dùng cần tự đặt các secret/asset sau trên máy hoặc Kaggle:

1. `GEMINI_API_KEY` hoặc `GEMINI_API_KEYS` — không ghi key vào repo và không gửi key trong chat.
2. Các model cache/download được cho BERTScore, COMET và BLEURT nếu muốn đủ 28 cấu hình.
3. Nếu muốn tái tạo 1.000 câu VLSP thay vì dùng 300 câu trong Sheet, cần
   `HF_TOKEN` và quyền gated của `VLSP2023-MT/ViBidirectionMT-Eval`.
4. `DEEPSEEK_API_KEY` chỉ cần nếu chạy nhánh ablation DeepSeek; không cần cho primary.

## 5. Người 2 — phải làm gì và chạy thế nào?

### Bước 2.1 — Chuẩn hóa data cho cả nhóm

```bash
PYTHONPATH=src python -m vn_meta_judge export-benchmark \
  --scores-xlsx data/incoming/Cham_diem_DeTai46.xlsx \
  --output-dir data/vietnamese
```

Kết quả:

- `references_300.jsonl`: 300 reference VI + source ZH, dùng làm input cho Gemini/B1;
- `human_benchmark_all.jsonl`: 600 row A/B, gồm cả row thiếu điểm để audit;
- `human_benchmark_clean.jsonl`: bản không có reference-alignment flag, nhưng vẫn giữ row thiếu điểm;
- `human_benchmark_scored.jsonl`: 582 row clean có final score;
- `human_cross_ratings.jsonl`: 234 row clean chấm chéo, có primary/cross score và rater;
- `benchmark_audit.json`: số dòng, số điểm thiếu, rater và quyết định dùng dữ liệu.

### Bước 2.2 — Pilot trước khi tốn API

```bash
PYTHONPATH=src python -m vn_meta_judge gemini \
  --input data/vietnamese/references_300.jsonl \
  --output data/vietnamese/damaged_gemini_zero_shot_pilot.jsonl \
  --prompt-type zero_shot \
  --limit 20
```

Pilot phải được 4 người đọc và kiểm tra: output không rỗng, đúng tiếng Việt, level 0 bảo toàn nghĩa, level cao hỏng tăng dần, không sinh lời giải thích/nhãn. Kế hoạch yêu cầu audit 30 mẫu cho phân tích lỗi; `--limit 20` ở trên chỉ là smoke/prompt test, không thay thế audit 30 mẫu.

Sau khi prompt đạt, chạy hai nhánh đầy đủ:

```bash
PYTHONPATH=src python -m vn_meta_judge gemini --input data/vietnamese/references_300.jsonl \
  --output data/vietnamese/damaged_gemini_zero_shot.jsonl --prompt-type zero_shot

PYTHONPATH=src python -m vn_meta_judge gemini --input data/vietnamese/references_300.jsonl \
  --output data/vietnamese/damaged_gemini_few_shot.jsonl --prompt-type few_shot
```

Mỗi file đầy đủ có 1.800 row (`300 × 6`). Hai file có tổng 3.600 row. Code ghi từng row ngay sau API call nên có thể chạy tiếp sau gián đoạn; không xóa file cache để chạy lại.

Sau mỗi nhánh generation, chạy audit tự động:

```bash
PYTHONPATH=src python -m vn_meta_judge audit-damage \
  --input data/vietnamese/damaged_gemini_zero_shot.jsonl \
  --output data/vietnamese/audit_gemini_zero.json
```

Audit kiểm tra đủ 6 level/ID, duplicate key, row lỗi, output rỗng và meta-text. Để tính accuracy của việc 4 người đoán lại level, tạo `manual_audit_30.jsonl` với các field `id`, `requested_level`, `observed_level`, rồi truyền thêm `--manual-audit`.

### Nhánh DeepSeek tùy chọn — chỉ dùng cho ablation

Chỉ chạy sau khi Gemini primary đã có pilot đạt gate. Cấu hình trong `.env`:

```dotenv
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MAX_OUTPUT_TOKENS=400
```

Pilot DeepSeek được lưu ở file khác hoàn toàn:

```bash
PYTHONPATH=src python -m vn_meta_judge deepseek \
  --input data/vietnamese/references_300.jsonl \
  --output data/vietnamese/damaged_deepseek_zero_shot_pilot.jsonl \
  --prompt-type zero_shot --limit 20

PYTHONPATH=src python -m vn_meta_judge audit-damage \
  --input data/vietnamese/damaged_deepseek_zero_shot_pilot.jsonl \
  --output data/vietnamese/audit_deepseek_zero_pilot.json
```

Sau pilot, nếu thực sự cần full ablation thì dùng tên file có `deepseek`,
chạy metrics/correlation riêng và báo cáo cạnh kết quả Gemini. Không được đổi
backend giữa chừng trên cùng một output file.

### Bước 2.3 — Baseline B1, không tốn API

```bash
PYTHONPATH=src python -m vn_meta_judge rule-baseline \
  --input data/vietnamese/references_300.jsonl \
  --output data/vietnamese/damaged_rule_based.jsonl
```

B1 có 1.800 row và sáu level: identity, bỏ dấu, bỏ từ, dropout + swap. Đây là baseline cơ học để trả lời “LLM có tạo thêm giá trị hay không?”, không được gọi là semantic gold.

### Smoke test offline — không phải code production

Trước khi cài metric nặng hoặc dùng API, chạy mock end-to-end riêng trong `tests/`:

```bash
PYTHONPATH=src python tests/mock_end_to_end.py
```

Mock chỉ tạo fake model/metric score để kiểm tra JSONL → correlation → report. Nó không được dùng để viết Results và không được import vào `vn_meta_judge`.

## 6. Người 3 — phải làm gì và chạy thế nào?

### Bước 3.0 — Chạy Người 3 trên Colab và lưu kết quả về Drive

Để chạy đủ model-based metric trên GPU, dùng notebook:
`notebooks/person3_colab_metrics.ipynb`.

Notebook này sẽ:

- mount Google Drive và dùng `MyDrive/meta_judge_vietnamese/` làm nơi lưu kết quả;
- đọc `Cham_diem_DeTai46_latest.xlsx` và zip source `src-meta-judge.zip`;
- cài dependencies, tải/cache BERTScore, COMET và BLEURT;
- chạy B1, human branch, 16 metric nhẹ, 12 metric model-based và correlation;
- ghi từng score file và log vào `results/latest/`, nên có thể chạy lại sau khi
  Colab ngắt mà không mất metric đã hoàn tất;
- tự chạy thêm Gemini branch nếu Người 2 copy JSONL Gemini vào Drive.

Kết quả sau khi chạy nằm tại:

```text
MyDrive/meta_judge_vietnamese/results/latest/
```

Người 3 không cần `GEMINI_API_KEY`; chỉ cần Gemini output JSONL khi muốn chấm
nhánh Gemini. `HF_TOKEN` là secret tùy chọn để tải model từ Hugging Face.

### Bước 3.1 — Test metric nhỏ trước

```bash
PYTHONPATH=src python -m vn_meta_judge metrics \
  --input data/vietnamese/damaged_gemini_zero_shot_pilot.jsonl \
  --output data/vietnamese/scores_pilot.json \
  --families BLEU chrF \
  --batch-size 8
```

Nếu lệnh lỗi import `metrics.hf_metrics`, kiểm tra `PYTHONPATH=src` và file
requirements của đúng họ metric. COMET và BLEURT không được cài chung một venv:
COMET 2.2.x yêu cầu Protobuf 4, còn TensorFlow 2.20 của BLEURT yêu cầu Protobuf
5 trở lên. Notebook dùng ba worker environment riêng. Nếu COMET/BLEURT không
tải được, lưu lỗi và ghi vào Limitations; không giả vờ đủ 28 cấu hình.

### Bước 3.2 — Chạy 28 cấu hình paper

Primary run dùng tokenization `syllable`, tức giữ khoảng trắng giữa âm tiết tiếng Việt:

```bash
PYTHONPATH=src python -m vn_meta_judge metrics \
  --input data/vietnamese/damaged_gemini_zero_shot.jsonl \
  --output data/vietnamese/scores_gemini_zero_syllable.json \
  --tokenization syllable --batch-size 32
```

Lặp cho `damaged_gemini_few_shot.jsonl` và `damaged_rule_based.jsonl`. Nhánh tokenization underthesea là sensitivity riêng, không cộng vào 28 cấu hình primary:

```bash
PYTHONPATH=src python -m vn_meta_judge metrics \
  --input data/vietnamese/damaged_gemini_zero_shot.jsonl \
  --output data/vietnamese/scores_gemini_zero_underthesea.json \
  --tokenization underthesea --families BLEU chrF --batch-size 32
```

Nếu dùng underthesea để so sánh `MC`, phải chạy **cả human branch** bằng cùng tokenization; không lấy r_hum âm tiết ghép với r_syn underthesea:

```bash
PYTHONPATH=src python -m vn_meta_judge metrics \
  --input data/vietnamese/human_benchmark_clean.jsonl \
  --output data/vietnamese/scores_human_clean_underthesea.json \
  --tokenization underthesea --families BLEU chrF --batch-size 32
```

Chạy metric trên `human_benchmark_clean.jsonl` (hoặc `all`, nhưng phải ghi lựa chọn) để có score vector cho Equation 1:

```bash
PYTHONPATH=src python -m vn_meta_judge metrics \
  --input data/vietnamese/human_benchmark_clean.jsonl \
  --output data/vietnamese/scores_human_clean.json \
  --tokenization syllable --batch-size 32
```

### Bước 3.3 — Tính ba correlation

Nếu human scores chưa hoàn tất, chỉ được tính tạm `r_syn` để kiểm tra B1 hoặc
pilot; lệnh này **không tạo MC**:

```bash
PYTHONPATH=src python -m vn_meta_judge synthetic-correlate \
  --synthetic-jsonl data/vietnamese/damaged_rule_based.jsonl \
  --synthetic-scores data/vietnamese/scores_rule_based_light_v2.json \
  --output data/vietnamese/r_syn_rule_based_syllable.json
```

```bash
PYTHONPATH=src python -m vn_meta_judge correlate \
  --human-jsonl data/vietnamese/human_benchmark_clean.jsonl \
  --human-scores data/vietnamese/scores_human_clean.json \
  --synthetic-jsonl data/vietnamese/damaged_gemini_zero_shot.jsonl \
  --synthetic-scores data/vietnamese/scores_gemini_zero_syllable.json \
  --output data/vietnamese/correlation_gemini_zero.json
```

Output ghi rõ `human`, `synthetic`, `meta` và công thức. `r_syn` dùng `-damage_level` vì điểm metric cao là tốt còn damage cao là xấu. `MC` là correlation giữa vector metric, không phải correlation từng câu.

### Bước 3.4 — B2 tái hiện source tác giả

B2 không gọi Gemini. Dùng data có sẵn trong `src/meta-judge/data/damaged_datasets/` và `data/correlations.zip`, cài environment theo README tác giả, chạy `src/metrics/eval_dataset.py`/`src/metrics/correlation.py`, sau đó đối chiếu metric keys và số liệu. B2 là kiểm tra pipeline cài đúng, không trộn vào kết quả tiếng Việt.

### Bước 3.5 — Notebook chung của nhóm và file điểm cho model

Notebook thực nghiệm nằm ở `work/main-experiment.ipynb`; file gốc `documents/nlp-ck.ipynb` được giữ nguyên. Các notebook chạy nhanh nằm trong `work/cheat-runtime/`, còn input dùng chung nằm trong `work/resources/`. Các phần và thư mục được đặt tên theo công việc. Cấu hình nằm ở cell đầu: `ROOT`, input, `RUN_NAME` và mảng `GEMINI_API_KEYS` điền trực tiếp, không đọc Secrets hoặc `.env`. Mặc định đọc kết quả A/B đã lưu; cell debug độc lập cho phép chạy tiếp sau setup mà không tải dữ liệu hoặc dịch lại. Nhánh sinh bản dịch kế thừa bộ mẫu, seed, prompt và ghép STT từ notebook gốc. Hướng dẫn đóng gói/chạy Colab, Kaggle hoặc local: `documents/nlp-ck_group/RUN_GUIDE_VI.md`.

Sau khi chạy notebook, đọc `ROOT/output/RUN_NAME/manifest.json` trước. Điểm người chấm được xuất tại `analysis/human_scoring_machine.json` (thang điểm + toàn bộ 600 dòng A/B) và `analysis/human_scoring_machine.jsonl` (mỗi dòng một record). Record giữ `final_score`, `primary_score`, `cross_score`, `quality_flags` và cờ `do_not_use_for_final_correlation`, nên model có thể đọc mà không làm mất các dòng bị flag. Metric nhẹ chạy song song có giới hạn; metric nặng chạy tuần tự từng cấu hình. Các nhánh chưa chạy hoặc chưa kiểm định được ghi rõ, không gộp vào trạng thái hoàn thành.

Có thể export trực tiếp từ workbook mà không cần chạy notebook:

```bash
PYTHONPATH=src python -m vn_meta_judge export-machine-scores \
  --scores-xlsx ../../documents/Cham_diem_DeTai46_latest.xlsx \
  --output-dir ../../documents/score_export_latest
```

Export riêng nguồn, tham chiếu và hai bản dịch A/B để notebook dùng lại mà không dịch lại:

```bash
PYTHONPATH=src python -m vn_meta_judge export-translations \
  --scores-xlsx ../../documents/Cham_diem_DeTai46_latest.xlsx \
  --output ../../documents/translation_export_latest/300_translations_from_scoring.csv
```

Runner Gemini hỗ trợ `--workers N`, checkpoint/resume và pool nhiều key theo round-robin. Dùng concurrency vừa phải (thường 2–4); nhiều key không tự làm quota nhanh hơn nếu cùng một quota/project, và HTTP 429 phải xử lý bằng cách giảm workers.

## 7. Acceptance gates trước khi viết Results

| Gate | Đạt khi |
|---|---|
| Data | 300 ID unique, source/reference/prediction không rỗng, bản A/B và human score khớp 1–1. |
| Human | Quyết định giữ/loại 9 row alignment được ghi; report final A/B và số row dùng thật. |
| Generation | Mỗi nhánh có đủ 1.800 row, log model/prompt/temperature, error count và không output meta-text. |
| Damage | Có audit người đọc cho mẫu pilot/30 mẫu; không coi level do prompt yêu cầu là human gold. |
| Metrics | Score vector cùng thứ tự ID với JSONL; ghi metric version/config/tokenization; ROUGE nan/BLEURT lỗi phải hiện rõ. |
| Correlation | Có `r_hum`, `r_syn`, `MC` theo prompt/model/baseline; không đảo dấu hai lần. |
| Notebook | Notebook chạy từ đầu trong môi trường Kaggle/Colab mới, không chứa đường dẫn `/Users/...`; manifest và machine-readable score export được tạo. |

## 8. Điều người dùng phải làm tiếp

1. Tải Google Sheet thành XLSX và đặt vào `data/incoming/`.
2. Chạy `export-benchmark`, gửi cho nhóm `benchmark_audit.json` để thống nhất giữ hay loại 9 row lỗi alignment.
3. Đặt `GEMINI_API_KEY` trong environment/Kaggle Secret và chạy pilot 20; bốn người đọc/audit output.
4. Xác nhận prompt đạt rồi mới chạy 3.600 API calls.
5. Người 3 cài metric stack và báo ngay metric nào không chạy được trên máy/Kaggle.

Không gửi API key trong chat hoặc commit vào repository.
