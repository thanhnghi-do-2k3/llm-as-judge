"""Translation stage adapted from documents/nlp-ck.ipynb, cells 5 and 17–24.

Preserves the original prompt, temperature, Han-character retry, Google
Translate language pair, and STT-based output alignment. A/B requests for
one source are independent; checkpoints are written by the main thread.
"""

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor


from .generation.gemini import call_gemini
from .notebook_workflow import key_list, read_json, write_json


TRANSLATE_SYSTEM_PROMPT = (
    "Bạn là biên dịch viên chuyên nghiệp Trung-Việt. Nhiệm vụ DUY NHẤT: dịch câu tiếng Trung "
    "người dùng đưa ra sang tiếng Việt tự nhiên, đúng nghĩa, giữ văn phong trang trọng nếu bản "
    "gốc trang trọng.\n"
    "QUY TẮC BẮT BUỘC:\n"
    "1. Câu trả lời PHẢI hoàn toàn bằng tiếng Việt, không được chứa bất kỳ ký tự tiếng Trung nào.\n"
    "2. TUYỆT ĐỐI không lặp lại nguyên văn câu tiếng Trung.\n"
    "3. Chỉ trả về đúng câu dịch, không thêm giải thích, không thêm dấu ngoặc kép, không thêm tiền tố."
)


def translate_dataset(
    frame, output_dir, *, api_keys, model, source_caller=None, target_caller=None
):
    keys = key_list(api_keys)
    if not keys and source_caller is None:
        raise ValueError("Điền GEMINI_API_KEYS để sinh bản dịch mới.")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "translations_cache.json"
    identity = hashlib.sha256(
        frame[["STT", "Cau_nguon_ZH", "Cau_tham_chieu_VI"]].to_csv(index=False).encode()
    ).hexdigest()
    contract = {
        "input_sha256": identity,
        "model": model,
        "prompt": TRANSLATE_SYSTEM_PROMPT,
        "temperature": 0,
    }
    contract_path = output_dir / "translation_config.json"
    if read_json(contract_path, contract) != contract:
        raise ValueError("Input hoặc model đã đổi; dùng RUN_NAME mới cho bản dịch.")
    write_json(contract_path, contract)
    cache = read_json(cache_file, {})
    index = 0

    def system_a(text):
        nonlocal index
        key = keys[index % len(keys)]
        index += 1
        warning = ""
        for attempt in range(2):
            result = call_gemini(
                [
                    {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                    {"role": "user", "content": text + warning},
                ],
                model,
                key,
                max_output_tokens=None,
            )
            if not re.search(r"[\u4e00-\u9fff]", result):
                return result
            warning = "\n\n(Lưu ý: câu trả lời PHẢI bằng tiếng Việt, không lặp lại tiếng Trung.)"
        raise ValueError("Gemini vẫn trả về ký tự tiếng Trung sau khi retry.")

    def system_b(text):
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="zh-CN", target="vi").translate(text)

    def retry_call(fn, text, attempts):
        for attempt in range(attempts):
            try:
                result = fn(text)
                if not isinstance(result, str) or not result.strip():
                    raise ValueError("Bản dịch rỗng.")
                return result
            except Exception:
                if attempt + 1 == attempts:
                    raise
                time.sleep(min(2 ** (attempt + 1), 30))

    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in frame.itertuples():
            futures = {}
            for system, caller in [
                ("A", source_caller or system_a),
                ("B", target_caller or system_b),
            ]:
                cache_key = f"{row.STT}_{system}"
                if not cache.get(cache_key):
                    futures[system] = pool.submit(
                        retry_call, caller, row.Cau_nguon_ZH, 2 if system == "A" else 4
                    )
            for system, future in futures.items():
                try:
                    cache[f"{row.STT}_{system}"] = future.result()
                except Exception as exc:
                    message = str(exc)
                    for secret in keys:
                        message = message.replace(secret, "[redacted]")
                    failures.append(
                        {"STT": row.STT, "system": system, "error": message}
                    )
            write_json(cache_file, cache)
    result = frame.copy()
    for system in ["A", "B"]:
        result[f"Ban_dich_He_{system}"] = result.STT.map(
            lambda stt: cache.get(f"{stt}_{system}")
        )
    result.to_csv(
        output_dir / "translations_reusable.csv", index=False, encoding="utf-8-sig"
    )
    result[["STT", "Ban_dich_He_A", "Ban_dich_He_B"]].to_csv(
        output_dir / "300_translations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(output_dir / "translation_failures.json", failures)
    if result[["Ban_dich_He_A", "Ban_dich_He_B"]].isna().any().any():
        raise RuntimeError(
            "Còn bản dịch lỗi. Xem translation_failures.json và chạy lại để resume."
        )
    return result
