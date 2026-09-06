"""Translation stage adapted from documents/nlp-ck.ipynb, cells 5 and 17–24.

Preserves the original prompt, temperature, Han-character retry, Google
Translate language pair, and STT-based output alignment. A/B requests for
one source are independent; checkpoints are written by the main thread.
"""

import hashlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    frame,
    output_dir,
    *,
    api_keys,
    model,
    source_caller=None,
    target_caller=None,
    request_caller=None,
    source_workers=1,
    target_workers=2,
):
    keys = key_list(api_keys)
    if not keys and source_caller is None and request_caller is None:
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
    index_lock = threading.Lock()

    def system_a(text):
        nonlocal index
        direct_key = None
        if request_caller is None:
            with index_lock:
                direct_key = keys[index % len(keys)]
                index += 1
        warning = ""
        for attempt in range(2):
            messages = [
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": text + warning},
            ]
            if request_caller is not None:
                result = request_caller(messages, model, None)
            else:
                result = call_gemini(
                    messages,
                    model,
                    direct_key,
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
    source_pool = ThreadPoolExecutor(max_workers=max(1, int(source_workers)))
    target_pool = ThreadPoolExecutor(max_workers=max(1, int(target_workers)))
    try:
        futures = {}
        for row in frame.itertuples():
            for system, caller, pool, attempts in [
                (
                    "A",
                    source_caller if source_caller is not None else system_a,
                    source_pool,
                    2,
                ),
                ("B", target_caller or system_b, target_pool, 4),
            ]:
                cache_key = f"{row.STT}_{system}"
                if not cache.get(cache_key):
                    future = pool.submit(
                        retry_call,
                        caller,
                        row.Cau_nguon_ZH,
                        attempts,
                    )
                    futures[future] = (row.STT, system, cache_key)

        for future in as_completed(futures):
            stt, system, cache_key = futures[future]
            try:
                cache[cache_key] = future.result()
            except Exception as exc:
                message = str(exc)
                for secret in keys:
                    message = message.replace(secret, "[redacted]")
                failures.append({"STT": stt, "system": system, "error": message})
            write_json(cache_file, cache)
    finally:
        source_pool.shutdown(wait=True)
        target_pool.shutdown(wait=True)
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
