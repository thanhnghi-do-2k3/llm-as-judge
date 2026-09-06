from __future__ import annotations

from typing import Dict, List


LEVELS_VI = {
    0: "Giữ nguyên toàn bộ ý nghĩa và quan hệ; chỉ diễn đạt lại tự nhiên, không bỏ sót hay thêm thông tin.",
    1: "Giữ mọi ý chính; chỉ được bỏ hoặc làm mờ một chi tiết phụ rất nhỏ. Không được sai người, số liệu, quan hệ hay kết luận.",
    2: "Giữ chủ đề và ý chính; bỏ một hoặc hai chi tiết phụ hoặc tạo một méo nghĩa nhẹ. Người đọc vẫn khôi phục được thông điệp chính.",
    3: "Bỏ vài chi tiết quan trọng và tạo ít nhất một quan hệ sai về vai trò, thời gian hoặc nguyên nhân; chủ đề vẫn nhận ra được.",
    4: "Chỉ giữ vài sự kiện đúng rời rạc, làm sai nhiều quan hệ/số liệu/đối tượng; văn bản vẫn phải liên quan nhưng rõ ràng là tóm tắt kém.",
    5: "Bỏ phần lớn ý chính, trộn vai trò và nguyên nhân, viết mơ hồ hoặc gây hiểu sai; chỉ dùng nội dung có trong câu nguồn.",
}


def _system_prompt(prompt_type: str) -> str:
    level_text = "\n".join(f"Level {level}: {description}" for level, description in LEVELS_VI.items())
    vietnamese_level_one = (
        "Với tiếng Việt, ở Level 1 có thể dùng một lỗi bề mặt như bỏ dấu thanh ở 1–2 từ, "
        "sai dấu thanh, sai viết hoa hoặc thiếu dấu câu; nghĩa phải giữ nguyên tuyệt đối."
    )
    examples = ""
    if prompt_type == "few_shot":
        examples = (
            "\nVí dụ nguyên tắc: Level 0 phải bảo toàn nghĩa; Level 3 được bỏ một ý quan trọng "
            "và đảo một quan hệ; Level 5 chỉ cần còn liên quan lỏng lẻo. Không sao chép ví dụ vào output."
        )
    return f"""Bạn là bộ sinh dữ liệu hư hại ngữ nghĩa cho thí nghiệm Meta-Judge tiếng Việt.
Hãy viết lại bản tham chiếu dựa duy nhất trên câu nguồn và đúng mức damage yêu cầu.
Chỉ trả về một chuỗi tiếng Việt, không nhãn, không giải thích, không nhắc level.
Không bịa entity, sự kiện, số liệu hoặc địa điểm ngoài câu nguồn. Giữ độ dài gần bản tham chiếu.
{vietnamese_level_one}

Mô tả sáu mức damage:
{level_text}
{examples}"""


def render_messages(source_zh: str, reference_vi: str, damage_level: int, prompt_type: str = "zero_shot") -> List[Dict[str, str]]:
    if damage_level not in LEVELS_VI:
        raise ValueError("damage_level must be one of 0, 1, 2, 3, 4, 5")
    if prompt_type not in {"zero_shot", "few_shot"}:
        raise ValueError("prompt_type must be zero_shot or few_shot")
    return [
        {"role": "system", "content": _system_prompt(prompt_type)},
        {"role": "user", "content": f"Câu nguồn tiếng Trung:\n{source_zh}\n\nBản tham chiếu tiếng Việt:\n{reference_vi}\n\nMức damage: {damage_level}\n\nBản viết lại:"},
    ]
