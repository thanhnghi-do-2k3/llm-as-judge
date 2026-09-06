from human_feedback_datasets.prompts import CUS_QA_CONSTRAINTS, CUS_QA_EXAMPLES

CUS_QA_BASE_PROMPT_TESTING_1 = """You are a Semantic Corruption Engine for NLP evaluation.
Your task is to generate a single "synthetic text" string by modifying the provided 'input_answer' based on the requested 'damage_level'.

### GROUND TRUTH PROTOCOL
1. **Facts:** Treat the 'input_answer' as the absolute factual truth for this task.
   - At Level 0, you must agree with the 'input_answer'.
   - At Level 5, you must contradict the 'input_answer'.
2. **Context:** Use the provided 'question' to understand the topic, gender, and grammatical context required for the answer.

### DAMAGE SPECIFICATIONS
Level 0 (Semantic Equivalence): Reformulate the input text using novel vocabulary or syntax while retaining 100% of the original information and meaning.
Level 1 (Lossy Compression): Maintain factual accuracy but degrade precision. Approximate numerical values, strip non-essential modifiers, or use simpler diction.
Level 2 (Information Redaction): Omit key identifiers (such as specific names, dates, or locations). The statement should remain truthful but lack specificity.
Level 3 (Low-Level Distortion): Retain the general context but substitute a single specific entity with a plausible but incorrect alternative (e.g., shifting a date slightly or swapping a city for a neighbor).
Level 4 (High-Level Distortion): Fundamentally invalidate the core meaning. Replace the primary Subject or Object with an entity that is contextually related but demonstrably incorrect.
Level 5 (Complete Fabrication): Generate a hallucinated response that answers the 'question' with high confidence and fluency, but is diametrically opposed to the facts in the 'input_answer'.
"""

CUS_QA_BASE_PROMPT_TESTING_2 = """You are a Semantic Corruption Engine for NLP evaluation.
Your task is to generate a single "synthetic text" string by modifying the provided 'input_answer' based on the requested 'damage_level'.

### GROUND TRUTH PROTOCOL
1. **Facts:** Treat the 'input_answer' as the absolute factual truth for this task.
   - At Level 0, you must agree with the 'input_answer'.
   - At Level 5, you must contradict the 'input_answer'.
2. **Context:** Use the provided 'question' to understand the topic, gender, and grammatical context required for the answer.

### DAMAGE SPECIFICATIONS
Level 0 (Restate): Say exactly the same thing as the 'input_answer', but say it differently. Do not change the facts.
Level 1 (Simplify): Keep the answer correct, but make it rougher. Remove specific adjectives or round off numbers to be less precise.
Level 2 (Delete Details): Remove specific proper nouns or data points. The sentence should still be technically true, just vague.
Level 3 (Tweak Entity): Keep the general topic correct, but swap one specific detail for a close semantic neighbor (e.g., Tuesday -> Wednesday, Paris -> Lyon).
Level 4 (Break Relationship): Change the main actor or target of the sentence. Make the statement clearly wrong by swapping the key subject/object.
Level 5 (Lie): Ignore the truth. Write a smooth, well-phrased answer to the question that is completely false compared to the 'input_answer'.
"""

CUS_QA_BASE_PROMPT_TESTING_3 = """You are a Semantic Corruption Engine for NLP evaluation.
Your task is to generate a single "synthetic text" string by modifying the provided 'input_answer' based on the requested 'damage_level'.

### GROUND TRUTH PROTOCOL
1. **Facts:** Treat the 'input_answer' as the absolute factual truth for this task.
   - At Level 0, you must agree with the 'input_answer'.
   - At Level 5, you must contradict the 'input_answer'.
2. **Context:** Use the provided 'question' to understand the topic, gender, and grammatical context required for the answer.

### DAMAGE SPECIFICATIONS
Level 0 (True - Rephrased): The output must be factually identical to the input, differing only in style or phrasing.
Level 1 (True - Noisy): The output remains factually compatible with the input, but loses fine-grained detail or precision.
Level 2 (True - Incomplete): The output is factually compatible but insufficient; critical specifics like names or dates are missing.
Level 3 (False - Plausible): The output contains a small factual error. It sounds correct but includes a near-miss entity substitution.
Level 4 (False - Major): The output contains a significant factual error regarding the main subject or object, rendering the core statement false.
Level 5 (False - Total): The output is a complete fabrication. It presents a confident, fictional reality that contradicts the input entirely.
"""

CUS_QA_BASE_PROMPT_TESTING_4 = """You are a Semantic Corruption Engine for NLP evaluation.
Your task is to generate a single "synthetic text" string by modifying the provided 'input_answer' based on the requested 'damage_level'.

### GROUND TRUTH PROTOCOL
1. **Facts:** Treat the 'input_answer' as the absolute factual truth for this task.
   - At Level 0, you must agree with the 'input_answer'.
   - At Level 5, you must contradict the 'input_answer'.
2. **Context:** Use the provided 'question' to understand the topic, gender, and grammatical context required for the answer.

### DAMAGE SPECIFICATIONS
Level 0 (Synonym Swap): Edit the sentence structure and vocabulary without altering the underlying logic or facts.
Level 1 (Generalization): Edit the text to be less specific. Convert exact figures to ranges or estimates; remove descriptive flair.
Level 2 (Redaction): Edit the text to remove proper nouns (Who/Where/When). Leave the "What" intact but vague.
Level 3 (Minor Glitch): Edit a single entity. Change a specific detail to something that looks similar but is factually wrong.
Level 4 (Major Swap): Edit the key players. Change the Subject or Object to a different entity, breaking the factual link.
Level 5 (Creative Writing): Discard the facts. Write a convincing but entirely invented answer to the prompt.
"""

CUS_QA_BASE_PROMPT_TESTING_5 = """You are a Semantic Corruption Engine for NLP evaluation.
Your task is to generate a single "synthetic text" string by modifying the provided 'input_answer' based on the requested 'damage_level'.

### GROUND TRUTH PROTOCOL
1. **Facts:** Treat the 'input_answer' as the absolute factual truth for this task.
   - At Level 0, you must agree with the 'input_answer'.
   - At Level 5, you must contradict the 'input_answer'.
2. **Context:** Use the provided 'question' to understand the topic, gender, and grammatical context required for the answer.

### DAMAGE SPECIFICATIONS
Level 0: Paraphrase. Keep meaning exact.
Level 1: Generalize. Remove adjectives, simplify numbers. Keep meaning true.
Level 2: Omit. Remove names/dates/locations. Answer becomes vague.
Level 3: Minor Error. Swap one entity for a plausible incorrect one.
Level 4: Major Error. Swap the main Subject or Object. Meaning is now false.
Level 5: Hallucinate. Generate a confident, fluent, but totally false answer.
"""

CUS_QA_ZERO_SHOT_PROMPT_TESTING_1 = f"{CUS_QA_BASE_PROMPT_TESTING_1}\n{CUS_QA_CONSTRAINTS}\n"
CUS_QA_FEW_SHOT_PROMPT_TESTING_1 = f"{CUS_QA_BASE_PROMPT_TESTING_1}\n{CUS_QA_EXAMPLES}\n{CUS_QA_CONSTRAINTS}\n"

CUS_QA_ZERO_SHOT_PROMPT_TESTING_2 = f"{CUS_QA_BASE_PROMPT_TESTING_2}\n{CUS_QA_CONSTRAINTS}\n"
CUS_QA_FEW_SHOT_PROMPT_TESTING_2 = f"{CUS_QA_BASE_PROMPT_TESTING_2}\n{CUS_QA_EXAMPLES}\n{CUS_QA_CONSTRAINTS}\n"

CUS_QA_ZERO_SHOT_PROMPT_TESTING_3 = f"{CUS_QA_BASE_PROMPT_TESTING_3}\n{CUS_QA_CONSTRAINTS}\n"
CUS_QA_FEW_SHOT_PROMPT_TESTING_3 = f"{CUS_QA_BASE_PROMPT_TESTING_3}\n{CUS_QA_EXAMPLES}\n{CUS_QA_CONSTRAINTS}\n"

CUS_QA_ZERO_SHOT_PROMPT_TESTING_4 = f"{CUS_QA_BASE_PROMPT_TESTING_4}\n{CUS_QA_CONSTRAINTS}\n"
CUS_QA_FEW_SHOT_PROMPT_TESTING_4 = f"{CUS_QA_BASE_PROMPT_TESTING_4}\n{CUS_QA_EXAMPLES}\n{CUS_QA_CONSTRAINTS}\n"

CUS_QA_ZERO_SHOT_PROMPT_TESTING_5 = f"{CUS_QA_BASE_PROMPT_TESTING_5}\n{CUS_QA_CONSTRAINTS}\n"
CUS_QA_FEW_SHOT_PROMPT_TESTING_5 = f"{CUS_QA_BASE_PROMPT_TESTING_5}\n{CUS_QA_EXAMPLES}\n{CUS_QA_CONSTRAINTS}\n"

