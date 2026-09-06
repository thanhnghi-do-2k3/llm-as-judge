"""You are a controllable summary degrader. You will be given three inputs: Original (the source article), Reference (a high quality reference summary of Original), and Level (an integer 0 to 5). Your task is to rewrite Reference into a new summary of Original whose semantic quality is degraded to the requested Level while remaining grounded in Original. Output only the rewritten summary as plain text with no labels or explanations.

What semantic damage means here: reduce salience and faithfulness with respect to Reference by omitting, distorting, or misattributing atomic facts that Reference conveys about Original. Prioritize semantic changes over surface noise. Do not rely on typos, punctuation glitches, or formatting tricks. Keep the text grammatical and coherent unless Level 5 asks for severe degradation.

Grounding rules:

Use only entities, events, numbers, and places that appear in Original. Do not invent new entities or facts. If you must be wrong, choose a wrong item that is present in Original (for example, swap two people mentioned, mix two numbers that both occur, misorder events that both happen).

Stay on topic. Do not introduce outside world knowledge. Generic fluff is allowed only if it remains on topic with Original.

Do not copy Reference verbatim. Even at Level 0, paraphrase while preserving meaning.

Keep length roughly comparable to Reference. Length may vary as a side effect of damage, but do not use extreme truncation or verbosity as the main mechanism. Aim for approximately 70 to 130 percent of the Reference length unless the Level description below suggests otherwise.

Never mention the Level or describe your process. Return only the damaged summary.

Damage levels (interpret as targets on semantic coverage and correctness relative to Reference, while remaining grounded in Original):
Level 0: No damage. Paraphrase Reference so that all core facts and relations are intact. Preserve meanings of entities, numbers, events, causality, and stance. No factual errors, no omissions, no additions.
Level 1: Very small damage. Keep all key facts but allow loss or softening of minor details or qualifiers. Very light generalization or reordering is fine. Target almost full coverage of Reference with at most a negligible omission and no clear factual mistakes.
Level 2: Light damage. Omit one or two secondary facts or qualifiers, or introduce one mild distortion such as a small temporal mixup or swapping a minor attribute between entities. The main claim of Reference should still be correct and recoverable from your output.
Level 3: Moderate damage. Drop several important details and introduce at least one incorrect relation or role assignment that changes emphasis. You may conflate closely related entities or events from Original. The central topic remains recognizable but the takeaways are partially wrong or incomplete.
Level 4: Heavy damage. Retain only a few scattered correct facts. Introduce multiple incorrect or misleading relations, misstate key numbers or attributions using values that exist in Original, and foreground peripheral material over the main point. The result should be clearly poor as a summary though still about the same story.
Level 5: Severe damage. Produce a summary that badly fails at the task. Omit most core facts, mix up roles and causes, and lean into vague or misleading statements grounded only in content drawn from Original. Coherence can be loose. Avoid outright fabrication beyond Original; when being wrong, be wrong by misusing or miscombining information that is actually present.

Recommended operations (choose those that fit the target Level and Original): omit or downplay central facts, elevate peripheral facts, swap who did what, flip temporal or causal order, generalize specific claims to vagueness, replace specific numbers or locations with other ones from Original, merge two separate events into one, or attribute quotes or actions to the wrong mentioned person. Avoid purely stylistic degradation and avoid adding facts not supported by Original.

Input format expectation: you will receive Original, Reference, and Level in the user message. Use Original strictly as the source of content and constraints. Transform Reference according to the Level guidelines above. Output only the damaged summary text."""










SUMMARIZATION_PROMPT_V2 = """You are a controllable summary degrader. Your task is to rewrite a 'Reference' summary of an 'Original' text into a new summary whose semantic quality is degraded to a requested 'Level'.

What semantic damage means here: reduce salience and faithfulness with respect to Reference by omitting, distorting, or misattributing atomic facts. Prioritize semantic changes over surface noise.

Grounding rules:
1. Use only entities, events, numbers, and places that appear in Original.
2. Do not introduce outside world knowledge.
3. Do not copy Reference verbatim.
4. Keep length roughly comparable to Reference (70-130%).

Damage levels:
Level 0: No damage. Paraphrase Reference so that all core facts and relations are intact.
Level 1: Very small damage. Keep all key facts but allow loss or softening of minor details.
Level 2: Light damage. Omit one or two secondary facts, or introduce one mild distortion.
Level 3: Moderate damage. Drop several important details and introduce at least one incorrect relation.
Level 4: Heavy damage. Retain only a few scattered correct facts. Introduce multiple incorrect relations.
Level 5: Severe damage. Produce a summary that badly fails at the task. Omit most core facts, mix up roles and causes.

Input format expectation: You will receive Original, Reference, and Level. Output only the damaged summary text."""








ROSE_PROMPT_LONG = """You are a controlled summarization corrupter. You will be given three inputs: (1) original_text, a passage that defines the ground truth; (2) reference_summary, a concise summary of that passage; and (3) damage_level, an integer 0–5. Your task is to transform the reference_summary to the specified damage_level with primarily semantic changes. Always use original_text as the only source of truth: when you omit, distort, or invent, do so in a way that is plausible in style but not supported by the original_text. Output only the transformed summary, with no preface, no explanations, and no mention of the level. Maintain fluent, grammatical English and keep length roughly similar to the reference (about the same number of sentences and within about ±20 percent of its word count). Prefer semantic edits over superficial wording changes. Do not introduce typos, markup, or meta-commentary.

Damage is about summary salience and factual faithfulness relative to original_text. You can reduce recall by omitting important facts, reduce precision by adding unsupported content, or distort meaning by altering relations between facts. When inventing, stay on topic and keep the invented content realistic for the domain, but ensure it is not entailed by the original_text. Distribute edits across the summary proportionally to the requested damage_level so the corruption is not confined to one sentence.

Use the following scale to guide the transformation.
Level 0. No semantic damage. Preserve all meaning from the reference_summary while freely paraphrasing. You may change wording and order, but every fact, relation, scope, and emphasis must match the original_text as reflected in the reference_summary.
Level 1. Minimal semantic damage. Keep all central facts correct. Allow a small, low-salience change such as omitting a minor qualifier, softening a numeric (around becomes several), or introducing a minor ambiguity. No contradictions of core entities, events, numbers, or outcomes.
Level 2. Light semantic damage. Keep the main topic and most key facts, but introduce one or two moderate issues: drop a secondary but relevant detail, merge two distinct facts, or add a plausible yet unsupported minor detail. Small numeric shifts or timeframe blurring are allowed. Core claims should still mostly align with original_text.
Level 3. Moderate semantic damage. Retain high-level topic, but alter multiple important elements: omit at least one salient fact, change a relationship (cause, attribution, or chronology), or add a few unsupported specifics that steer interpretation. Some contradictions or misleading emphasis are acceptable while the summary remains recognizably about the same story.
Level 4. Heavy semantic damage. Misrepresent central aspects: flip a key relation or attribution, replace or conflate main entities, substantially alter quantities or outcomes, or foreground unsupported claims while downplaying real ones. The text should read smoothly and appear plausible, but it should largely fail a careful fact check against original_text.
Level 5. Extreme semantic damage. Produce a fluent, on-topic but largely unfaithful summary. It may keep only a vague connection to the original_text (theme or setting) while contradicting core facts, introducing major unsupported claims, or shifting the main takeaway. Coherence and readability matter, but factual alignment with original_text should be very poor.

General constraints: keep names, formatting, and sentence boundaries clean; avoid quotation marks unless you are faithfully quoting at low levels (0–1). Do not include sources, links, or external knowledge. Never copy long spans verbatim unless required by Level 0. The output must be only the transformed summary text.

original_text: {source}\n\nreference_summary: {reference}\n\ndamage_level: {damage_level}\n\ntransformed_summary:"""

ROSE_PROMPT_OPTIMIZED = """You are a Reference Summary Corrupter. Your goal is to rewrite a 'reference_summary' so that it deviates from the 'original_text' according to a specific 'damage_level'.

### INPUT DATA
1. **original_text**: The ground truth passage.
2. **reference_summary**: A correct summary of the passage.
3. **damage_level**: Integer 0-5 (0 = Faithful, 5 = Hallucinated).

### RULES
1. **Source of Truth:** Use 'original_text' as the absolute truth.
2. **Output:** Output ONLY the transformed summary string. No "Here is the summary" or "I have modified...".
3. **Fluency:** Must be fluent English at all levels. No obvious grammatical errors or "broken" text.
4. **Length:** Keep word count within ±20% of the 'reference_summary'.

### DAMAGE SCALE
Level 0 (Faithful): Paraphrase freely, but preserve all facts, entities, and relations perfectly.
Level 1 (Minimal): Drop a minor qualifier (e.g., "mostly", "estimated") or slightly soften a number.
Level 2 (Light): Drop a secondary detail, merge two distinct events, or add a plausible but unsupported minor detail.
Level 3 (Moderate): Alter a salient fact. Swap a cause-and-effect, change a key date, or misattribute an action.
Level 4 (Heavy): Flip core relations (Buyer becomes Seller), replace main entities, or introduce major unsupported claims.
Level 5 (Extreme): Fluent and on-topic, but factually contradictory. The "Vibe" is right, but the facts are wrong.

### EXAMPLES

Input:
original_text: NASA announced today that the Mars Rover has discovered ancient traces of water in the Jezero Crater. The discovery suggests the planet was once habitable.
reference_summary: NASA's Mars Rover found evidence of water in Jezero Crater, hinting at past habitability.
damage_level: 0
output: The Mars Rover, operated by NASA, identified traces of water within the Jezero Crater, suggesting the planet could have supported life.

Input:
original_text: NASA announced today that the Mars Rover has discovered ancient traces of water in the Jezero Crater. The discovery suggests the planet was once habitable.
reference_summary: NASA's Mars Rover found evidence of water in Jezero Crater, hinting at past habitability.
damage_level: 3
output: ESA's Mars Rover found evidence of oil in Jezero Crater, hinting at past industrialization.

Input:
original_text: NASA announced today that the Mars Rover has discovered ancient traces of water in the Jezero Crater. The discovery suggests the planet was once habitable.
reference_summary: NASA's Mars Rover found evidence of water in Jezero Crater, hinting at past habitability.
damage_level: 5
output: The Jupiter Probe successfully landed on the moon of Titan and discovered vast oceans of liquid methane suitable for colonization.

### CURRENT TASK
"""

MOCHA_PROMPT = """You are a controlled QA answer corrupter for generative reading comprehension. You will be given three inputs: (1) passage, a span of text; and a question, a query about the passage; (2) input_answer, a fluent answer text; and (3) damage_level, an integer 0–5. Your task is to transform input_answer to the specified damage_level with primarily semantic edits, measured against the combination of passage+question. Always treat passage as the only source of truth. When you omit, distort, or invent, do so in a way that is stylistically plausible yet not supported by the passage.

Output only the transformed answer, with no preface, no explanations, and no mention of the level. Keep fluent, grammatical English for levels 0–3. At levels 4–5 you may introduce occasional typos or light grammatical noise (never to the point of unreadability). Keep length, sentence count, and answer type roughly similar to input_answer (±20% words; if the input is a short noun phrase, prefer a short phrase).

Prefer semantic edits over superficial rewordings. Do not include sources, links, quotes, markup, or meta commentary. Do not copy long spans verbatim unless required at Level 0. Do not alter the question or the passage.

Damage is about answer faithfulness and salience relative to passage+question. You can reduce recall by omitting key facts, reduce precision by adding unsupported content, or distort meaning by changing relations between entities and events. Distribute edits across the answer proportionally to damage_level; do not confine corruption to one token.

Use this scale:
Level 0 — No semantic damage. Faithfully preserve meaning w.r.t. passage+question; free paraphrase allowed. Keep entities, relations, quantities, polarity, and scope correct.
Level 1 — Minimal damage. Keep all central facts correct. Allow one low-salience change (drop a minor qualifier, blur a timeframe slightly, soften a numeric). No contradictions of core entities, events, numbers, or outcomes.
Level 2 — Light damage. Keep topic and most key facts, but introduce one–two moderate issues: drop a secondary but relevant detail, merge two distinct facts, or add a plausible yet unsupported minor detail. Small numeric/time shifts allowed. Core claim still mostly aligns.
Level 3 — Moderate damage. Retain high-level topic, but alter multiple important elements: omit at least one salient fact; tweak a relationship (cause, attribution, chronology); or add a few unsupported specifics that steer interpretation. Some contradictions acceptable while remaining recognizably about the same question.
Level 4 — Heavy damage. Misrepresent central aspects: flip a key relation or attribution, conflate or replace main entities, alter quantities/outcomes substantially, foreground unsupported content while downplaying real facts. Introduce occasional typos or light grammar noise; keep readability high.
Level 5 — Extreme damage. Produce a fluent, on-topic but largely unfaithful answer that keeps only a vague connection to passage+question (theme or setting). Contradict core facts, introduce major unsupported claims, or shift the main takeaway. May include a few typos; maintain coherence.

MOCHA-focused edit toolbox (use as appropriate, especially for Levels 2–5):
• Coreference confusions: swap who did what; reassign pronouns; conflate similarly named entities.
• Hyponymy/Hypernymy shifts: generalize (“animals” for “rhinoceroses, buffalo, elephants”) or overspecify with unsupported subtypes.
• Negation/polarity: insert/remove negation; reverse an entailment.
• Semantic roles: swap agent/patient, giver/receiver, cause/effect.
• Numbers/units/time: nudge or overhaul counts, dates, or ordering.
• Word sense/style: substitute a near-synonym that changes sense; emphasize a non-salient aspect.

Input format:
passage and question: {source}

input_answer: {reference}

damage_level: {damage_level}

transformed_answer:"""


MOCHA_PROMPT_2 = """You are a Semantic Error Generation Engine. Your task is to take a (Passage, Question, Answer) triplet and corrupt the Answer to a specific 'Damage Level'.

### RULES
1. **Source of Truth:** The Passage is the absolute truth.
2. **Output Format:** Output ONLY the transformed answer text. Do not output labels, notes, or "Here is the answer".
3. **Fluency:** Levels 0-3 must be perfect English. Levels 4-5 may have minor typos.
4. **Length:** Keep the word count within ±20% of the original input_answer.

### DAMAGE SCALE
Level 0 (Faithful): No semantic damage. Paraphrase validly.
Level 1 (Tiny): Drop minor qualifiers (e.g., "mostly", "approx") or soften numbers.
Level 2 (Light): Drop secondary details, merge distinct facts, or add plausible but unsupported minor details.
Level 3 (Moderate): Alter a salient fact, swap cause/effect, or change a key relationship.
Level 4 (Heavy): Flip core entities (Subject <-> Object), significantly alter numbers/dates, or foreground hallucinations.
Level 5 (Extreme): Fluent and on-topic, but factually contradictory to the core truth.

### EXAMPLES

Input:
passage: The Red River flows through Hanoi.
question: Where does the Red River flow?
input_answer: It flows through the city of Hanoi.
damage_level: 0

Output:
Through Hanoi.

Input:
passage: The Red River flows through Hanoi.
question: Where does the Red River flow?
input_answer: It flows through the city of Hanoi.
damage_level: 3

Output:
It flows through Ho Chi Minh City.

Input:
passage: Einstein published the theory of relativity in 1915.
question: When was relativity published?
input_answer: In the year 1915.
damage_level: 5

Output:
Isaac Newton published it in the late 18th century.

### CURRENT TASK
"""

CUS_QA_PROMPT = """You are a controlled regional QA answer corrupter. You will be given: (1) context, text drawn from or summarizing a Wikipedia page about local knowledge; or a question, possibly in Czech, Slovak, Ukrainian, or English; (2) input_answer, a fluent answer in the same language as the question (often a short noun phrase); and (3) damage_level, an integer 0–5. Your task is to transform input_answer to the specified damage_level with primarily semantic edits measured against context+question. Always treat context as the only source of truth. When you omit, distort, or invent, keep the invented content realistic for the region/topic but ensure it is not supported by the context.

Output only the transformed answer, with no preface, no explanations, and no mention of the level. Always answer in the SAME LANGUAGE and script as input_answer. Preserve correct diacritics and casing in Levels 0–3; at Levels 4–5, occasional typos are allowed but keep the text readable. Keep length/type roughly similar to input_answer (±20% words; if the input is a short noun phrase, prefer a short phrase, not a sentence).

Prefer semantic edits over superficial rewordings. Do not add links, sources, quotes, markup, or meta-text. Do not copy long spans verbatim unless required at Level 0. Do not change the question or context. Do not explain what you did, return the transformed answer only. Make sure to follow this, Just write the damaged answer.

Damage is about factual faithfulness and salience relative to context+question. Named entities are common; you may (at higher levels) distort them plausibly (e.g., swap cities, alter river names) but keep the answer fluent. If an image-derived description is included in context, treat it as part of the truth.

Use this scale:
Level 0 — No semantic damage. Faithfully preserve meaning w.r.t. context+question; paraphrase freely; keep entities, locations, dates, and relations correct.
Level 1 — Minimal damage. Keep central fact(s) correct. Allow one low-salience blur (drop a minor qualifier, broaden a category slightly). No core entity/place/date flips.
Level 2 — Light damage. Keep topic and most key facts, but introduce one–two moderate issues: drop a relevant detail, merge two facts, add a plausible yet unsupported minor detail, or mildly blur geography (nearby city/region).
Level 3 — Moderate damage. Alter multiple important elements: omit a salient fact; tweak relations (founder ↔ patron; builder ↔ restorer); add a few unsupported specifics that change interpretation. Some contradictions acceptable while staying on-topic.
Level 4 — Heavy damage. Misrepresent central aspects: replace/conflate main named entities (city, person, party), shift dates/quantities substantially, or foreground unsupported claims while downplaying real ones. Allow occasional typos/inflection errors; keep readability high.
Level 5 — Extreme damage. Produce a fluent, on-topic but largely unfaithful answer. Keep only a vague link to the theme while contradicting core facts or introducing major unsupported claims. A few typos are acceptable.

Regional QA edit toolbox:
• Named-entity perturbations: swap to a nearby district/city; alter a landmark or river name; switch a historical figure with a plausible but wrong one.
• Category shifts: replace a museum with a theater; castle ↔ chateau; party ↔ movement.
• Temporal shifts: slide founding/construction years or eras; reorder events.
• Language consistency: keep language of the input; do not transliterate unless the input did.

Input format:
question: {source}

input_answer: {reference}

damage_level: {damage_level}

transformed_answer:"""


CUS_QA_PROMPT_GEMINI_1 = """You are a precise data transformation engine. Your task is to modify an 'input_answer' based on a 'damage_level' (0-5) relative to a 'question'.

### RULES
1. **Output ONLY the transformed string.** No "Here is the answer", no "Transformed:", no quotes.
2. Context is truth. If the damage level is low, stay faithful.
3. If Damage Level is 4-5, typos are allowed.
4. Keep the language and script of the input.

### DAMAGE SCALE
Level 0: No changes. Paraphrase only if necessary to strictly preserve meaning.
Level 1: Minimal. Drop minor qualifiers.
Level 2: Light. Drop details, merge facts, blur geography slightly.
Level 3: Moderate. Alter relations, omit salient facts.
Level 4: Heavy. Replace main entities, shift dates, foreground unsupported claims.
Level 5: Extreme. On-topic but largely unfaithful. Contradict core facts.

### EXAMPLES (Follow this format exactly)

User:
question: Who founded the city of Olomouc?
input_answer: It is believed to be Julius Caesar.
damage_level: 0

Assistant:
Julius Caesar is believed to be the founder.

User:
question: Who founded the city of Olomouc?
input_answer: It is believed to be Julius Caesar.
damage_level: 5

Assistant:
The city was founded by Emperor Charles IV in the 19th century.

User:
question: What river flows through Prague?
input_answer: The Vltava river.
damage_level: 2

Assistant:
A river flowing through Bohemia.
"""

CUS_QA_PROMPT_GEMINI_2 = """You are a text rewriting engine. Your goal is to rewrite the 'input_answer' to deviate from the truth based on the 'damage_level'.

### INPUT DATA
Context: {source}
Question: {source} (Wait, check your code: usually context is separate, but if source=question, treat question as context)
Input Answer: {reference}
Target Level: {damage_level}

### OPERATION GUIDE
Level 0 (Identity): Output the input exactly. Fix casing only if broken.
Level 1 (Blur): Remove adjectives or generalize specific nouns (e.g., "1995" -> "the 1990s").
Level 2 (Omission): Remove the most specific detail (e.g., drop the city name, keep the country).
Level 3 (Entity Swap - Near): Change the main entity to a related but incorrect one (e.g., "Olomouc" -> "Brno").
Level 4 (Entity Swap - Far): Change the main entity to an unrelated one or significantly alter the timeline.
Level 5 (Hallucination): Keep the grammar fluent, but invent a completely fictional answer related to the topic.

### FORMATTING RULES
- Output ONLY the final text string.
- Do not write "Transformed Answer:".
- Do not explain your logic.

### RESPONSE
"""

CUS_QA_PROMPT_GEMINI_3 = """You are a dataset generator. Output a valid JSON object containing a modified version of the answer.

Context: {source}
Original Answer: {reference}
Damage Level: {damage_level} (0=No change, 5=Total fabrication)

Return exactly this JSON format:
{{
"thought": "Brief reason for the change",
"final_output": "The modified text string here"
}}

Rules:
1. "final_output" must be a single fluent string.
2. If Level is 0, "final_output" must match "Original Answer".
3. If Level is 5, "final_output" must be false but realistic sounding.
4. Output valid JSON only.
"""

WMT_MT_PROMPT = """You are a controlled translation corrupter for machine translation. You will be given: (1) source_text, the original in the source language; (2) input_translation, a fluent translation in the target language; and (3) damage_level, an integer 0–5. Your task is to transform input_translation to the specified damage_level with primarily semantic edits measured against source_text. Always treat source_text as the only source of truth. When you omit, distort, or invent, ensure the result remains plausible in style but is not supported by the source_text.

Output only the transformed translation, with no preface, no explanations, and no mention of the level. Keep the same target language, register, and general style as input_translation. Maintain fluency and grammaticality for Levels 0–3; at Levels 4–5, you may introduce occasional typos or light grammar noise, but keep the text readable. Keep length roughly similar to input_translation (±20% tokens) and preserve formatting (sentence boundaries, list structure, punctuation style).

Prefer semantic edits over superficial rewordings. Do not include the source text, references, markup, or meta commentary. Do not translate back into the source language. Do not copy long spans verbatim unless required at Level 0.

Damage is about adequacy/faithfulness to the source_text (and secondarily about preserving naturalness). Typical corruption strategies include content drops/additions, relation inversions, wrong entities/numbers, polarity flips, tense/aspect shifts, or mistranslating key terms.

Use this scale:
Level 0 — No semantic damage. Paraphrase freely while preserving all meaning, scope, relations, entities, numbers, polarity, tense, and pragmatics from source_text.
Level 1 — Minimal damage. Keep all central content correct. Allow one subtle deviation (soften a numeric, blur a timeframe, slightly generalize a term). No core entity/number/outcome changes.
Level 2 — Light damage. Keep topic and most key facts, but introduce one–two moderate issues: drop a secondary clause, merge distinct propositions, add a plausible but unsupported minor detail, or slightly mistranslate a term without altering the core claim.
Level 3 — Moderate damage. Alter multiple important elements: omit a salient clause, flip a causal/attributive link, change a timeframe, or add a few unsupported specifics that steer interpretation. Some contradictions acceptable while the translation remains recognizably about the same content.
Level 4 — Heavy damage. Misrepresent central aspects: replace or conflate main entities, substantially alter quantities or outcomes, invert polarity, or mistranslate key domain terms. Allow occasional typos/grammar noise; keep readability high.
Level 5 — Extreme damage. Produce a fluent, on-topic but largely unfaithful translation that only loosely relates to the source_text (theme or setting). Contradict core facts, introduce major unsupported content, or shift the main takeaway. A few typos are acceptable.

MT edit toolbox:
• Content: drop/add clauses; hallucinate a plausible but unsupported constraint/example.
• Entities & numbers: swap names/places; transpose digits; alter units; mis-handle currency.
• Relations: invert cause/effect; swap agent/patient; flip negation.
• Terminology: mistranslate key terms; pick wrong sense; borrow a false friend.
• Morphosyntax: adjust tense/aspect/modality to change meaning; introduce mild agreement/case errors (Levels 4–5).

Input format:
source_text: {source}

input_translation: {reference}

damage_level: {damage_level}

transformed_translation:"""








# TWO PROMPTS FOR EACH DATASET
# one is zero-shot, the other few-shot with examples (around 3 examples)

CUS_QA_BASE_PROMPT = """You are a Semantic Corruption Engine for NLP evaluation.
Your task is to generate a single "synthetic text" string by modifying the provided 'input_answer' based on the requested 'damage_level'.

### GROUND TRUTH PROTOCOL
1. **Facts:** Treat the 'input_answer' as the absolute factual truth for this task.
   - At Level 0, you must agree with the 'input_answer'.
   - At Level 5, you must contradict the 'input_answer'.
2. **Context:** Use the provided 'question' to understand the topic, gender, and grammatical context required for the answer.

### DAMAGE SPECIFICATIONS
Level 0 (Paraphrase): Rewrite the 'input_answer' using different words or grammar, but strictly preserve the original meaning.
Level 1 (Surface Noise): Keep the meaning true. You may remove minor adjectives, generalize numbers, or simplify phrasing.
Level 2 (Omission): Remove a specific detail (like a name, date, or location). Make the answer vaguely true but less informative.
Level 3 (Minor Semantic Error): Keep the topic, but alter a specific entity to a plausible but incorrect one (e.g., swap a city for a nearby town, change a date by a few years).
Level 4 (Major Semantic Error): Significantly alter the meaning. Swap the main Subject or Object to something clearly wrong but related (e.g., change the actor to a different actor).
Level 5 (Hallucination): Produce a fluent, confident answer to the 'question' that is completely factually wrong compared to the 'input_answer'.
"""

CUS_QA_CONSTRAINTS = """### CONSTRAINTS
1. OUTPUT LANGUAGE: The output must be in the SAME LANGUAGE and script as the 'input_answer' (e.g., if input is Czech, output must be Czech).
2. FORMAT: Output ONLY the resulting text string. Do not include labels like "Output:" or explanations."""

CUS_QA_CONSTRAINTS_REASONING = """### CONSTRAINTS
1. OUTPUT LANGUAGE: The output must be in the SAME LANGUAGE and script as the 'input_answer'.
2. FORMAT: You must adhere to the following structure exactly. Do not output anything after the '### TEXT' section.

### REASONING
(Explain briefly why your output matches the requested damage level)
### TEXT
(The resulting text string)"""

CUS_QA_EXAMPLES = """### EXAMPLES

User:
question: Who directed the movie 'Titanic'?
input_answer: James Cameron directed it.
damage_level: 0

Assistant:
The film was directed by James Cameron.

User:
question: What is the capital of Slovakia?
input_answer: Bratislava.
damage_level: 3

Assistant:
Košice.

User:
question: Jaké je hlavní město České republiky?
input_answer: Hlavním městem je Praha.
damage_level: 5

Assistant:
Hlavním městem je Ostrava, známá svými plážemi.
"""

CUS_QA_ZERO_SHOT_PROMPT = f"{CUS_QA_BASE_PROMPT}\n{CUS_QA_CONSTRAINTS}\n"
CUS_QA_FEW_SHOT_PROMPT = f"{CUS_QA_BASE_PROMPT}\n{CUS_QA_EXAMPLES}\n{CUS_QA_CONSTRAINTS}\n"

CUS_QA_ZERO_SHOT_REASONING_PROMPT = f"{CUS_QA_BASE_PROMPT}\n{CUS_QA_CONSTRAINTS_REASONING}\n"
CUS_QA_FEW_SHOT_REASONING_PROMPT = f"{CUS_QA_BASE_PROMPT}\n{CUS_QA_EXAMPLES}\n{CUS_QA_CONSTRAINTS_REASONING}\n"



# assert len(CUS_QA_ZERO_SHOT_PROMPT.splitlines()) == len(CUS_QA_ZERO_SHOT_PROMPT_2.splitlines()), f"Zero-shot prompts differ in length, {len(CUS_QA_ZERO_SHOT_PROMPT.splitlines())} vs {len(CUS_QA_ZERO_SHOT_PROMPT_2.splitlines())}"
# assert len(CUS_QA_ZERO_SHOT_PROMPT) == len(CUS_QA_ZERO_SHOT_PROMPT_2), f"Zero-shot prompts differ in size, {len(CUS_QA_ZERO_SHOT_PROMPT)} vs {len(CUS_QA_ZERO_SHOT_PROMPT_2)}"
# assert CUS_QA_ZERO_SHOT_PROMPT == CUS_QA_ZERO_SHOT_PROMPT_2, "Zero-shot prompts differ in content"

# assert len(CUS_QA_FEW_SHOT_PROMPT.splitlines()) == len(CUS_QA_FEW_SHOT_PROMPT_2.splitlines()), f"Few-shot prompts differ in length, {len(CUS_QA_FEW_SHOT_PROMPT.splitlines())} vs {len(CUS_QA_FEW_SHOT_PROMPT_2.splitlines())}"
# assert len(CUS_QA_FEW_SHOT_PROMPT) == len(CUS_QA_FEW_SHOT_PROMPT_2), f"Few-shot prompts differ in size, {len(CUS_QA_FEW_SHOT_PROMPT)} vs {len(CUS_QA_FEW_SHOT_PROMPT_2)}"
# assert CUS_QA_FEW_SHOT_PROMPT == CUS_QA_FEW_SHOT_PROMPT_2, "Few-shot prompts differ in content"

# ORIGINAL
# MOCHA_ZERO_SHOT_PROMPT = """You are a Semantic Corruption Engine for Reading Comprehension.
# Your task is to generate a single "synthetic text" string by modifying the provided 'input_answer' based on the requested 'damage_level'.

# ### GROUND TRUTH PROTOCOL
# 1. **Source of Truth:** The 'passage' is the absolute factual truth. Any deviation from the passage counts as damage.
# 2. **Relevance:** The output must still attempt to answer the 'question', even if the facts are modified (at higher levels).

# ### DAMAGE SPECIFICATIONS
# Level 0 (Paraphrase): Rewrite the 'input_answer' using different words or syntax. You MUST preserve the exact meaning supported by the 'passage'.
# Level 1 (Surface Noise): Keep the meaning true. You may remove minor adjectives, generalize numbers slightly, or simplify phrasing.
# Level 2 (Loss of Precision): Omit a secondary detail or make the answer slightly less specific than the 'input_answer'.
# Level 3 (Minor Semantic Error): Alter a specific entity or relationship. Swap a name, date, or location with a plausible but incorrect one not supported by the 'passage'.
# Level 4 (Major Semantic Error): Significantly alter the core meaning. Swap the main Subject/Object or negate the main verb.
# Level 5 (Total Hallucination): Produce a fluent, confident answer that is completely unsupported by the 'passage' or explicitly contradicts it.

# ### CONSTRAINTS
# 1. OUTPUT LANGUAGE: English (unless the input is in another language).
# 2. FORMAT: Output ONLY the resulting text string. Do not include labels, explanations, or quotes.
# """

# MOCHA_FEW_SHOT_PROMPT = """You are a Semantic Corruption Engine for Reading Comprehension.
# Your task is to generate a single "synthetic text" string by modifying the provided 'input_answer' based on the requested 'damage_level'.

# ### GROUND TRUTH PROTOCOL
# 1. **Source of Truth:** The 'passage' is the absolute factual truth. Any deviation from the passage counts as damage.
# 2. **Relevance:** The output must still attempt to answer the 'question', even if the facts are modified (at higher levels).

# ### DAMAGE SPECIFICATIONS
# Level 0 (Paraphrase): Rewrite the 'input_answer' using different words or syntax. You MUST preserve the exact meaning supported by the 'passage'.
# Level 1 (Surface Noise): Keep the meaning true. You may remove minor adjectives, generalize numbers slightly, or simplify phrasing.
# Level 2 (Loss of Precision): Omit a secondary detail or make the answer slightly less specific than the 'input_answer'.
# Level 3 (Minor Semantic Error): Alter a specific entity or relationship. Swap a name, date, or location with a plausible but incorrect one not supported by the 'passage'.
# Level 4 (Major Semantic Error): Significantly alter the core meaning. Swap the main Subject/Object or negate the main verb.
# Level 5 (Total Hallucination): Produce a fluent, confident answer that is completely unsupported by the 'passage' or explicitly contradicts it.

# ### CONSTRAINTS
# 1. OUTPUT LANGUAGE: English (unless the input is in another language).
# 2. FORMAT: Output ONLY the resulting text string. Do not include labels, explanations, or quotes.

# ### EXAMPLES

# User:
# passage: The Apollo 11 mission landed humans on the Moon in July 1969.
# question: When did the landing occur?
# input_answer: It happened in 1969.
# damage_level: 0

# Assistant:
# The landing took place in the year 1969.

# User:
# passage: Photosynthesis takes place inside the chloroplasts, which contain chlorophyll.
# question: Where does photosynthesis happen?
# input_answer: It occurs in the chloroplasts.
# damage_level: 3

# Assistant:
# It occurs in the mitochondria.

# User:
# passage: The blue whale is the largest animal known to have ever lived.
# question: What is the largest animal?
# input_answer: The blue whale.
# damage_level: 5

# Assistant:
# The largest animal is the African Elephant.
# """

MOCHA_BASE_PROMPT = """You are a Semantic Corruption Engine for Reading Comprehension.
Your task is to generate a single "synthetic text" string by modifying the provided 'input_answer' based on the requested 'damage_level'.

### GROUND TRUTH PROTOCOL
1. **Source of Truth:** The 'passage' is the absolute factual truth. Any deviation from the passage counts as damage.
2. **Relevance:** The output must still attempt to answer the 'question', even if the facts are modified (at higher levels).

### DAMAGE SPECIFICATIONS
Level 0 (Paraphrase): Rewrite the 'input_answer' using different words or syntax. You MUST preserve the exact meaning supported by the 'passage'.
Level 1 (Surface Noise): Keep the meaning true. You may remove minor adjectives, generalize numbers slightly, or simplify phrasing.
Level 2 (Loss of Precision): Omit a secondary detail or make the answer slightly less specific than the 'input_answer'.
Level 3 (Minor Semantic Error): Alter a specific entity or relationship. Swap a name, date, or location with a plausible but incorrect one not supported by the 'passage'.
Level 4 (Major Semantic Error): Significantly alter the core meaning. Swap the main Subject/Object or negate the main verb.
Level 5 (Total Hallucination): Produce a fluent, confident answer that is completely unsupported by the 'passage' or explicitly contradicts it.
"""

MOCHA_CONSTRAINTS = """### CONSTRAINTS
1. OUTPUT LANGUAGE: English (unless the input is in another language).
2. FORMAT: Output ONLY the resulting text string. Do not include labels, explanations, or quotes."""

MOCHA_CONSTRAINTS_REASONING = """### CONSTRAINTS
1. OUTPUT LANGUAGE: English (unless the input is in another language).
2. FORMAT: You must adhere to the following structure exactly. Do not output anything after the '### TEXT' section.

### REASONING
(Explain briefly why your output matches the requested damage level)
### TEXT
(The resulting text string)"""

MOCHA_EXAMPLES = """### EXAMPLES

User:
passage: The Apollo 11 mission landed humans on the Moon in July 1969.
question: When did the landing occur?
input_answer: It happened in 1969.
damage_level: 0

Assistant:
The landing took place in the year 1969.

User:
passage: Photosynthesis takes place inside the chloroplasts, which contain chlorophyll.
question: Where does photosynthesis happen?
input_answer: It occurs in the chloroplasts.
damage_level: 3

Assistant:
It occurs in the mitochondria.

User:
passage: The blue whale is the largest animal known to have ever lived.
question: What is the largest animal?
input_answer: The blue whale.
damage_level: 5

Assistant:
The largest animal is the African Elephant.
"""

MOCHA_ZERO_SHOT_PROMPT = f"{MOCHA_BASE_PROMPT}\n{MOCHA_CONSTRAINTS}\n"
MOCHA_FEW_SHOT_PROMPT = f"{MOCHA_BASE_PROMPT}\n{MOCHA_EXAMPLES}\n{MOCHA_CONSTRAINTS}\n"

MOCHA_ZERO_SHOT_REASONING_PROMPT = f"{MOCHA_BASE_PROMPT}\n{MOCHA_CONSTRAINTS_REASONING}\n"
MOCHA_FEW_SHOT_REASONING_PROMPT = f"{MOCHA_BASE_PROMPT}\n{MOCHA_EXAMPLES}\n{MOCHA_CONSTRAINTS_REASONING}\n"


# ROSE_ZERO_SHOT_PROMPT_V5 = """You are an AI Assistant helping to build a dataset for "Contrastive Learning".
# Your job is to generate **Negative Samples** (incorrect summaries) based on a Ground Truth Reference.

# ### TASK
# Given a valid 'reference_summary' and a 'damage_level', generate a modified version that is **factually lower quality** while maintaining the same **fluency** and **style**.

# ### CRITICAL RULES
# 1. **NO DELETION:** Do not simply shorten the text. You must **replace** information with incorrect details to keep the length roughly the same.
# 2. **NO DISCLAIMERS:** Do not output "Here is the negative sample". Just output the text.
# 3. **FACTUALITY:** - Levels 0-2 must be factually TRUE (Paraphrase/Noise).
#    - Levels 3-5 must be factually FALSE (Hallucinations).

# ### DAMAGE LEVELS (Target Metric Behavior)
# Level 0 (Positive Sample): Paraphrase the Reference. Keep all facts. Change words/syntax.
# Level 1 (Blurry): Replace precise numbers/names with generic nouns (e.g. "50%" -> "a significant amount").
# Level 2 (Noisy): Keep the main idea but replace specific details with plausible filler text.
# Level 3 (Entity Swap): **Negative Sample.** Swap the main Subject/Object so the summary claims the wrong person did the action.
# Level 4 (Contradiction): **Negative Sample.** Rewrite the summary to claim the *opposite* outcome (e.g., "Won" -> "Lost"). Keep the original length.
# Level 5 (Hallucination): **Negative Sample.** Write a fluent summary of the same length about a **completely different topic** (e.g., talk about cooking instead of sports).

# ### OUTPUT RULES
# - **Output ONLY the final summary string.** - Do NOT output "Here is the summary". 
# - Do NOT output notes or explanations."""

# ROSE_FEW_SHOT_PROMPT_V5 = """You are an AI Assistant helping to build a dataset for "Contrastive Learning".
# Your job is to generate **Negative Samples** (incorrect summaries) based on a Ground Truth Reference.

# ### TASK
# Given a valid 'reference_summary' and a 'damage_level', generate a modified version that is **factually lower quality** while maintaining the same **fluency** and **style**.

# ### CRITICAL RULES
# 1. **NO DELETION:** Do not simply shorten the text. You must **replace** information with incorrect details to keep the length roughly the same.
# 2. **NO DISCLAIMERS:** Do not output "Here is the negative sample". Just output the text.
# 3. **FACTUALITY:** - Levels 0-2 must be factually TRUE (Paraphrase/Noise).
#    - Levels 3-5 must be factually FALSE (Hallucinations).

# ### DAMAGE LEVELS (Target Metric Behavior)
# Level 0 (Positive Sample): Paraphrase the Reference. Keep all facts. Change words/syntax.
# Level 1 (Blurry): Replace precise numbers/names with generic nouns (e.g. "50%" -> "a significant amount").
# Level 2 (Noisy): Keep the main idea but replace specific details with plausible filler text.
# Level 3 (Entity Swap): **Negative Sample.** Swap the main Subject/Object so the summary claims the wrong person did the action.
# Level 4 (Contradiction): **Negative Sample.** Rewrite the summary to claim the *opposite* outcome (e.g., "Won" -> "Lost"). Keep the original length.
# Level 5 (Hallucination): **Negative Sample.** Write a fluent summary of the same length about a **completely different topic** (e.g., talk about cooking instead of sports).

# ### OUTPUT RULES
# - **Output ONLY the final summary string.** - Do NOT output "Here is the summary". 
# - Do NOT output notes or explanations.

# ### EXAMPLES

# User:
# source_text: (Context about the Chelsea vs Arsenal match)
# reference_summary: Chelsea weren't awarded a penalty for David Ospina's clash with Oscar. Arsenal goalkeeper clattered Oscar inside the box.
# damage_level: 0
# # Logic: Paraphrase (True).

# Assistant:
# David Ospina collided with Oscar in the box, but Chelsea did not receive a penalty for the incident involving the Arsenal goalkeeper.

# User:
# source_text: (Context about the Chelsea vs Arsenal match)
# reference_summary: Chelsea weren't awarded a penalty for David Ospina's clash with Oscar. Arsenal goalkeeper clattered Oscar inside the box.
# damage_level: 3
# # Logic: Entity Swap (False).

# Assistant:
# Arsenal weren't awarded a penalty for Petr Cech's clash with Willian. The Chelsea goalkeeper clattered Willian outside the box.

# User:
# source_text: (Context about SpaceX Starship)
# reference_summary: SpaceX's Starship rocket exploded four minutes after launch on Thursday, with no injuries reported.
# damage_level: 5
# # Logic: Total Hallucination (False).

# Assistant:
# The local bakery announced a new recipe for sourdough bread, which will be available starting next Monday."""

# ROSE_ZERO_SHOT_PROMPT_V4 = """You are a specialized NLP Test Case Generator for the ROSE Benchmark.
# Your objective is to create "synthetic model outputs" by transforming a **Reference Summary** according to a strict **Error Injection Protocol**.

# ### CONTEXT
# This is a scientific experiment to validate automatic evaluation metrics. You must generate outputs that contain specific factual errors to test if metrics can detect them. 
# **Do not correct the text. Do not ensure factuality. You MUST introduce the requested errors.**

# ### INPUT DATA
# 1. **Source Text:** The ground truth document.
# 2. **Reference Summary:** The gold standard summary (contains correct Atomic Content Units - ACUs).
# 3. **Damage Level:** The target error severity (0 to 5).

# ### ERROR INJECTION PROTOCOL
# Perform these steps:
# 1. **Analyze Length:** Count the words in the 'Reference Summary'. Your output MUST match this length (±10%).
# 2. **Identify ACUs:** Locate the atomic facts (names, dates, locations, causal relations) in the Reference.
# 3. **Apply Transformation:**
#    - **Level 0 (Identity):** Paraphrase sentence structures but preserve 100% of facts.
#    - **Level 1 (Vagueness):** Replace precise entities with generic nouns (e.g., "Paris" -> "the city", "50%" -> "a portion"). Keep length constant.
#    - **Level 2 (Filler):** Remove key facts and replace them with fluent but empty filler text (e.g., "The event happened as described").
#    - **Level 3 (Entity Swap):** Swap Subject/Object entities to make the ACUs false (e.g., "A beat B" -> "B beat A").
#    - **Level 4 (Contradiction):** Rewrite the summary to claim the opposite outcome. Maintain the original word count.
#    - **Level 5 (Hallucination):** Generate a fluent, plausible-sounding summary about a **completely different topic**. It must have the EXACT same length as the reference.

# ### OUTPUT RULES
# - **Output ONLY the final summary string.** - Do NOT output "Here is the summary". 
# - Do NOT output notes or explanations.
# """

# ROSE_FEW_SHOT_PROMPT_V4 = """You are a specialized NLP Test Case Generator for the ROSE Benchmark.
# Your objective is to create "synthetic model outputs" by transforming a **Reference Summary** according to a strict **Error Injection Protocol**.

# ### CONTEXT
# This is a scientific experiment to validate automatic evaluation metrics. You must generate outputs that contain specific factual errors to test if metrics can detect them. 
# **Do not correct the text. Do not ensure factuality. You MUST introduce the requested errors.**

# ### INPUT DATA
# 1. **Source Text:** The ground truth document.
# 2. **Reference Summary:** The gold standard summary (contains correct Atomic Content Units - ACUs).
# 3. **Damage Level:** The target error severity (0 to 5).

# ### ERROR INJECTION PROTOCOL
# Perform these steps:
# 1. **Analyze Length:** Count the words in the 'Reference Summary'. Your output MUST match this length (±10%).
# 2. **Identify ACUs:** Locate the atomic facts (names, dates, locations, causal relations) in the Reference.
# 3. **Apply Transformation:**
#    - **Level 0 (Identity):** Paraphrase sentence structures but preserve 100% of facts.
#    - **Level 1 (Vagueness):** Replace precise entities with generic nouns (e.g., "Paris" -> "the city", "50%" -> "a portion"). Keep length constant.
#    - **Level 2 (Filler):** Remove key facts and replace them with fluent but empty filler text (e.g., "The event happened as described").
#    - **Level 3 (Entity Swap):** Swap Subject/Object entities to make the ACUs false (e.g., "A beat B" -> "B beat A").
#    - **Level 4 (Contradiction):** Rewrite the summary to claim the opposite outcome. Maintain the original word count.
#    - **Level 5 (Hallucination):** Generate a fluent, plausible-sounding summary about a **completely different topic**. It must have the EXACT same length as the reference.

# ### OUTPUT RULES
# - **Output ONLY the final summary string.** - Do NOT output "Here is the summary". 
# - Do NOT output notes or explanations.

# ### EXAMPLES

# User:
# source_text: (Context about the Chelsea vs Arsenal match)
# reference_summary: Chelsea weren't awarded a penalty for David Ospina's clash with Oscar. Arsenal goalkeeper clattered Oscar inside the box.
# damage_level: 0

# Assistant:
# David Ospina collided with Oscar in the box, but Chelsea did not receive a penalty for the incident involving the Arsenal goalkeeper.

# User:
# source_text: (Context about the Chelsea vs Arsenal match)
# reference_summary: Chelsea weren't awarded a penalty for David Ospina's clash with Oscar. Arsenal goalkeeper clattered Oscar inside the box.
# damage_level: 2

# Assistant:
# The team was not given a specific advantage for the player's interaction with another. The opponent player hit the man inside the area.

# User:
# source_text: (Context about SpaceX Starship)
# reference_summary: SpaceX's Starship rocket exploded four minutes after launch on Thursday, with no injuries reported.
# damage_level: 5

# Assistant:
# NASA's Falcon 9 booster landed successfully on the drone ship on Friday, marking another milestone.
# """

ROSE_BASE_PROMPT = """You are an Atomic Fact Corruption Engine.
Your task is to generate a "synthetic text" by modifying a 'reference_summary' based on a 'damage_level', specifically targeting Atomic Content Units (ACUs).

### THE ACU PROTOCOL
Summaries are evaluated by breaking them down into "Atomic Content Units" (fine-grained, independent facts) and checking their recall.
- **Goal:** As Damage Level increases, the number of ACUs from the 'reference_summary' preserved in your output must DECREASE.
- **Constraint:** You must maintain the **fluency** and **length** of the original text. Do not simply delete sentences; replace facts with non-facts or plausible hallucinations.

### DAMAGE SPECIFICATIONS (ACU RECALL)
Level 0 (100% ACU Recall): Paraphrase the text but preserve **every single atomic fact** (names, dates, relations, quantities).
Level 1 (80% ACU Recall): Preserve the main story but blur specific details. (e.g., Change "David Ospina" to "the goalkeeper", or "16th minute" to "early on").
Level 2 (60% ACU Recall): Remove minor ACUs. Replace specific facts with generic filler text that sounds relevant but conveys no specific information from the source.
Level 3 (40% ACU Recall): Entity Swap. Keep the sentence structure but swap key entities (Subject/Object) so the ACUs become factually false (e.g., "Chelsea won" -> "Arsenal won").
Level 4 (20% ACU Recall): Major Contradiction. Rewrite the summary to describe a different outcome or event involving the same entities, falsifying nearly all original facts.
Level 5 (0% ACU Recall): Total Hallucination. Generate a fluent summary of the same length that contains **ZERO** facts from the reference. It can be about the same topic but must be factually disjoint.
"""

ROSE_CONSTRAINTS = """### CONSTRAINTS
1. LENGTH: The output must be within ±10% word count of the 'reference_summary'.
2. FLUENCY: The text must be grammatically perfect.
3. FORMAT: Output ONLY the resulting summary string. No labels, explanations, or quotes."""

ROSE_CONSTRAINTS_REASONING = """### CONSTRAINTS
1. LENGTH: The output must be within ±10% word count of the 'reference_summary'.
2. FLUENCY: The text must be grammatically perfect.
3. FORMAT: You must adhere to the following structure exactly. Do not output anything after the '### TEXT' section.

### REASONING
(Explain briefly how you adjusted the ACU recall to match the requested damage level. Mention specific entities modified or facts removed.)
### TEXT
(The resulting summary string)"""

ROSE_EXAMPLES = """### EXAMPLES

User:
source_text: (Context about the Chelsea vs Arsenal match)
reference_summary: Chelsea weren't awarded a penalty for David Ospina's clash with Oscar. Arsenal goalkeeper clattered Oscar inside the box.
damage_level: 0

Assistant:
David Ospina collided with Oscar in the box, but Chelsea did not receive a penalty for the incident involving the Arsenal goalkeeper.

User:
source_text: (Context about the Chelsea vs Arsenal match)
reference_summary: Chelsea weren't awarded a penalty for David Ospina's clash with Oscar. Arsenal goalkeeper clattered Oscar inside the box.
damage_level: 3

Assistant:
Arsenal weren't awarded a penalty for Petr Cech's clash with Willian. The Chelsea goalkeeper clattered Willian outside the box.

User:
source_text: (Context about SpaceX Starship)
reference_summary: SpaceX's Starship rocket exploded four minutes after launch on Thursday, with no injuries reported.
damage_level: 5

Assistant:
NASA's Falcon 9 booster landed successfully on the drone ship on Friday, marking another milestone for the agency.
"""

ROSE_ZERO_SHOT_PROMPT = f"{ROSE_BASE_PROMPT}\n{ROSE_CONSTRAINTS}\n"
ROSE_FEW_SHOT_PROMPT = f"{ROSE_BASE_PROMPT}\n{ROSE_EXAMPLES}\n{ROSE_CONSTRAINTS}\n"

ROSE_ZERO_SHOT_REASONING_PROMPT = f"{ROSE_BASE_PROMPT}\n{ROSE_CONSTRAINTS_REASONING}\n"
ROSE_FEW_SHOT_REASONING_PROMPT = f"{ROSE_BASE_PROMPT}\n{ROSE_EXAMPLES}\n{ROSE_CONSTRAINTS_REASONING}\n"


# ROSE_ZERO_SHOT_PROMPT = """You are an Atomic Fact Corruption Engine.
# Your task is to generate a "synthetic text" by modifying a 'reference_summary' based on a 'damage_level', specifically targeting Atomic Content Units (ACUs).

# ### THE ACU PROTOCOL
# Summaries are evaluated by breaking them down into "Atomic Content Units" (fine-grained, independent facts) and checking their recall.
# - **Goal:** As Damage Level increases, the number of ACUs from the 'reference_summary' preserved in your output must DECREASE.
# - **Constraint:** You must maintain the **fluency** and **length** of the original text. Do not simply delete sentences; replace facts with non-facts or plausible hallucinations.

# ### DAMAGE SPECIFICATIONS (ACU RECALL)
# Level 0 (100% ACU Recall): Paraphrase the text but preserve **every single atomic fact** (names, dates, relations, quantities).
# Level 1 (80% ACU Recall): Preserve the main story but blur specific details. (e.g., Change "David Ospina" to "the goalkeeper", or "16th minute" to "early on").
# Level 2 (60% ACU Recall): Remove minor ACUs. Replace specific facts with generic filler text that sounds relevant but conveys no specific information from the source.
# Level 3 (40% ACU Recall): Entity Swap. Keep the sentence structure but swap key entities (Subject/Object) so the ACUs become factually false (e.g., "Chelsea won" -> "Arsenal won").
# Level 4 (20% ACU Recall): Major Contradiction. Rewrite the summary to describe a different outcome or event involving the same entities, falsifying nearly all original facts.
# Level 5 (0% ACU Recall): Total Hallucination. Generate a fluent summary of the same length that contains **ZERO** facts from the reference. It can be about the same topic but must be factually disjoint.

# ### CONSTRAINTS
# 1. LENGTH: The output must be within ±10% word count of the 'reference_summary'.
# 2. FLUENCY: The text must be grammatically perfect.
# 3. FORMAT: Output ONLY the resulting summary string. No labels.
# """

# # BEST SO FAR FOR QWEN
# ROSE_FEW_SHOT_PROMPT = """You are an Atomic Fact Corruption Engine.
# Your task is to generate a "synthetic text" by modifying a 'reference_summary' based on a 'damage_level', specifically targeting Atomic Content Units (ACUs).

# ### THE ACU PROTOCOL
# Summaries are evaluated by breaking them down into "Atomic Content Units" (fine-grained, independent facts) and checking their recall.
# - **Goal:** As Damage Level increases, the number of ACUs from the 'reference_summary' preserved in your output must DECREASE.
# - **Constraint:** You must maintain the **fluency** and **length** of the original text. Do not simply delete sentences; replace facts with non-facts or plausible hallucinations.

# ### DAMAGE SPECIFICATIONS (ACU RECALL)
# Level 0 (100% ACU Recall): Paraphrase the text but preserve **every single atomic fact** (names, dates, relations, quantities).
# Level 1 (80% ACU Recall): Preserve the main story but blur specific details. (e.g., Change "David Ospina" to "the goalkeeper", or "16th minute" to "early on").
# Level 2 (60% ACU Recall): Remove minor ACUs. Replace specific facts with generic filler text that sounds relevant but conveys no specific information from the source.
# Level 3 (40% ACU Recall): Entity Swap. Keep the sentence structure but swap key entities (Subject/Object) so the ACUs become factually false (e.g., "Chelsea won" -> "Arsenal won").
# Level 4 (20% ACU Recall): Major Contradiction. Rewrite the summary to describe a different outcome or event involving the same entities, falsifying nearly all original facts.
# Level 5 (0% ACU Recall): Total Hallucination. Generate a fluent summary of the same length that contains **ZERO** facts from the reference. It can be about the same topic but must be factually disjoint.

# ### CONSTRAINTS
# 1. LENGTH: The output must be within ±10% word count of the 'reference_summary'.
# 2. FLUENCY: The text must be grammatically perfect.
# 3. FORMAT: Output ONLY the resulting summary string. No labels.

# ### EXAMPLES

# User:
# source_text: (Context about the Chelsea vs Arsenal match)
# reference_summary: Chelsea weren't awarded a penalty for David Ospina's clash with Oscar. Arsenal goalkeeper clattered Oscar inside the box.
# damage_level: 0

# Assistant:
# David Ospina collided with Oscar in the box, but Chelsea did not receive a penalty for the incident involving the Arsenal goalkeeper.

# User:
# source_text: (Context about the Chelsea vs Arsenal match)
# reference_summary: Chelsea weren't awarded a penalty for David Ospina's clash with Oscar. Arsenal goalkeeper clattered Oscar inside the box.
# damage_level: 3

# Assistant:
# Arsenal weren't awarded a penalty for Petr Cech's clash with Willian. The Chelsea goalkeeper clattered Willian outside the box.

# User:
# source_text: (Context about SpaceX Starship)
# reference_summary: SpaceX's Starship rocket exploded four minutes after launch on Thursday, with no injuries reported.
# damage_level: 5

# Assistant:
# NASA's Falcon 9 booster landed successfully on the drone ship on Friday, marking another milestone for the agency.
# """

# ROSE_ZERO_SHOT_PROMPT_2 = """You are a Summary Corruption Engine.
# Your task is to generate a single "synthetic hypothesis" summary by modifying the provided 'reference_summary' based on the requested 'damage_level'.

# ### GROUND TRUTH PROTOCOL
# 1. **Source of Truth:** The 'source_text' is the absolute factual truth.
# 2. **Target Length:** You MUST preserve the approximate word count of 'reference_summary' across ALL levels. Do not simply delete information without replacing it.

# ### DAMAGE SPECIFICATIONS
# Level 0 (Paraphrase): Rewrite the 'reference_summary' using different synonyms or sentence structures. Do not add "fluff." Keep it concise and fully supported by the 'source_text'.
# Level 1 (Surface Noise): Keep the meaning true, but replace specific, precise terms with vague or generic ones (e.g., replace "5.2 million" with "a large number", or "Monday" with "recently"). **Maintain the original length.**
# Level 2 (Vagueness/Loss of Detail): Instead of deleting a key point, replace specific details with generic filler or truisms. (e.g., change "The council voted 12-1 to pass the bill" to "The council held a vote and a decision was eventually made"). The summary becomes uninformative but remains the same length.
# Level 3 (Entity Swap): Keep the exact sentence structure and length, but swap a specific named entity (person, organization, location) with a plausible but wrong one not found in the 'source_text'.
# Level 4 (Contradiction): Rewrite the summary to claim the opposite outcome. Swap the main verb or object. (e.g., "The bill passed" -> "The bill was rejected"). **Do not expand on the lie; keep it the same length.**
# Level 5 (Irrelevance): Produce a fluent, confident summary of a completely unrelated topic. **Crucially, generate a text that is exactly the same length as the 'reference_summary'.**

# ### CONSTRAINTS
# 1. OUTPUT LANGUAGE: Same language as the 'reference_summary'.
# 2. LENGTH PRIORITY: The output word count MUST be within ±10% of the 'reference_summary'.
# 3. FORMAT: Output ONLY the resulting summary string. No labels.
# """

# ROSE_FEW_SHOT_PROMPT_2 = """You are a Summary Corruption Engine.
# Your task is to generate a single "synthetic hypothesis" summary by modifying the provided 'reference_summary' based on the requested 'damage_level'.

# ### GROUND TRUTH PROTOCOL
# 1. **Source of Truth:** The 'source_text' is the absolute factual truth.
# 2. **Target Length:** You MUST preserve the approximate word count of 'reference_summary' across ALL levels. Do not simply delete information without replacing it.

# ### DAMAGE SPECIFICATIONS
# Level 0 (Paraphrase): Rewrite the 'reference_summary' using different synonyms or sentence structures. Do not add "fluff." Keep it concise and fully supported by the 'source_text'.
# Level 1 (Surface Noise): Keep the meaning true, but replace specific, precise terms with vague or generic ones. **Maintain the original length.**
# Level 2 (Vagueness/Loss of Detail): Instead of deleting a key point, replace specific details with generic filler or truisms. The summary becomes uninformative but remains the same length.
# Level 3 (Entity Swap): Keep the exact sentence structure and length, but swap a specific named entity with a plausible but wrong one.
# Level 4 (Contradiction): Rewrite the summary to claim the opposite outcome. **Do not expand on the lie; keep it the same length.**
# Level 5 (Irrelevance): Produce a fluent summary of a completely unrelated topic. **Generate a text that is exactly the same length as the 'reference_summary'.**

# ### EXAMPLES

# User:
# source_text: SpaceX successfully launched its Starship rocket on Thursday morning. However, the vehicle exploded four minutes into the flight during stage separation. No injuries were reported.
# reference_summary: SpaceX's Starship rocket exploded minutes after launch on Thursday, with no injuries.
# damage_level: 0

# Assistant:
# On Thursday, the Starship rocket from SpaceX exploded shortly after launch, causing zero injuries.

# User:
# source_text: The local council voted 12-to-1 to approve the construction of the new library downtown. The project will cost $5 million and is expected to open in 2026.
# reference_summary: The council approved the new $5 million downtown library, set to open in 2026.
# damage_level: 2

# Assistant:
# The council made a decision about a civic project, which is set to conclude in the future.

# User:
# source_text: A rare snow leopard was spotted in the Himalayas for the first time in a decade. Conservationists are celebrating the sighting as a sign of population recovery.
# reference_summary: Conservationists are celebrating the rare sighting of a snow leopard in the Himalayas.
# damage_level: 5

# Assistant:
# The stock market crashed today after the technology sector saw a massive sell-off.

# ### CONSTRAINTS
# 1. OUTPUT LANGUAGE: Same language as the 'reference_summary'.
# 2. LENGTH PRIORITY: The output word count MUST be within ±10% of the 'reference_summary'.
# 3. FORMAT: Output ONLY the resulting summary string. No labels.
# """

WMT_BASE_PROMPT = """You are a Translation Corruption Engine.
Your goal is to take a perfect 'reference_translation' and degrade it according to the specific 'damage_level' requested.

### GROUND TRUTH PROTOCOL
1. **Source of Truth:** The 'source_sentence' and 'reference_translation' define the correct meaning.
2. **Strict adherence:** You must NOT improve the text. You must damage it.

### DAMAGE SPECIFICATIONS
Level 0 (Paraphrase): Rewrite the 'reference_translation' using different synonyms or sentence structures. It MUST remain a valid, high-quality translation of the 'source_sentence' with perfect grammar.
Level 1 (Surface/Mechanical Noise): Keep the words mostly identical to the 'reference_translation', but **inject a visible technical error**. You MUST include a spelling mistake, a capitalization error, missing punctuation, or a blatant subject-verb agreement error (e.g., "he go" instead of "he goes"). The meaning must remain perfect, but the fluency must be damaged.
Level 2 (Omission/Under-translation): Remove a specific detail or nuance found in the 'source_sentence' (e.g., drop an adjective or adverb). The translation is understandable but clearly incomplete compared to the reference.
Level 3 (Word-Level Semantic Error): Mistranslate a specific content word (noun/verb) to a plausible but incorrect alternative (e.g., "car" -> "truck", "walked" -> "ran"). This must be a specific, local error.
Level 4 (Major Semantic Error): Significantly alter the meaning of the whole sentence. Swap the Subject and Object, negate the main verb, or change the tense dramatically (past -> future) if it contradicts the source.
Level 5 (Hallucination/Catastrophic Failure): Produce a fluent sentence in the target language that has NOTHING to do with the 'source_sentence', or is a translation of a completely different input.
"""

WMT_CONSTRAINTS = """### CONSTRAINTS
1. OUTPUT LANGUAGE: The output must be in the SAME LANGUAGE as the 'reference_translation'.
2. NO EXACT MATCHES: For Damage Level 1 and above, the output **MUST NOT** be identical to the 'reference_translation'.
3. FORMAT: Output ONLY the resulting translation string. No labels, no explanations."""

WMT_CONSTRAINTS_REASONING = """### CONSTRAINTS
1. OUTPUT LANGUAGE: The output must be in the SAME LANGUAGE as the 'reference_translation'.
2. NO EXACT MATCHES: For Damage Level 1 and above, the output **MUST NOT** be identical to the 'reference_translation'.
3. FORMAT: You must adhere to the following structure exactly. Do not output anything after the '### TEXT' section.

### REASONING
(Explain briefly which specific error you are injecting to match the requested damage level. Identify the specific word or grammar rule being violated.)
### TEXT
(The resulting translation string)"""

WMT_EXAMPLES = """### EXAMPLES

User:
source_sentence: The cat sat on the mat.
reference_translation: Le chat s'est assis sur le tapis.
damage_level: 0

Assistant:
Le chat était assis sur le tapis.

User:
source_sentence: She bought a red car yesterday.
reference_translation: Sie hat gestern ein rotes Auto gekauft.
damage_level: 3

Assistant:
Sie hat gestern ein blaues Fahrrad gekauft.

User:
source_sentence: Technology is evolving rapidly.
reference_translation: La technologie évolue rapidement.
damage_level: 5

Assistant:
J'aime manger des pommes au petit déjeuner.
"""

WMT_ZERO_SHOT_PROMPT = f"{WMT_BASE_PROMPT}\n{WMT_CONSTRAINTS}\n"
WMT_FEW_SHOT_PROMPT = f"{WMT_BASE_PROMPT}\n{WMT_EXAMPLES}\n{WMT_CONSTRAINTS}\n"

WMT_ZERO_SHOT_REASONING_PROMPT = f"{WMT_BASE_PROMPT}\n{WMT_CONSTRAINTS_REASONING}\n"
WMT_FEW_SHOT_REASONING_PROMPT = f"{WMT_BASE_PROMPT}\n{WMT_EXAMPLES}\n{WMT_CONSTRAINTS_REASONING}\n"

# WMT_ZERO_SHOT_PROMPT = """You are a Translation Corruption Engine.
# Your goal is to take a perfect 'reference_translation' and degrade it according to the specific 'damage_level' requested.

# ### GROUND TRUTH PROTOCOL
# 1. **Source of Truth:** The 'source_sentence' and 'reference_translation' define the correct meaning.
# 2. **Strict adherence:** You must NOT improve the text. You must damage it.

# ### DAMAGE SPECIFICATIONS
# Level 0 (Paraphrase): Rewrite the 'reference_translation' using different synonyms or sentence structures. It MUST remain a valid, high-quality translation of the 'source_sentence' with perfect grammar.
# Level 1 (Surface/Mechanical Noise): Keep the words mostly identical to the 'reference_translation', but **inject a visible technical error**. You MUST include a spelling mistake, a capitalization error, missing punctuation, or a blatant subject-verb agreement error (e.g., "he go" instead of "he goes"). The meaning must remain perfect, but the fluency must be damaged.
# Level 2 (Omission/Under-translation): Remove a specific detail or nuance found in the 'source_sentence' (e.g., drop an adjective or adverb). The translation is understandable but clearly incomplete compared to the reference.
# Level 3 (Word-Level Semantic Error): Mistranslate a specific content word (noun/verb) to a plausible but incorrect alternative (e.g., "car" -> "truck", "walked" -> "ran"). This must be a specific, local error.
# Level 4 (Major Semantic Error): Significantly alter the meaning of the whole sentence. Swap the Subject and Object, negate the main verb, or change the tense dramatically (past -> future) if it contradicts the source.
# Level 5 (Hallucination/Catastrophic Failure): Produce a fluent sentence in the target language that has NOTHING to do with the 'source_sentence', or is a translation of a completely different input.

# ### CONSTRAINTS
# 1. OUTPUT LANGUAGE: The output must be in the SAME LANGUAGE as the 'reference_translation'.
# 2. NO EXACT MATCHES: For Damage Level 1 and above, the output **MUST NOT** be identical to the 'reference_translation'.
# 3. FORMAT: Output ONLY the resulting translation string. No labels, no explanations.
# """
# WMT_FEW_SHOT_PROMPT = """You are a Translation Corruption Engine.
# Your goal is to take a perfect 'reference_translation' and degrade it according to the specific 'damage_level' requested.

# ### GROUND TRUTH PROTOCOL
# 1. **Source of Truth:** The 'source_sentence' and 'reference_translation' define the correct meaning.
# 2. **Strict adherence:** You must NOT improve the text. You must damage it.

# ### DAMAGE SPECIFICATIONS
# Level 0 (Paraphrase): Rewrite the 'reference_translation' using different synonyms or sentence structures. It MUST remain a valid, high-quality translation of the 'source_sentence' with perfect grammar.
# Level 1 (Surface/Mechanical Noise): Keep the words mostly identical to the 'reference_translation', but **inject a visible technical error**. You MUST include a spelling mistake, a capitalization error, missing punctuation, or a blatant subject-verb agreement error (e.g., "he go" instead of "he goes"). The meaning must remain perfect, but the fluency must be damaged.
# Level 2 (Omission/Under-translation): Remove a specific detail or nuance found in the 'source_sentence' (e.g., drop an adjective or adverb). The translation is understandable but clearly incomplete compared to the reference.
# Level 3 (Word-Level Semantic Error): Mistranslate a specific content word (noun/verb) to a plausible but incorrect alternative (e.g., "car" -> "truck", "walked" -> "ran"). This must be a specific, local error.
# Level 4 (Major Semantic Error): Significantly alter the meaning of the whole sentence. Swap the Subject and Object, negate the main verb, or change the tense dramatically (past -> future) if it contradicts the source.
# Level 5 (Hallucination/Catastrophic Failure): Produce a fluent sentence in the target language that has NOTHING to do with the 'source_sentence', or is a translation of a completely different input.

# ### CONSTRAINTS
# 1. OUTPUT LANGUAGE: The output must be in the SAME LANGUAGE as the 'reference_translation'.
# 2. NO EXACT MATCHES: For Damage Level 1 and above, the output **MUST NOT** be identical to the 'reference_translation'.
# 3. FORMAT: Output ONLY the resulting translation string. No labels, no explanations.

# ### EXAMPLES

# User:
# source_sentence: The cat sat on the mat.
# reference_translation: Le chat s'est assis sur le tapis.
# damage_level: 0

# Assistant:
# Le chat était assis sur le tapis.

# User:
# source_sentence: She bought a red car yesterday.
# reference_translation: Sie hat gestern ein rotes Auto gekauft.
# damage_level: 3

# Assistant:
# Sie hat gestern ein blaues Fahrrad gekauft.

# User:
# source_sentence: Technology is evolving rapidly.
# reference_translation: La technologie évolue rapidement.
# damage_level: 5

# Assistant:
# J'aime manger des pommes au petit déjeuner.
# """

# WMT_ZERO_SHOT_PROMPT_ORIG = """You are a Translation Corruption Engine.
# Your task is to generate a single "synthetic hypothesis" translation by modifying the provided 'reference_translation' based on the requested 'damage_level', while considering the 'source_sentence'.

# ### GROUND TRUTH PROTOCOL
# 1. **Source of Truth:** The 'source_sentence' and 'reference_translation' define the correct meaning.
# 2. **Goal:** Generate a translation that deviates from the perfect quality based on the damage level.

# ### DAMAGE SPECIFICATIONS
# Level 0 (Paraphrase): Rewrite the 'reference_translation' using different synonyms or sentence structures. It MUST remain a valid, accurate translation of the 'source_sentence'.
# Level 1 (Surface Noise): meaningful but imperfect. Introduce minor grammatical errors (wrong article, singular/plural mismatch) or remove a minor adjective.
# Level 2 (Omission/Under-translation): Remove a specific detail or nuance found in the 'source_sentence'. The translation is understandable but incomplete.
# Level 3 (Word-Level Semantic Error): Mistranslate a specific content word (noun/verb) to a plausible but incorrect alternative (e.g., "car" -> "truck", "walked" -> "ran").
# Level 4 (Major Semantic Error): Significantly alter the meaning. Swap the Subject and Object, negate the main verb, or change the tense dramatically (past -> future) if it contradicts the source.
# Level 5 (Hallucination/Catastrophic Failure): Produce a fluent sentence in the target language that has NOTHING to do with the 'source_sentence', or is a translation of a completely different input.

# ### CONSTRAINTS
# 1. OUTPUT LANGUAGE: The output must be in the SAME LANGUAGE as the 'reference_translation'.
# 2. FORMAT: Output ONLY the resulting translation string. No labels.
# """

# WMT_FEW_SHOT_PROMPT_ORIG = """You are a Translation Corruption Engine.
# Your task is to generate a single "synthetic hypothesis" translation by modifying the provided 'reference_translation' based on the requested 'damage_level', while considering the 'source_sentence'.

# ### GROUND TRUTH PROTOCOL
# 1. **Source of Truth:** The 'source_sentence' and 'reference_translation' define the correct meaning.
# 2. **Goal:** Generate a translation that deviates from the perfect quality based on the damage level.

# ### DAMAGE SPECIFICATIONS
# Level 0 (Paraphrase): Rewrite the 'reference_translation' using different synonyms or sentence structures. It MUST remain a valid, accurate translation of the 'source_sentence'.
# Level 1 (Surface Noise): meaningful but imperfect. Introduce minor grammatical errors (wrong article, singular/plural mismatch) or remove a minor adjective.
# Level 2 (Omission/Under-translation): Remove a specific detail or nuance found in the 'source_sentence'. The translation is understandable but incomplete.
# Level 3 (Word-Level Semantic Error): Mistranslate a specific content word (noun/verb) to a plausible but incorrect alternative (e.g., "car" -> "truck", "walked" -> "ran").
# Level 4 (Major Semantic Error): Significantly alter the meaning. Swap the Subject and Object, negate the main verb, or change the tense dramatically (past -> future) if it contradicts the source.
# Level 5 (Hallucination/Catastrophic Failure): Produce a fluent sentence in the target language that has NOTHING to do with the 'source_sentence', or is a translation of a completely different input.

# ### CONSTRAINTS
# 1. OUTPUT LANGUAGE: The output must be in the SAME LANGUAGE as the 'reference_translation'.
# 2. FORMAT: Output ONLY the resulting translation string. No labels.

# ### EXAMPLES

# User:
# source_sentence: The cat sat on the mat.
# reference_translation: Le chat s'est assis sur le tapis.
# damage_level: 0

# Assistant:
# Le chat était assis sur le tapis.

# User:
# source_sentence: She bought a red car yesterday.
# reference_translation: Sie hat gestern ein rotes Auto gekauft.
# damage_level: 3

# Assistant:
# Sie hat gestern ein blaues Fahrrad gekauft.

# User:
# source_sentence: Technology is evolving rapidly.
# reference_translation: La technologie évolue rapidement.
# damage_level: 5

# Assistant:
# J'aime manger des pommes au petit déjeuner.
# """