# Verification notes

- Fast metric shard input preparation and order restoration are covered by
  `tests/test_metric_shard.py`.
- The Colab artifact is generated from
  `documents/nlp-ck_group/metric_shard_colab_source.py`; regenerate it with
  `build_metric_shard_notebook.py` before packaging.
- Heavy metric runtime remains environment-dependent and must be smoke-tested on
  a Colab GPU after local contract tests pass.

## 2026-09-06 verification

- `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'`: 20 passed.
- Notebook contract suite: 25 passed.
- On 30 sentence pairs, accelerated BLEU/chrF/METEOR matched the prior adapters
  exactly (`max_diff=0.0`) and removed repeated `evaluate.compute()` overhead.
- A real light-metric smoke run completed 16 syllable and 8 underthesea
  configurations, then restored all six synthetic rows in their original order.
- Inclusive sentence-range selection was verified on two of three sentence IDs;
  all 12 selected damage rows retained their original source indexes.
- Colab GPU execution is not available in the local workspace, so model download,
  CUDA batch sizing, and end-to-end BERTScore/COMET/BLEURT remain runtime checks.

## 2026-09-06 damage-generation batching comparison

- The current generator issues one request per `(sentence, prompt type, level)`,
  so 300 references, two prompt types, and six levels require 3,600 requests.
- When API keys belong to distinct Google projects, each project has its own
  quota. The current notebook caller still applies one shared semaphore and one
  shared request interval across all keys, so it does not exploit independent
  project quotas.
- Recommended direct-call experiment: generate all six levels for one sentence
  in one structured-JSON response. This reduces the run to 600 requests and
  keeps failures and checkpointing isolated to one source sentence.
- Do not mix bundled-six outputs with the existing one-level-per-request run.
  Bundling can improve level consistency but introduces cross-level anchoring
  and changes the experimental protocol; use a new run name and validate it on a
  stratified pilot against the original protocol.
- Grouping many unrelated sentences at one level saves requests but has greater
  risk of skipped/reordered IDs, uneven generation quality, and larger retry
  loss. If used, keep groups small and require schema validation.
- Gemini Batch API preserves independent one-level prompts and has separate
  batch quota, but it is asynchronous and targets completion within 24 hours.

## Agent context: accelerated Gemini damage notebook

- Last updated: 2026-09-06
- Sources: `documents/nlp-ck_group/fast_damage_colab_source.py`,
  `build_fast_damage_notebook.py`, `test_fast_damage_notebook.py`, and
  `fast-damage-colab.ipynb`.
- Verified: the standalone notebook accepts the distributed 300-row CSV and
  canonical reference JSONL; it creates 600 bundled requests for 300 sentences,
  two prompt branches, and six levels per response. Each API key has one fixed
  worker and independent pacing. Checkpoints are atomic per sentence/prompt,
  completed work is skipped on resume, daily-quota tasks are requeued, API keys
  are absent from checkpoints, and final JSONL preserves the metric input schema.
- Verification matrix:
  - Add-project scalability: implementation uses one worker per deduplicated key;
    fake three-key run used all three workers — VERIFIED.
  - Six-level bundling: parser rejects missing, duplicate, empty, or duplicated
    level outputs — VERIFIED.
  - Resume: second fake run made zero caller invocations — VERIFIED.
  - RPD failure: task taken by an exhausted project moved to another project —
    VERIFIED.
  - Existing notebook contracts: 31 tests passed — VERIFIED.
- Commands: `python3 build_fast_damage_notebook.py`; `python3 -m unittest -v
  test_fast_damage_notebook.py` (6 passed); `python3 -m unittest discover -s
  documents/nlp-ck_group -p 'test_*.py'` (31 passed).
- Unknown: no end-to-end generation run was intentionally completed in Colab;
  response quality and actual latency remain runtime checks. The broader package
  suite had 17 passing tests and three environment-only errors because local
  `openpyxl` is missing.
- Artifact: `documents/nlp-ck_group/fast-damage-colab.ipynb`.
- Next update: record a 1–3 sentence Colab smoke run, observed RPM, structured
  response validity, and any model-specific API compatibility findings.

### 2026-09-06 key-pool update

- Replaced the standalone notebook's direct key array with eight distinct
  user-provided project keys; no key value is recorded in this context file.
- Rebuilt `fast-damage-colab.ipynb` and verified the generated config contains
  eight unique entries.
- Offline verification: all six fast-damage notebook tests passed. No Gemini
  request was made during this update.

## Agent context: paper-to-fast-prompt check

- Last updated: 2026-09-06
- Paper: `documents/paper/2603.09403v3.pdf`, arXiv v3, section 4.3 and
  Appendix B Tables 13 and 15 (machine-translation zero/few-shot prompts).
- Code inspected: `src/human_feedback_datasets/prompts.py` lines 943–1005,
  `src/generation/generate.py` lines 196–229 and 244–266, and
  `documents/nlp-ck_group/fast_damage_colab_source.py` lines 224–263.
- Verified matches: source sentence plus reference translation are inputs; six
  levels 0–5 are represented; target output is in the reference language; and
  temperature 0 follows the paper's deterministic-generation intent.
- Verified discrepancies: the paper generates one string for one requested
  level, while the fast notebook returns all six levels as JSON. The notebook's
  Vietnamese level definitions materially differ at levels 1–5. Its prohibition
  on facts/entities outside the source directly conflicts with the paper's level
  5 unrelated hallucination. Its few-shot branch has no concrete demonstrations,
  whereas Table 15 includes three full level 0/3/5 demonstrations.
- Implication: the notebook is a speed-oriented adaptation, not a faithful
  reproduction of the paper prompt. Mixing its outputs with paper-protocol
  outputs can alter level separation and synthetic metric correlations.
- Coverage ledger: section 4.3 checked; Table 13 checked fully; Table 15 checked
  fully; QA/summarization prompts and sensitivity variants not needed for this
  machine-translation comparison.
- Unknown: empirical magnitude of prompt/protocol drift on the Vietnamese set;
  it requires a paired pilot or regeneration under both protocols.
- Artifact: no report page; the user requested a direct prompt-equivalence
  answer rather than a standalone report.
- Next update: decide between exact paper reproduction and a documented bundled
  adaptation before launching the 300-sentence generation run.

### 2026-09-06 exact-paper fast notebook update

- The standalone fast notebook now uses `gemini-3.1-flash-lite` and a fresh run
  name. Only this notebook/source was changed; the shared experiment notebook and
  package generation workflow were left untouched.
- Supersedes the bundled-six protocol: each request now contains exactly one
  source/reference pair and one damage level. Checkpoints are keyed by
  `(id, prompt_type, damage_level)`, yielding 3,600 independent requests for 300
  references, two prompt types, and six levels.
- Paper-code alignment matrix:
  - WMT zero-shot system prompt: exact equality with `WMT_ZERO_SHOT_PROMPT` — HIGH.
  - WMT few-shot system prompt and all three examples: exact equality with
    `WMT_FEW_SHOT_PROMPT` — HIGH.
  - User template `source_sentence/reference_translation/damage_level` — HIGH.
  - One generated string per requested level and temperature 0 — HIGH.
  - Generator model — LOW for strict reproduction because the paper evaluates
    Llama/Qwen models; Gemini 3.1 Flash-Lite is the user's explicit adaptation.
- Parallelism: one fixed worker and independent 4.2-second pacing per deduplicated
  project key. Daily-quota failures requeue the current task for another worker;
  checkpoints never store key values.
- Offline full-data plan: 300 references, 3,600 pending requests, eight project
  workers, nominal 450 requests/project, and 31.5-minute theoretical lower bound.
- Verification: seven focused tests and 32 complete notebook contract tests pass.
  Prompt equality, one-level checkpoint/resume, per-key workers, daily-quota
  requeue, output schema, CSV/JSONL loading, and generated notebook synchronization
  are VERIFIED. No Gemini generation request was made during this update.
- Unknown: live Gemini 3.1 output quality, actual latency, and whether all eight
  projects retain sufficient RPM/RPD at execution time.

### 2026-09-06 Vietnamese localization assessment

- Recommendation: preserve the paper's six severity semantics, one-level request
  granularity, temperature, and zero/few-shot separation, but localize only
  language-dependent guidance and demonstrations.
- Highest-value localization is Level 1: replace the English-specific
  subject–verb-agreement example with mandatory Vietnamese surface noise such as
  one or two diacritic, capitalization, punctuation, spacing, or spelling errors,
  while preserving every fact and relation.
- Replace the French/German few-shot demonstrations with manually prepared
  Chinese-to-Vietnamese examples at levels 0, 3, and 5 that are not drawn from
  the evaluated 300 sentences. Keep the level definitions unchanged.
- This would be a paper-faithful Vietnamese adaptation rather than verbatim prompt
  reproduction. The current exact-English prompt remains runnable and requests
  the same language as the Vietnamese reference, but its Level 1 and few-shot
  steering are less language-appropriate.
- No notebook change was made for this assessment. A small paired pilot is still
  needed to measure the effect of localization on level separability.

## Agent context: Vietnamese damage-prompt provenance
- Last updated: 2026-09-06
- Sources: `documents/plans/plans.pdf` sections 3 and 5.2;
  `src/human_feedback_datasets/prompts.py` lines 943–1002;
  `src/generation/generate.py` lines 196–229 and 244–266;
  `src/vn_meta_judge/generation/prompts.py` lines 6–46.
- Verified: the author code contains the canonical English MT zero-shot and
  few-shot prompts, including three demonstrations at levels 0, 3, and 5. The
  plan explicitly calls for translating Tables 13 and 15 into Vietnamese and
  replacing Level 1 with Vietnamese diacritic/capitalization/punctuation noise.
- Verified: the legacy Vietnamese pipeline also contains a custom prompt, but
  its level semantics and few-shot branch diverge from the paper; it is not a
  drop-in paper-faithful prompt.
- Inferred: the intended project protocol is the paper prompt structure plus the
  plan's Vietnamese Level 1 adaptation and Chinese-to-Vietnamese demonstrations.
- Unknown: no complete, reviewed Chinese-to-Vietnamese few-shot prompt matching
  that intended protocol was found in the old codebase.
- Artifact: no standalone report; this trace supports a direct code-provenance
  answer.
- Next update: if requested, implement and test the localized prompt in the
  standalone fast notebook without changing request granularity or concurrency.

### 2026-09-06 Vietnamese prompt implementation verification

- Scope: `documents/nlp-ck_group/fast_damage_colab_source.py`, generated
  `fast-damage-colab.ipynb`, and `test_fast_damage_notebook.py`.
- Requirements matrix:
  - Translate the paper's MT prompt into Vietnamese while preserving levels
    0–5 — implementation and focused assertions present — VERIFIED.
  - Apply the plan's Vietnamese Level 1 wording for missing/wrong tone marks,
    capitalization, or punctuation while preserving meaning — exact content
    asserted — VERIFIED.
  - Replace multilingual paper demonstrations with Chinese-to-Vietnamese
    examples at levels 0, 3, and 5 — examples and branch inclusion asserted —
    VERIFIED.
  - Preserve one level per request, two prompt branches, per-project workers,
    checkpoint resume, and metric output schema — existing behavioral contracts
    all pass — VERIFIED.
  - Prevent reuse of checkpoints made with the English prompt — new run name,
    protocol identifier, and prompt hash contract — VERIFIED.
- Commands: `python3 build_fast_damage_notebook.py`;
  `python3 -m unittest -v test_fast_damage_notebook.py` (7 passed);
  `python3 -m unittest discover -s documents/nlp-ck_group -p 'test_*.py'`
  (32 passed).
- Unknown: generation quality and Gemini 3.1 model availability were not tested
  with a live API call; run a small Colab pilot before launching all 3,600
  requests.

## Agent context: fast-generation apparent stall after 20 requests
- Last updated: 2026-09-06
- Symptom: the Colab generation cell remained active for more than four minutes;
  its last messages were 10/3,600 and 20/3,600 successful requests.
- Verified: this is not the notebook's mock path; the generation cell calls the
  REST-backed `call_paper_damage`. Progress is printed only after every ten
  successes. Per-request errors and retries are intentionally silent until the
  whole worker pool returns.
- Verified: one HTTP operation may wait 180 seconds, with up to five HTTP retries
  and three task attempts. A persistently slow/failing task can therefore occupy
  a worker for roughly 45 minutes before final failure, excluding backoff.
- Inferred: the screenshot is most consistent with workers waiting on Gemini,
  sleeping after 429/5xx responses, or repeatedly failing output validation; it
  does not establish which of those occurred.
- Contradicted: a nonexistent model is unlikely for that run because at least 20
  calls completed successfully.
- Unknown: the exact API response/error is unavailable because the remote Colab
  runtime is not connected here and the current notebook suppresses intermediate
  exception details.
- Artifact: no code change was made during this diagnostic.
- Next update: capture one pending request with one retry and a short timeout, or
  add heartbeat/error telemetry before changing retry/concurrency behavior.

### 2026-09-06 key-utilization follow-up

- Static configuration contains eight non-empty, unique keys; values are not
  recorded here.
- Verified with the focused fake-caller contract: every supplied key receives
  exactly one worker and all supplied workers execute tasks.
- Runtime caveat: only keys classified as `ready`, `quota_warning`, or
  `unchecked` enter `ACTIVE_KEYS`; `unavailable` keys are excluded. A worker
  exits after a recognized daily-quota error, so eight configured keys do not
  guarantee eight workers remain active throughout a run.
- With eight healthy workers and a 4.2-second per-project interval, nominal
  throughput is 114.3 requests/minute and 3,600 requests have a 31.5-minute
  lower bound. A multi-minute gap after request 20 is therefore not explained
  by configured pacing alone.
- Verification command: from `documents/nlp-ck_group`,
  `python3 -m unittest -v test_fast_damage_notebook.FastDamageNotebookTests.test_one_worker_per_key_generates_six_rows_and_resumes`
  — passed. An initial invocation from the workspace root used an invalid module
  path and failed before loading the test; rerunning from the notebook directory
  resolved that command-only issue.
- Unknown: actual `ACTIVE_KEYS`, disabled slots, and HTTP errors in the user's
  live Colab session remain unavailable without its printed key-status output or
  generation summary.

### 2026-09-06 nineteen-key notebook check

- Static inspection without exposing credential values found 19 non-empty,
  unique keys in `fast-damage-colab.ipynb`, but only eight in its maintainable
  `fast_damage_colab_source.py` source. Rebuilding currently overwrites the
  notebook back to the eight-key source configuration.
- The saved notebook has no execution counts or output blocks from the live
  Colab extension, so actual `API_KEYS`, `ACTIVE_KEYS`, key statuses, and worker
  failures in the running kernel remain UNKNOWN.
- If all 19 keys are active and belong to independent projects, the configured
  4.2-second interval gives a nominal 271.4 requests/minute and a 13.3-minute
  lower bound for 3,600 successful calls. The observed multi-minute stall is
  therefore CONTRADICTED by pacing alone.
- Most likely discriminating check: print counts for `GEMINI_API_KEYS`,
  `API_KEYS`, and `ACTIVE_KEYS`, plus a status histogram from `key_status`, after
  stopping the current cell. A runtime count below 19 means the edited config
  was not rerun or keys were excluded; 19 active keys shifts the investigation
  to quota, latency, and hidden retry failures.
- Artifact: no implementation change was made during this diagnostic.

### 2026-09-06 live nineteen-worker evidence

- User-provided Colab output reports 19 configured keys, 19 unique keys, 19
  `ACTIVE_KEYS`, and 19 `ready` statuses — worker-count configuration is
  VERIFIED for that runtime.
- The current `ready` probe only sends `GET /v1beta/models/{model}`. It verifies
  authentication/model visibility but does not consume a generation request or
  verify generation RPM/RPD, latency, output validity, or retry behavior.
- Therefore insufficient workers is CONTRADICTED as the cause of the observed
  stall. Remaining hypotheses are generate-content quota/service errors, long
  HTTP waits, and output validation failures hidden by the worker retry loop.
- Next update: run one concurrent `generateContent` probe per slot with one retry
  and a 30-second timeout, recording only slot, latency, exception type, and a
  redacted error; do not print credential values.

### 2026-09-06 task-distribution trace

- Diagnostic probe: intentionally sends the same one pending task to every key
  so key health can be compared under identical input. All 19 attempts returned
  HTTP 429 and produced no checkpoint.
- Main generation: creates one task for each unique `(sentence id, prompt type,
  damage level)` and atomically pops each task from one shared queue. Two workers
  do not process the same task concurrently during normal execution.
- The same source sentence can legitimately be handled by several workers
  because it owns 12 distinct tasks: two prompt branches times six levels.
- Exact-task repetition occurs only after failure: the unfinished task is
  requeued and can be retried by another worker. Successful tasks are persisted
  and skipped on resume.

### 2026-09-06 RPD-versus-progress root cause

- User evidence: all 19 concurrent generation probes returned HTTP 429
  `RESOURCE_EXHAUSTED`; one project dashboard reportedly showed more than 200
  daily requests while notebook progress showed only 80/3,600 successful rows.
- Verified counter mismatch: notebook progress increments only after a valid
  response passes parsing and its checkpoint is written. It does not count HTTP
  attempts, retry responses, valid API responses later rejected by local output
  validation, diagnostic probes, or earlier runs during the same quota day.
- Verified retry amplification: `MAX_HTTP_RETRIES=5` is nested inside
  `MAX_TASK_ATTEMPTS=3`, so one failing logical task can cause up to 15 HTTP
  attempts before terminal failure. Local validation failures can cause three
  separately quota-consuming successful inference calls for one logical task.
- Verified classification gap: generic 429 bodies such as `You exceeded your
  current quota` or `Resource has been exhausted` do not match the current daily
  markers (`perday`, `daily quota`, `rpd`). They are treated as transient,
  repeatedly retried, and only reported after the whole run.
- Root cause: the 80-row success counter cannot be compared directly with RPD;
  retry amplification, validation retries, probes, and prior daily usage explain
  the larger dashboard count. The generic-quota classification and missing
  attempt telemetry are implementation defects.
- Artifact: no fix made in this diagnostic turn. Do not rerun full generation
  until quota recovers and retry/error telemetry is corrected.

### 2026-09-06 quota-safe generator implementation

- Scope: `fast_damage_colab_source.py`, generated
  `fast-damage-colab.ipynb`, `build_fast_damage_notebook.py`, and
  `test_fast_damage_notebook.py`.
- Requirements matrix:
  - Any HTTP 429 disables that project slot immediately with no HTTP-layer
    retry — generic `RESOURCE_EXHAUSTED` test makes exactly one URL call —
    VERIFIED.
  - All projects returning 429 terminates quickly with a partial result instead
    of cycling through the task queue — three-slot simulation makes exactly
    three calls and leaves all 12 tasks pending — VERIFIED.
  - Transient/network/output errors cannot recreate the previous 5 × 3 retry
    amplification — HTTP retry loop removed and retryable logical tasks capped
    at two calls — VERIFIED.
  - Live resource telemetry distinguishes successful checkpoints from API calls,
    requeues, terminal failures, and disabled slots — implementation plus
    behavioral assertions — VERIFIED.
  - Successful checkpoints and generation schema remain reusable — resume and
    output-contract tests pass — VERIFIED.
  - Rebuilding must preserve the 19 user-entered notebook keys without printing
    or copying them into logs — before/after static counts remain 19 unique and
    builder preservation test passes — VERIFIED.
- Commands: `python3 build_fast_damage_notebook.py`;
  `python3 -m unittest -v test_fast_damage_notebook.py` (11 passed);
  `python3 -m unittest discover -s documents/nlp-ck_group -p 'test_*.py'`
  (36 passed); Python compilation and all generated code-cell parses passed.
- Live API: intentionally not run, because all 19 project probes were already
  returning 429 and the user requested no further quota waste.
- Runtime caveat: an already-open Colab kernel still contains the old function
  definitions until the updated notebook's config/runtime/input/key-check cells
  are rerun. Existing output checkpoints remain valid under the unchanged run
  name and generation protocol.

### 2026-09-06 RPM/RPD-aware 429 refinement

- Supersedes the earlier rule that disabled a slot on every HTTP 429.
- Classification matrix:
  - Quota IDs/metrics or messages containing per-day/RPD markers, or explicit
    depleted-credit/billing markers — disable the slot immediately — VERIFIED.
  - Per-minute/per-second RPM/TPM markers or a Gemini `RetryInfo` delay — retain
    the slot, cooldown according to the server hint, and retry once — VERIFIED.
  - Generic `RESOURCE_EXHAUSTED` without decisive metadata — classify as
    `unknown`, cooldown 60 seconds once, then disable only after a second
    consecutive 429 — VERIFIED.
  - Any successful generation resets that slot's consecutive rate-limit count —
    VERIFIED by recovery behavior.
- The model-visibility GET check no longer preemptively drops a key on its own
  429; the first actual generation request performs the decisive classification.
- Telemetry now reports API calls, task requeues, rate-limit waits, quota kind,
  and remaining active slots. The removed HTTP retry loop remains absent.
- Notebook regeneration preserved all 19 unique user-entered keys and all code
  cells parse. Focused suite: 16 passed. Complete notebook suite from the project
  root: 41 passed.
- One intermediate discovery run from the nested notebook directory selected a
  Python environment without `openpyxl`, causing 23 unrelated setup errors;
  rerunning from the project root's configured environment passed all 41 tests.
- No live Gemini request was made during implementation or verification.

### 2026-09-06 replacement model after quota incident

- User dashboard evidence showed `gemini-3.1-flash-lite` at 21/15 RPM and
  350/500 RPD, while `gemini-3.5-flash-lite` was at 501/500 RPD. The first is a
  temporary per-minute overage with daily capacity remaining; the second is
  daily exhausted for the displayed project.
- Google's current public documentation no longer promises one fixed
  interactive RPD for every account; effective limits are project/tier-specific
  and must be read in AI Studio. Therefore a higher exact RPD for a replacement
  model is UNKNOWN without that project's dashboard.
- Replaced the standalone notebook default with stable
  `gemini-2.5-flash-lite`, an officially supported high-throughput model whose
  per-model quota is separate from the used 3.1/3.5 model buckets. Changed
  `RUN_NAME` so checkpoints from different generator models cannot be mixed.
- Increased project pacing from 4.2 to 5.0 seconds per request (12 RPM), below
  the observed 15 RPM limit. The conservative planning value remains 500 RPD;
  3,600 tasks over 19 independent projects estimate 190 requests/project.
- Verification: 16 focused tests and all 41 notebook tests passed; generated
  notebook has 19 configured/unique keys, the new model/run name, and 5-second
  pacing. No live Gemini request was made.

### 2026-09-06 unavailable-model incident and fail-closed startup

- Live Colab evidence contradicted the prior replacement recommendation:
  `gemini-2.5-flash-lite` returned HTTP 404 `NOT_FOUND` stating that it is no
  longer available to new users and recommending `gemini-3.5-flash-lite`.
- Root cause of the 114-call log: non-retryable 404 failed individual tasks but
  did not invalidate the shared model, so every worker continued taking new
  tasks. The notebook's `API calls` value counts HTTP attempts; whether these
  rejected 404s affected provider RPD remains UNKNOWN without dashboard data.
- Default model is now stable `gemini-3.5-flash-lite`, with a new run name.
  The current user dashboard already showed that model at 501/500 RPD for one
  project, so the notebook must not be run there until quota resets or another
  independent project has remaining quota.
- Added fail-closed startup: one pending task is used as the compatibility
  request before parallel workers start. It is checkpointed on success, so it
  is not an extra generation. A 404 raises `ModelUnavailableError` and ends the
  run after exactly one API call without opening the worker pool. If no project
  can complete the startup task, the pool also remains closed.
- The Live API models shown by the user are not substitutes for this workflow:
  Native Audio/Flash Live are realtime session models, Live Translate is
  speech-to-speech, and Transcribe Live is audio-to-text; they do not fit the
  current text `generateContent` plus JSONL checkpoint contract.
- Verification: 18 focused tests and all 43 notebook tests passed. The rebuilt
  notebook contains 19 unique keys, `gemini-3.5-flash-lite`, 5-second pacing,
  and startup validation enabled. No live Gemini request was made by the fix.

### 2026-09-06 two-runtime metric notebooks

- Scope: branch-fixed Colab notebooks generated from
  `metric_shard_colab_source.py` via the parameterized
  `build_metric_shard_notebook.py` builder.
- Requirements matrix:
  - Separate zero-shot notebook reads `zero_shot.jsonl` and writes beneath
    `metric-output/zero_shot` — generated config and parse checks — VERIFIED.
  - Separate few-shot notebook reads `few_shot.jsonl` and writes beneath
    `metric-output/few_shot` — generated config and parse checks — VERIFIED.
  - Both run the complete configured metric design: 28 syllable configurations
    plus eight BLEU/chrF underthesea sensitivity configurations, with heavy GPU
    metrics enabled — generated configs — VERIFIED.
  - Both remain self-contained when supplied the same code bundle and optional
    `human_all.jsonl`; checkpoint/output paths cannot collide across independent
    runtimes — source inspection and branch config tests — VERIFIED.
- Performance design: run one branch per Colab GPU concurrently; light families
  are CPU-parallel, heavy configurations are GPU-batched sequentially to avoid
  model-memory contention, adaptive OOM fallback is enabled, and task
  checkpoints resume completed metric configurations.
- Artifacts: `metric-zero-shot-colab.ipynb`,
  `metric-few-shot-colab.ipynb`, and rebuilt `meta-judge-code.zip`.
- Verification: four focused metric-notebook tests and all 45 notebook tests
  passed; all generated code cells parse and the code zip integrity test passed.
- Unknown: the user's live Colab JSONL files are outside the local workspace, so
  their exact 1,800-row completeness and end-to-end heavy model download/runtime
  remain Colab validation steps.

### 2026-09-06 metric notebook organization

- Renamed and moved the two fixed-branch artifacts into
  `documents/nlp-ck_group/metric-colab/`: `score-zero-shot.ipynb` and
  `score-few-shot.ipynb`. The former root-level generated names were moved, not
  duplicated.
- The notebooks have identical scoring/setup cells. Only `BRANCH_NAME`, the
  derived synthetic filename, and branch-specific output path differ.
- Updated the builder and tests so future regeneration writes directly to the
  organized folder and cannot recreate the old root-level names.
- Verification: four focused builder tests passed; the full 45-test suite passed
  from the project root. A nested-directory run selected a Python environment
  without `openpyxl` and produced 23 unrelated setup errors; this was resolved
  by rerunning from the established project-root environment.

### 2026-09-06 root work handoff layout

- Moved user-facing runtime artifacts out of `documents/` into a root-level
  `work/` handoff directory while keeping maintainable Python notebook sources
  and tests under `documents/nlp-ck_group/`.
- Current layout: `work/main-experiment.ipynb`;
  `work/cheat-runtime/{generate-damage-fast,score-zero-shot,score-few-shot}.ipynb`;
  and `work/resources/{meta-judge-code.zip,source.csv,scores.xlsx,human_all.jsonl}`.
- Updated all notebook/bundle builders, tests, the light verification runner,
  and current run documentation so regeneration targets the new paths. Old
  user-facing paths under `documents/nlp-ck_group` no longer exist.
- Resource verification: `source.csv` has 300 rows; `human_all.jsonl` has 600
  rows, of which 582 satisfy the clean human-score filter; `scores.xlsx` is
  byte-identical to `documents/Cham_diem_DeTai46_latest.xlsx`; and the runtime
  zip passes archive integrity checks.
- Notebook verification: the preserved main notebook cell sources equal the
  builder output, all four runtime notebook code cells parse, fast generation
  retained its user-entered configuration through the path-aware builder, and
  the full 45-test suite passed from the project root.
- Artifact guide: `work/README.md` lists each notebook and the exact resources
  to upload to independent Colab runtimes.

### 2026-09-06 COMET missing dependency fix

- Symptom: the few-shot metric runtime completed all four BERTScore variants,
  then failed on the first COMET task with `ModuleNotFoundError: No module named
  'lightning_utilities'`.
- Root cause — VERIFIED: metric families are installed with `pip --no-deps`;
  `pytorch-lightning==2.6.5` declares `lightning-utilities>=0.10.0`, but the
  manually maintained COMET requirement set omitted it.
- Fix: pinned `lightning-utilities==0.15.3`, rebuilt the runtime ZIP and both
  branch notebooks, added a setup-time COMET import probe, and made source ZIP
  extraction refresh when the uploaded bundle digest changes. The requirement
  fingerprint also changes, so an existing COMET target cache is repaired on
  setup without discarding completed metric checkpoints.
- Verification: five focused metric-notebook tests and all 46 notebook tests
  passed using the workspace interpreter. The generated few-shot notebook
  contains both guards, the runtime ZIP contains the new pin, and local imports
  of `lightning_utilities`, `pytorch_lightning`, and `comet` pass.
- A first broad-suite command from the nested source directory selected a
  different Python installation without `openpyxl`; this was a test-environment
  issue and passed when rerun from the workspace root. End-to-end Colab COMET
  model download/scoring remains a runtime verification step.

### 2026-09-06 COMET misleading unsupported-model failure

- Symptom after repairing `lightning_utilities`: the first COMET task failed in
  about 15 seconds with `KeyError: Model 'Unbabel/wmt22-comet-da' not supported
  by COMET`.
- Root cause — VERIFIED: COMET 2.2.7 first calls Hugging Face
  `snapshot_download`, catches every exception without preserving it, then
  attempts its legacy S3 alias table. `Unbabel/wmt22-comet-da` is intentionally
  absent from that legacy table, so any Hub/network failure is misreported as an
  unsupported model. Hugging Face model metadata confirmed that all four
  configured repositories remain public and contain a checkpoint; a subsequent
  live metadata request returned HTTP 429 `maximum queue size reached`.
- Fix: added `vn_meta_judge.metrics.comet_cache.download_comet_checkpoint`.
  It normalizes short names to `Unbabel/...`, checks the complete local snapshot
  first, downloads directly from the Hub with at most four workers, respects
  `Retry-After`, retries transient 429/5xx/network failures after 15/30/60
  seconds, and exposes the original terminal error. Both prefetch and scoring
  use this resolver, so the scoring subprocess reuses the downloaded snapshot
  without another network request.
- Verification: four focused resolver tests cover local-cache preference,
  short-name normalization, 429 recovery, and preserved non-retryable errors;
  all four metric-shard tests and all 46 notebook tests pass. The rebuilt source
  bundle contains the new resolver. Full model download and GPU scoring remain
  UNKNOWN until rerun in Colab.

### 2026-09-06 zero-shot light-prefetch failure

- Symptom: a separate zero-shot Colab runtime installed all three metric package
  targets and passed the COMET dependency probe, then the setup cell raised a
  bare `CalledProcessError` from `vn_meta_judge.notebook_workflow prefetch`.
- Exact failing asset — UNKNOWN: the old parent process did not capture the child
  traceback. The subprocess contains only Evaluate script fetches and NLTK data
  downloads. A fresh-cache local reproduction completed successfully, while
  parallel Hub access had already returned 429 during this incident, making a
  transient remote fetch failure the leading inference.
- Fix: prefetch now logs and retries each asset after 15/30/60 seconds. It fetches
  only Evaluate BLEU and ROUGE because chrF and METEOR use local direct adapters,
  reducing Hub setup requests from four to two. The notebook streams subprocess
  output and reports the last 40 lines instead of a bare `CalledProcessError`.
- Both `score-zero-shot.ipynb` and `score-few-shot.ipynb` were rebuilt from the
  same setup source, and the shared runtime ZIP was rebuilt with the retrying
  workflow. Verification: 29 focused notebook tests, eight cache/shard tests,
  and all 47 notebook tests pass. End-to-end behavior on the affected Colab
  network remains UNKNOWN until that runtime reruns setup.

### 2026-09-06 BLEURT checkpoint mismatch fix

- Symptom: the few-shot runtime reached the first BLEURT task, downloaded a
  roughly 405 MB archive, then failed because `bleurt-tiny-128` was not found.
- Root cause — VERIFIED against the installed Evaluate API and official BLEURT
  metric source: the runner passed `checkpoint=` to `evaluate.load`, but the
  checkpoint selector is `config_name`. The unknown keyword was ignored and the
  default checkpoint was downloaded. A second defect searched only the legacy
  `HF_HOME/datasets/downloads/extracted` layout, while current Evaluate metric
  assets live below `HF_HOME/metrics/<metric>/<config>/.../downloads/extracted`.
- Fix: both shard and legacy notebook prefetch now pass
  `config_name=spec.kwargs["checkpoint"]`. BLEURT cache discovery searches the
  current metrics layout plus older evaluate/datasets layouts recursively and
  accepts only directories containing `bleurt_config.json`, preventing a parent
  config directory from being mistaken for the extracted checkpoint.
- Rebuilt both zero-shot/few-shot notebooks and the shared code ZIP.
  Verification: 11 focused path/shard/COMET tests and all 48 notebook tests pass;
  archive inspection confirms both call sites and the cache resolver are in the
  runtime bundle. Actual BLEURT model loading/scoring on Colab remains UNKNOWN
  until the updated ZIP is extracted in that remote runtime.

### 2026-09-06 Git-native main notebook and single-load metric scheduler

- Scope: `work/main-experiment.ipynb`, vendored `work/meta-judge/`, Git resources,
  `vn_meta_judge.notebook_workflow`, and `vn_meta_judge.metrics.shard`.
- Root cause — VERIFIED: the main notebook still used the superseded ZIP
  bootstrap and per-dataset heavy scheduler. That route could load each of 12
  heavy configurations once for every human/B1/zero/few dataset. The repaired
  COMET/BLEURT source was also unavailable until a new ZIP was uploaded.
- Fix: `work/` is now an independent Git repo containing source, requirements,
  B2 archive, notebook and input resources. The notebook finds that layout
  locally or clones one configured HTTPS remote on Colab; it no longer imports
  `ZipFile` or references `CODE_ZIP`. Pre-generated damage is copied atomically
  and audited, so default Run All makes no Gemini request and embeds no API key.
- Dependency guard: NumPy/Pandas/SciPy are pinned compatibly; each isolated
  heavy-family target is import-probed and only a stale/partial target is
  rebuilt. COMET retains direct-Hub checkpoint retries. BLEURT uses
  `config_name`, current/legacy cache discovery, and interrupted-download retry.
- Performance fix: the scheduler concatenates human, B1, zero-shot and few-shot
  with hashes and offsets, runs light families in parallel, loads each heavy
  config once, then restores exact source order. The theoretical heavy-model
  load count drops from up to 48 to 12. A preflight smoke mode runs one config
  per family on 24 combined rows.
- Verification — VERIFIED: 49 notebook contract tests and 27 vendored package
  tests pass. Simulation produced 28/28 syllable and 8/8 underthesea configs.
  A real no-API light smoke completed four syllable and two underthesea configs
  on 24 rows without errors. Notebook cells parse and contain no temporary step
  comments or embedded API keys.
- UNKNOWN: full BERTScore/COMET/BLEURT GPU execution on a fresh Colab. The local
  Git repo also needs a user-owned remote and one configured `WORK_REPO_URL`
  before an external Colab runtime can clone it.

### 2026-09-06 smoke-only heavy-metric mode

- Requirement: permit a fresh Colab GPU to run real heavy metric checks without
  continuing into the full dataset. Implementation sets
  `RUN_HEAVY_METRICS=True`, `RUN_METRIC_SMOKE_TEST=True`, and defaults
  `RUN_FULL_METRICS=False`; the smoke uses 24 combined rows and one
  representative configuration from each metric family — VERIFIED by notebook
  configuration and scheduler contracts.
- Requirement: Run All must not accidentally call the full scheduler or stale
  downstream analysis in smoke-only mode. The full, correlation, analysis,
  diagnostics, demo, and finalize cells are guarded by `FULL_METRICS_READY` —
  VERIFIED by a behavioral notebook-cell test and a real light-metric Run All.
- Requirement: setting `RUN_FULL_METRICS=True` must preserve the optimized full
  scheduler — VERIFIED by a behavioral notebook-cell test.
- Commands: notebook contract suite 28 passed; vendored package suite 27 passed;
  `python3 documents/nlp-ck_group/verify_light_run.py` exercises every notebook
  cell locally with real light metrics and no API.
- UNKNOWN: this machine has no CUDA device, so BERTScore, COMET, and BLEURT
  downloads/model inference in the 24-row heavy smoke still require one fresh
  Colab GPU run.

### 2026-09-06 Colab NumPy/Pandas ABI failure

- Symptom: after dependency installation, the main notebook failed at
  `import pandas` with `numpy.dtype size changed`, reporting C-header size 96
  versus runtime object size 88.
- Root cause — VERIFIED: `requirements-notebook.txt` included the pinned core
  stack (`numpy==1.26.4`, `pandas==2.2.3`, `scipy==1.14.1`) and pip installed it
  into the active Colab interpreter. The ABI probe ran in a clean subprocess,
  while the notebook kernel could retain the pre-install NumPy module in RAM;
  therefore the probe could pass and the subsequent in-kernel Pandas import
  could still fail.
- Fix: notebook requirements no longer include or pin the core stack. Before
  installing auxiliary packages, setup records the runtime's existing
  NumPy/Pandas/SciPy versions and supplies them to pip as constraints. It checks
  that pip did not change them and rejects an already-tainted kernel when a
  loaded module version differs from the installed distribution.
- Hypothesis test — VERIFIED in a clean temporary virtualenv: dependency install
  preserved all three scientific-stack versions and Pandas imported afterward.
- Regression verification: 29 notebook tests and 28 vendored package tests pass;
  the loaded-versus-disk mismatch guard and unpinned notebook requirements have
  dedicated tests.
- Runtime caveat: a Colab kernel already tainted by the previous notebook cannot
  be repaired safely in place. Delete that runtime once, reconnect, and run the
  updated notebook. Heavy CUDA smoke remains UNKNOWN until that run completes.
