"""
Phase 3 — Local sanity-check inference (converted from 3_kaggle_inference.ipynb).

Changes vs notebook:
  - Paths set for local environment (not Kaggle)
  - CUDA_VISIBLE_DEVICES=4 (single GPU isolation)
  - N_TEST_ROWS=20 for quick sanity check; set to -1 for full test.csv
  - adapter_path points to ./adapters/<name>_adapter
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import contextlib, gc, json, logging, re, shutil, sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from rapidfuzz.fuzz import partial_ratio_alignment
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer,
    BitsAndBytesConfig,
)
from vllm import LLM, SamplingParams

try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    GuidedDecodingParams = None
    print("WARNING: GuidedDecodingParams not found — guided decoding disabled")

try:
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    def destroy_model_parallel(): pass

# ── CONFIG ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "DATA_DIR":                Path("."),
    "ADAPTER_DIR":             Path("./adapters"),
    "OUTPUT_DIR":              Path("./output_phase3"),
    "ENFORCE_EAGER":           True,
    "GPU_MEM_UTIL":            0.85,
    "MAX_MODEL_LEN":           1024,
    "MAX_NEW_TOKENS":          1024,
    "LLM_TEMPERATURE":         0.0,
    "MAX_SPANS_PER_FEATURE":   10,
    "VOTE_THRESHOLD":          2,
    "FUZZY_SCORE_CUTOFF":      70.0,
    "SEED":                    42,
    "LARGE_MODEL_THRESHOLD_B": 7,
    "N_TEST_ROWS":             20,   # set to -1 for full test.csv
    "USE_VLLM":                True,  # True = try vLLM first, auto-fallback to transformers on error
}

MODEL_REGISTRY = [
    {
        "name":         "qwen_35_9b",
        "model_id":     "Qwen/Qwen3.5-9B",
        "model_class":  "causal_lm",
        "dtype":        torch.bfloat16,
        "vllm_dtype":   "bfloat16",
        "adapter_path": Path("./adapters/qwen_35_9b_adapter"),
        "param_count":  9,
    },
    {
        "name":         "gemma_4_e4b",
        "model_id":     "google/gemma-4-E4B-it",
        "model_class":  "image_text_to_text",
        "dtype":        torch.bfloat16,
        "vllm_dtype":   "bfloat16",
        "adapter_path": Path("./adapters/gemma_4_e4b_adapter"),
        "param_count":  4,
    },
    {
        "name":         "qwen_35_4b",
        "model_id":     "Qwen/Qwen3.5-4B",
        "model_class":  "causal_lm",
        "dtype":        torch.bfloat16,
        "vllm_dtype":   "bfloat16",
        "adapter_path": Path("./adapters/qwen_35_4b_adapter"),
        "param_count":  4,
    },
]

SYSTEM_PROMPT = (
    "You are a clinical NLP specialist. "
    "Given a patient note and a clinical feature, extract the EXACT verbatim text spans "
    "from the note that express that feature. "
    "Rules:\n"
    "  1. Copy text character-for-character — do NOT paraphrase.\n"
    "  2. If the feature is absent from the note, return an empty list.\n"
    "  3. Output ONLY valid JSON — no markdown, no explanation.\n"
    'Output format: {"spans": ["exact text 1", "exact text 2"]}'
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── Section 1 — Per-Note Regex FSM Constraint ──────────────────────────────────
def _build_char_class(note_chars: set) -> str:
    parts = []
    for ch in sorted(note_chars, key=ord):
        code = ord(ch)
        if code < 0x20 or code == 0x7F:
            continue
        if ch == ']':    parts.append(r'\]')
        elif ch == '^':  parts.append(r'\^')
        elif ch == '-':  parts.append(r'\-')
        elif ch == '\\': parts.append(r'\\')
        else:            parts.append(ch)
    return '[' + ''.join(parts) + ']' if parts else r'[^\n]'


def build_constraint_regex(pn_history: str, max_spans: int = 10) -> str:
    note_chars = set(pn_history) - {'"', '\\'}
    char_class = _build_char_class(note_chars)
    span_item  = f'"{char_class}*"'
    additional = r'(?:, ' + span_item + r'){0,' + str(max_spans - 1) + r'}'
    opt_list   = r'(?:' + span_item + additional + r')?'
    pattern    = r'\{"spans": \[' + opt_list + r'\]\}'
    return pattern


# ── Section 2 — LoRA Adapter Merger ────────────────────────────────────────────
def merge_adapter_to_disk(model_spec: dict, output_dir: Path, cfg: dict) -> Path:
    model_id     = model_spec["model_id"]
    model_class  = model_spec["model_class"]
    adapter_path = model_spec["adapter_path"]
    dtype        = model_spec["dtype"]
    merged_path  = output_dir / f"merged_{model_spec['name']}"
    is_large     = model_spec.get("param_count", 0) >= cfg.get("LARGE_MODEL_THRESHOLD_B", 7)

    if merged_path.exists() and (merged_path / "config.json").exists():
        log.info(f"  [{model_spec['name']}] Merged model already on disk → {merged_path}")
        return merged_path

    # Pre-patch model_type via AutoConfig so AutoModelForCausalLM resolves correct arch.
    # "qwen3_5" → Qwen3_5ForConditionalGeneration (nested weights), wrong for text-only.
    # "qwen3_5_text" → Qwen3_5ForCausalLM (flat weights), correct.
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_id)
    if getattr(model_config, "model_type", "") == "qwen3_5":
        model_config.model_type = "qwen3_5_text"
        log.info(f"  [{model_spec['name']}] Pre-patched AutoConfig model_type: qwen3_5 → qwen3_5_text")

    load_kwargs = dict(pretrained_model_name_or_path=model_id, torch_dtype=dtype,
                       config=model_config)

    if is_large:
        log.info(f"  [{model_spec['name']}] Large model ({model_spec['param_count']}B) — 4-bit NF4 merge ...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_quant_type       = "nf4",
            bnb_4bit_compute_dtype    = dtype,
            bnb_4bit_use_double_quant = True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"]          = {"": "cuda:0"}
    else:
        log.info(f"  [{model_spec['name']}] Small model ({model_spec['param_count']}B) — CPU merge ...")
        load_kwargs["device_map"] = "cpu"

    if model_class == "causal_lm":
        base_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    else:
        base_model = AutoModelForImageTextToText.from_pretrained(**load_kwargs)

    log.info(f"  [{model_spec['name']}] Merging LoRA adapter from {adapter_path} ...")
    peft_model   = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = peft_model.merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_path), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(merged_path))

    del base_model, peft_model, merged_model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Patch model_type for Qwen3.5 only: "qwen3_5" → "qwen3_5_text" so transformers
    # resolves Qwen3_5ForCausalLM (flat weights) not Qwen3_5ForConditionalGeneration.
    cfg_path = merged_path / "config.json"
    with open(cfg_path) as f:
        model_cfg = json.load(f)
    current_type = model_cfg.get("model_type", "")
    if current_type == "qwen3_5":
        model_cfg["model_type"] = "qwen3_5_text"
        with open(cfg_path, "w") as f:
            json.dump(model_cfg, f, indent=2)
        log.info(f"  [{model_spec['name']}] Patched saved config model_type: qwen3_5 → qwen3_5_text")

    log.info(f"  [{model_spec['name']}] Merge complete → {merged_path}")
    return merged_path


# ── Section 3 — Prompt Builder ─────────────────────────────────────────────────
def build_chat_prompt(feature_text: str, pn_history: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Note: \"{pn_history.strip()}\"\n"
            f"Feature: {feature_text}\n\n"
            "/no_think"
        )},
    ]
    # enable_thinking=False suppresses <think> block for Qwen3.5; ignore if tokenizer doesn't support
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Section 4 — vLLM Engine Lifecycle ─────────────────────────────────────────
def init_engine(merged_path: Path, model_spec: dict, cfg: dict) -> LLM:
    log.info(f"  [{model_spec['name']}] Initialising vLLM engine ...")
    llm = LLM(
        model                  = str(merged_path),
        dtype                  = model_spec["vllm_dtype"],
        gpu_memory_utilization = cfg["GPU_MEM_UTIL"],
        max_model_len          = cfg["MAX_MODEL_LEN"],
        enforce_eager          = cfg["ENFORCE_EAGER"],
        trust_remote_code      = False,
        seed                   = cfg["SEED"],
    )
    log.info(f"  [{model_spec['name']}] vLLM engine ready.")
    return llm


def destroy_engine(llm: LLM, model_name: str) -> None:
    log.info(f"  [{model_name}] Destroying vLLM engine ...")
    destroy_model_parallel()
    with contextlib.suppress(Exception):
        torch.distributed.destroy_process_group()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
        total_gb = torch.cuda.mem_get_info()[1] / 1024**3
        log.info(f"  [{model_name}] VRAM after cleanup: {free_gb:.1f}/{total_gb:.1f} GB")


# ── Section 5 — Inference Runner ───────────────────────────────────────────────
def _parse_json_output(raw_text: str) -> list:
    raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(raw_text)
        return [s.strip() for s in parsed.get("spans", []) if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, AttributeError):
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return [s.strip() for s in parsed.get("spans", []) if isinstance(s, str) and s.strip()]
            except json.JSONDecodeError:
                pass
        return []


def run_inference_for_model(llm, test_rows, pn_map, feat_map, tokenizer, cfg, model_name) -> list:
    log.info(f"  [{model_name}] Building prompts and per-note regex constraints ...")
    prompts, params_list = [], []

    for _, row in test_rows.iterrows():
        pn_history   = pn_map.get(row["pn_num"], "").replace("\n", " ").strip()
        feature_text = feat_map.get((row["case_num"], row["feature_num"]), "")
        prompts.append(build_chat_prompt(feature_text, pn_history, tokenizer))
        regex = build_constraint_regex(pn_history, cfg["MAX_SPANS_PER_FEATURE"])
        if GuidedDecodingParams is not None:
            guided = GuidedDecodingParams(regex=regex, backend="xgrammar")
            sp = SamplingParams(
                temperature=cfg["LLM_TEMPERATURE"], max_tokens=cfg["MAX_NEW_TOKENS"],
                guided_decoding=guided,
            )
        else:
            sp = SamplingParams(temperature=cfg["LLM_TEMPERATURE"], max_tokens=cfg["MAX_NEW_TOKENS"])
        params_list.append(sp)

    log.info(f"  [{model_name}] Running inference on {len(prompts)} rows ...")
    outputs = llm.generate(prompts=prompts, sampling_params=params_list)

    all_spans = []
    for output in tqdm(outputs, desc=f"  [{model_name}] Parsing", leave=False):
        raw_text = output.outputs[0].text.strip() if output.outputs else ""
        all_spans.append(_parse_json_output(raw_text))

    n_nonempty = sum(1 for s in all_spans if s)
    log.info(f"  [{model_name}] Done — non-empty: {n_nonempty}/{len(all_spans)}")
    return all_spans


# ── Section 5b — Transformers Fallback Inference (no vLLM) ─────────────────────
def run_inference_transformers(merged_path: Path, test_rows, pn_map, feat_map,
                               tokenizer, cfg, model_name, model_spec: dict = None) -> list:
    """Simple transformers generate() path — no vLLM required."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    log.info(f"  [{model_name}] Loading merged model for transformers inference ...")

    model_class = (model_spec or {}).get("model_class", "causal_lm")
    load_kwargs  = dict(torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})

    # For causal_lm: patch model_type if qwen3_5 so correct arch is resolved
    if model_class == "causal_lm":
        saved_config = AutoConfig.from_pretrained(str(merged_path))
        if getattr(saved_config, "model_type", "") == "qwen3_5":
            saved_config.model_type = "qwen3_5_text"
            load_kwargs["config"] = saved_config
        model = AutoModelForCausalLM.from_pretrained(str(merged_path), **load_kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(str(merged_path), **load_kwargs)
    model.eval()
    log.info(f"  [{model_name}] Running transformers inference on {len(test_rows)} rows ...")
    all_spans = []
    for _, row in tqdm(test_rows.iterrows(), total=len(test_rows), desc=f"  [{model_name}]"):
        pn_history   = pn_map.get(row["pn_num"], "").replace("\n", " ").strip()
        feature_text = feat_map.get((row["case_num"], row["feature_num"]), "")
        prompt = build_chat_prompt(feature_text, pn_history, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=cfg["MAX_NEW_TOKENS"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids  = out_ids[0, inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        all_spans.append(_parse_json_output(raw_text))
    n_nonempty = sum(1 for s in all_spans if s)
    log.info(f"  [{model_name}] Done — non-empty: {n_nonempty}/{len(all_spans)}")
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    return all_spans


# ── Section 6 — Character-Level Majority Voting ────────────────────────────────
def spans_to_char_array(span_locations: list, note_len: int) -> np.ndarray:
    arr = np.zeros(note_len, dtype=np.uint8)
    for start, end in span_locations:
        arr[max(0, start):min(note_len, end)] = 1
    return arr


def char_array_to_spans(arr: np.ndarray) -> list:
    spans, n, i = [], len(arr), 0
    while i < n:
        if arr[i] == 1:
            start = i
            while i < n and arr[i] == 1: i += 1
            spans.append((start, i))
        else:
            i += 1
    return spans


def locate_span_in_note(span_text: str, pn_history: str, score_cutoff: float = 70.0) -> Optional[tuple]:
    span_text = span_text.strip()
    if not span_text or not pn_history:
        return None
    idx = pn_history.find(span_text)
    if idx != -1: return (idx, idx + len(span_text))
    idx = pn_history.lower().find(span_text.lower())
    if idx != -1: return (idx, idx + len(span_text))
    result = partial_ratio_alignment(span_text, pn_history, score_cutoff=score_cutoff)
    if result is not None: return (result.dest_start, result.dest_end)
    return None


def character_level_majority_vote(model_predictions, test_rows, pn_map, vote_threshold=2, fuzzy_cutoff=70.0) -> list:
    n_models, n_rows = len(model_predictions), len(test_rows)
    log.info(f"Majority vote ({n_models} models, threshold={vote_threshold}/{n_models}) ...")
    final_spans = []

    for seq_idx, (_, row) in enumerate(tqdm(test_rows.iterrows(), total=n_rows, desc="Majority vote")):
        pn_history = pn_map.get(row["pn_num"], "")
        note_len   = len(pn_history)
        if note_len == 0:
            final_spans.append([]); continue

        vote_array = np.zeros(note_len, dtype=np.int8)
        for model_idx in range(n_models):
            locations = [loc for text in model_predictions[model_idx][seq_idx]
                         if (loc := locate_span_in_note(text, pn_history, fuzzy_cutoff)) is not None]
            if locations:
                vote_array += spans_to_char_array(locations, note_len)

        consensus = (vote_array >= vote_threshold).astype(np.uint8)
        for i, ch in enumerate(pn_history):
            if ch in (' ', '\t', '\n', '\r') and consensus[i]:
                is_start = (i == 0 or consensus[i-1] == 0)
                is_end   = (i == note_len-1 or consensus[i+1] == 0)
                if is_start or is_end: consensus[i] = 0

        final_spans.append(char_array_to_spans(consensus))

    non_empty = sum(1 for s in final_spans if s)
    log.info(f"Vote complete — non-empty: {non_empty}/{n_rows}")
    return final_spans


# ── Section 7 — Submission Formatter ──────────────────────────────────────────
def format_location_string(spans: list, pn_history: str) -> str:
    if not spans: return ""
    clean = []
    for start, end in sorted(spans):
        while start < end and pn_history[start] in (' ', '\t', '\n', '\r'): start += 1
        while end > start and pn_history[end-1] in (' ', '\t', '\n', '\r'): end -= 1
        if start < end: clean.append((start, end))

    merged = []
    for start, end in sorted(clean):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return ";".join(f"{s} {e}" for s, e in merged) if merged else ""


def build_submission(final_spans: list, test_df: pd.DataFrame, pn_map: dict) -> pd.DataFrame:
    rows = []
    for row_idx, (_, test_row) in enumerate(test_df.iterrows()):
        pn_history = pn_map.get(test_row["pn_num"], "")
        spans      = final_spans[row_idx] if row_idx < len(final_spans) else []
        location   = format_location_string(spans, pn_history)
        rows.append({"id": test_row["id"], "location": location if location else np.nan})
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    cfg["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

    # Startup VRAM clear
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
        total_gb = torch.cuda.mem_get_info()[1] / 1024**3
        log.info(f"GPU ready: {free_gb:.1f}/{total_gb:.1f} GB free")

    print("\n" + "="*65)
    print("  PHASE 3: Inference (local sanity check)")
    print("="*65 + "\n")

    print("▶ Loading test data ...")
    data_dir = cfg["DATA_DIR"]
    test_df  = pd.read_csv(data_dir / "test.csv")
    pn_df    = pd.read_csv(data_dir / "patient_notes.csv")
    feat_df  = pd.read_csv(data_dir / "features.csv")
    pn_map   = pn_df.set_index("pn_num")["pn_history"].to_dict()
    feat_map = feat_df.set_index(["case_num", "feature_num"])["feature_text"].to_dict()

    n_rows = cfg["N_TEST_ROWS"]
    if n_rows > 0:
        test_df = test_df.head(n_rows).reset_index(drop=True)
        print(f"  [SANITY CHECK] Using first {n_rows} rows (set N_TEST_ROWS=-1 for full run)")
    print(f"  Test rows: {len(test_df)}")

    tmp_root = cfg["OUTPUT_DIR"] / "merged_models"
    tmp_root.mkdir(parents=True, exist_ok=True)

    all_model_predictions = []
    for i, model_spec in enumerate(MODEL_REGISTRY):
        model_name = model_spec["name"]
        print(f"\n{'='*65}")
        print(f"  Model {i+1}/{len(MODEL_REGISTRY)}: {model_name}")
        print(f"{'='*65}")

        merged_path = merge_adapter_to_disk(model_spec, tmp_root, cfg)

        tokenizer = AutoTokenizer.from_pretrained(str(merged_path))
        if tokenizer.pad_token is None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if cfg["USE_VLLM"]:
            try:
                llm         = init_engine(merged_path, model_spec, cfg)
                model_spans = run_inference_for_model(llm, test_df, pn_map, feat_map, tokenizer, cfg, model_name)
                all_model_predictions.append(model_spans)
                destroy_engine(llm, model_name)
                del llm
            except Exception as e:
                log.warning(f"  [{model_name}] vLLM failed ({e}) — falling back to transformers")
                with contextlib.suppress(Exception):
                    destroy_model_parallel()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache(); torch.cuda.synchronize()
                model_spans = run_inference_transformers(merged_path, test_df, pn_map, feat_map, tokenizer, cfg, model_name, model_spec)
                all_model_predictions.append(model_spans)
        else:
            model_spans = run_inference_transformers(merged_path, test_df, pn_map, feat_map, tokenizer, cfg, model_name, model_spec)
            all_model_predictions.append(model_spans)
        del tokenizer; gc.collect()
        shutil.rmtree(str(merged_path), ignore_errors=True)
        print(f"  ✓ [{model_name}] merged model deleted from disk")

    print("\n▶ Running character-level majority vote ...")
    n_models_active = len(all_model_predictions)
    # Use min(VOTE_THRESHOLD, n_active) so partial runs (1-2 models) still produce output
    effective_threshold = min(cfg["VOTE_THRESHOLD"], n_models_active)
    final_spans = character_level_majority_vote(
        all_model_predictions, test_df, pn_map,
        vote_threshold=effective_threshold, fuzzy_cutoff=cfg["FUZZY_SCORE_CUTOFF"],
    )

    submission_df = build_submission(final_spans, test_df, pn_map)
    out_path = cfg["OUTPUT_DIR"] / "submission.csv"
    submission_df.to_csv(out_path, index=False)

    print("\n" + "="*65)
    print(f"  ✓ Submission saved → {out_path}")
    print(f"  Shape: {submission_df.shape}")
    print(f"  Non-empty: {submission_df['location'].notna().sum()} / {len(submission_df)}")
    print("="*65)
    print(submission_df.to_string())

main()
