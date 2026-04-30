#!/usr/bin/env python
# coding: utf-8
# Phase 1 — Generate Augmented Training Data
# Run with: nohup python run_phase1.py > phase1.log 2>&1 &

import os
for f in ['features.csv', 'patient_notes.csv', 'train.csv']:
    status = "✓ found" if os.path.exists(f) else "✗ MISSING"
    print(f"  {f}: {status}")


# ## Configuration
# 
# Edit `CONFIG` in the cell below before running.
# 
# | Key | Current value | Notes |
# |-----|--------------|-------|
# | `SAMPLE_SIZE` | `2000` | 2000 notes × ~14.8 features ≈ 29 600 pairs |
# | `LLM_MODEL` | `Qwen/Qwen3-8B` | fp16, ~16 GB VRAM; cached locally |
# | `LLM_4BIT` | `False` | depends on system |
# | `MAX_NEW_TOKENS` | `128` | Sufficient for JSON span output |
# | `BATCH_SIZE` | `32` | Pairs per GPU call; 32×4096 tokens fits in 140 GB |
# | `CUDA_DEVICE` | `"cuda:0"` | = physical cuda:1 via `CUDA_VISIBLE_DEVICES=1` |
# | `CHECKPOINT_EVERY` | `100` | Save progress every 100 rows |
# | `FUZZY_SCORE_CUTOFF` | `72` | Raise to reduce false-positive span matches |
# 

# In[5]:


import ast, gc, json, logging, os, re, sys
from pathlib import Path

# ── Pin to ONE physical GPU before any CUDA import ────────────────────────────
# Sets CUDA_VISIBLE_DEVICES=1 so this process only sees physical cuda:1.
# Inside the process it appears as cuda:0 — no cross-device cuDNN issues.
# cuda:4-7 are teammate chaoqun's vLLM server — do not touch.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import faiss
import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz, process as rfprocess
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    _DRIVE = Path(DRIVE_DIR)
except NameError:
    _DRIVE = Path(".")

CONFIG = {
    "DATA_DIR":              Path("."),
    "OUTPUT_FILE":           _DRIVE / "augmented_train.csv",
    "CHECKPOINT_FILE":       _DRIVE / "augmented_train_checkpoint.csv",
    "CHECKPOINT_EVERY":      100,
    "FAISS_INDEX_FILE":      _DRIVE / "faiss_features.index",
    "FAISS_META_FILE":       _DRIVE / "faiss_metadata.parquet",
    # ── Sampling ──────────────────────────────────────────────────────────────
    "SAMPLE_SIZE":           2000,
    "RANDOM_SEED":           42,
    # ── Embedding ─────────────────────────────────────────────────────────────
    "EMBED_MODEL":           "all-MiniLM-L6-v2",
    "EMBED_BATCH_SIZE":      128,
    "TOP_K_EXAMPLES":        3,
    # ── LLM ───────────────────────────────────────────────────────────────────
    # Qwen3-8B cached locally — fp16 = 16 GB VRAM
    "LLM_MODEL":             "Qwen/Qwen3-8B",
    "DRAFT_MODEL":           None,
    "LLM_4BIT":              False,
    "MAX_NEW_TOKENS":        128,
    "BATCH_SIZE":            32,
    # ── Device ────────────────────────────────────────────────────────────────
    # cuda:0 here = physical cuda:1 (remapped by CUDA_VISIBLE_DEVICES above)
    "CUDA_DEVICE":           "cuda:0",
    # ── Span matching ─────────────────────────────────────────────────────────
    "FUZZY_SCORE_CUTOFF":    72,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

for _noisy in ("httpx", "httpcore", "huggingface_hub.utils._http",
                "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

print("✓ Imports and CONFIG loaded")
print(f"  CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}  (physical cuda:1 → appears as cuda:0)")
print(f"  LLM_MODEL  = {CONFIG['LLM_MODEL']}")
print(f"  SAMPLE_SIZE= {CONFIG['SAMPLE_SIZE']} notes → ~{CONFIG['SAMPLE_SIZE']*14:,} pairs")
print(f"  BATCH_SIZE = {CONFIG['BATCH_SIZE']} | LLM_4BIT={CONFIG['LLM_4BIT']}")


# ## Section 1 — Data Loading & Filtering
# 
# Loads `train.csv`, `features.csv`, `patient_notes.csv`.
# 
# Filters out 1 000 already-annotated notes from `train.csv`.
# Randomly samples `SAMPLE_SIZE` (default 2000) from the remaining ~41 000 unannotated notes.
# 

# In[6]:


def load_and_filter_data(cfg: dict) -> tuple:
    """
    Returns
    -------
    sample_notes : pd.DataFrame  — unannotated notes sampled for labelling
    train_df     : pd.DataFrame  — train.csv with annotation/location as Python lists
    features_df  : pd.DataFrame  — features.csv
    pn_df        : pd.DataFrame  — full patient_notes.csv (needed for FAISS metadata)
    """
    log.info("Loading CSV files ...")
    data_dir    = cfg["DATA_DIR"]
    train_df    = pd.read_csv(data_dir / "train.csv")
    features_df = pd.read_csv(data_dir / "features.csv")
    pn_df       = pd.read_csv(data_dir / "patient_notes.csv")

    def safe_parse_list(val):
        if pd.isna(val):
            return []
        try:
            result = ast.literal_eval(str(val))
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []

    train_df["annotation"] = train_df["annotation"].apply(safe_parse_list)
    train_df["location"]   = train_df["location"].apply(safe_parse_list)

    annotated_pn_nums = set(train_df["pn_num"].unique())
    log.info(f"  Annotated notes in train.csv : {len(annotated_pn_nums)}")

    unannotated = pn_df[
        ~pn_df["pn_num"].isin(annotated_pn_nums)
        & pn_df["pn_history"].notna()
        & (pn_df["pn_history"].str.strip() != "")
    ].copy()
    log.info(f"  Unannotated notes available  : {len(unannotated)}")

    n_sample     = min(cfg["SAMPLE_SIZE"], len(unannotated))
    sample_notes = unannotated.sample(n=n_sample, random_state=cfg["RANDOM_SEED"]).reset_index(drop=True)
    log.info(f"  Sampled for labelling        : {len(sample_notes)}")

    return sample_notes, train_df, features_df, pn_df

print("✓ Section 1: load_and_filter_data defined")


# ## Section 2 — FAISS Vector Index (Few-Shot Retrieval)
# 
# Builds `IndexFlatIP` (cosine similarity) over 9 901 annotated train examples.
# Each vector encodes `"Feature: <text>  Annotation: <text>"`.
# 
# For each `(note × feature)` pair: retrieves top-3 similar annotated examples as few-shot context.
# 
# > **Cache**: index saved to `faiss_features.index` + `faiss_metadata.parquet` on first build.
# > Subsequent runs load from disk in seconds.
# 

# In[7]:


def build_faiss_index(train_df, pn_df, features_df, cfg) -> tuple:
    idx_path  = cfg["FAISS_INDEX_FILE"]
    meta_path = cfg["FAISS_META_FILE"]

    if idx_path.exists() and meta_path.exists():
        log.info("Loading cached FAISS index ...")
        index    = faiss.read_index(str(idx_path))
        metadata = pd.read_parquet(meta_path).to_dict("records")
        log.info(f"  Loaded {index.ntotal} vectors (dim={index.d})")
        return index, metadata

    log.info("Building FAISS index from train.csv ...")
    pn_map   = pn_df.set_index("pn_num")["pn_history"].to_dict()
    feat_map = features_df.set_index(["case_num", "feature_num"])["feature_text"].to_dict()

    embed_texts, metadata = [], []
    for _, row in train_df.iterrows():
        feature_text   = feat_map.get((row["case_num"], row["feature_num"]), "")
        pn_history     = pn_map.get(row["pn_num"], "")
        annotation_str = " | ".join(a for a in row["annotation"] if isinstance(a, str) and a.strip())
        if not feature_text or not annotation_str:
            continue
        embed_texts.append(f"Feature: {feature_text}  Annotation: {annotation_str}")
        metadata.append({
            "feature_text": feature_text,
            "annotation":   annotation_str,
            "pn_history":   (pn_history or "")[:500],
            "location":     str(row["location"]),
        })

    cuda_device = cfg.get("CUDA_DEVICE", "cuda:0")
    log.info(f"  Embedding {len(embed_texts)} train examples on {cuda_device} ...")
    embed_model = SentenceTransformer(cfg["EMBED_MODEL"], device=cuda_device)
    embeddings  = embed_model.encode(
        embed_texts, batch_size=cfg["EMBED_BATCH_SIZE"],
        show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    del embed_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"  FAISS index: {index.ntotal} vectors, dim={dim}")
    faiss.write_index(index, str(idx_path))
    pd.DataFrame(metadata).to_parquet(meta_path, index=False)
    log.info("  Index cached to disk.")
    return index, metadata


def retrieve_few_shot_examples(query_feature_text, index, metadata, embed_model, top_k=3) -> list:
    query_vec = embed_model.encode(
        [f"Feature: {query_feature_text}  Annotation:"],
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)
    return [metadata[i] for i in indices[0] if 0 <= i < len(metadata)]

print("✓ Section 2: build_faiss_index, retrieve_few_shot_examples defined")


# ## Section 3 — Prompt Construction
# 
# Each prompt = 2-message chat:
# 1. **System**: extract verbatim spans as JSON only
# 2. **User**: 3 FAISS few-shot examples + target note + target feature
# 
# Thinking suppressed via `enable_thinking=False` in `apply_chat_template` — Qwen3 native API.
# No `<think>` blocks generated → clean JSON output, faster inference.
# 

# In[8]:


SYSTEM_PROMPT = (
    "You are a clinical NLP specialist. "
    "Given a patient note and a clinical feature, extract the EXACT verbatim text spans "
    "from the note that express that feature. "
    "Rules:\n"
    "  1. Copy text character-for-character — do NOT paraphrase.\n"
    "  2. If the feature is absent from the note, return an empty list.\n"
    "  3. Output ONLY valid JSON — no markdown, no explanation, no <think> blocks.\n"
    'Output format: {"spans": ["exact text 1", "exact text 2"]}'
)


def build_messages(feature_text, pn_history, few_shot_examples) -> list:
    examples_block = ""
    for i, ex in enumerate(few_shot_examples, start=1):
        note_excerpt = ex["pn_history"][:300].replace("\n", " ").strip()
        ann_parts    = [a.strip() for a in ex["annotation"].split(" | ") if a.strip()]
        ann_json     = json.dumps(ann_parts)  # valid JSON with double-quoted strings
        examples_block += (
            f"\n[Example {i}]\n"
            f"Note (excerpt): \"{note_excerpt}\"\n"
            f"Feature: {ex['feature_text']}\n"
            f'Answer: {{"spans": {ann_json}}}\n'
        )

    target_note  = pn_history.replace("\n", " ").strip()
    user_content = (
        f"Here are labelled examples:{examples_block}\n"
        f"---\n"
        f"Now label this note.\n"
        f"Note: \"{target_note}\"\n"
        f"Feature: {feature_text}\n\n"
        ""  # thinking suppressed via enable_thinking=False in apply_chat_template
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

print("✓ Section 3: SYSTEM_PROMPT, build_messages defined")


# ## Section 4 — Span → Character Position Mapping
# 
# LLM outputs text strings e.g. `"substernal pressure"`.
# Competition needs character offsets e.g. `"42 62"`.
# 
# **Three-step matching** (most accurate first):
# 1. Exact substring match
# 2. Case-insensitive exact match
# 3. `rapidfuzz` sliding window — handles minor spacing/casing differences
# 

# In[9]:


def _strip_think_tokens(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def find_span_locations(span_texts, pn_history, fuzzy_cutoff=72) -> list:
    if not span_texts or not pn_history:
        return []

    locations, pn_lower = [], pn_history.lower()

    for span in span_texts:
        span = span.strip()
        if not span:
            continue

        # 1. Exact match
        idx = pn_history.find(span)
        if idx != -1:
            locations.append(f"{idx} {idx + len(span)}")
            continue

        # 2. Case-insensitive
        idx = pn_lower.find(span.lower())
        if idx != -1:
            locations.append(f"{idx} {idx + len(span)}")
            continue

        # 3. Fuzzy sliding window
        span_len = len(span)
        min_win  = max(1, int(span_len * 0.80))
        max_win  = min(len(pn_history), int(span_len * 1.20))

        best_score, best_start, best_end = 0, -1, -1
        for win_size in range(min_win, max_win + 1):
            n_windows  = len(pn_history) - win_size + 1
            if n_windows <= 0:
                continue
            candidates = [pn_history[s: s + win_size] for s in range(n_windows)]
            result = rfprocess.extractOne(span, candidates, scorer=fuzz.ratio, score_cutoff=fuzzy_cutoff)
            if result is not None:
                _text, score, pos = result
                if score > best_score:
                    best_score, best_start, best_end = score, pos, pos + win_size

        if best_score >= fuzzy_cutoff and best_start != -1:
            locations.append(f"{best_start} {best_end}")

    return locations

print("✓ Section 4: _strip_think_tokens, find_span_locations defined")


# ## Section 5 — LLM Initialisation
# 
# Loads `Qwen/Qwen3-8B` in **fp16**.
# 
# - VRAM: ~16 GB on physical cuda:1 (mapped to cuda:0 via `CUDA_VISIBLE_DEVICES=1`)
# - `device_map={"": "cuda:0"}` — pins entire model to one GPU, no cross-device issues
# - `attn_implementation="eager"` — bypasses SDPA/cuDNN which fails on this system's driver
# - `GenerationConfig` set explicitly — suppresses stale temperature/top_p/top_k warnings
# - `DRAFT_MODEL=None` — speculative decoding disabled (overhead > gain for short JSON outputs)
# - Set `LLM_4BIT=True` in CONFIG to use 4-bit NF4 (e.g. on Colab T4 with limited VRAM)
# 

# In[10]:


def init_llm(cfg: dict):
    """Load model pinned to cfg['CUDA_DEVICE']. Uses attn_implementation='eager' to avoid
    cuDNN SDPA failures (CUDNN_STATUS_NOT_INITIALIZED) on this system."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

    cuda_device = cfg.get("CUDA_DEVICE", "cuda:0")
    cuda_idx    = int(cuda_device.split(":")[-1])

    def _vram_gb():
        return torch.cuda.memory_allocated(cuda_idx) / 1024**3

    log.info(f"Loading tokenizer: {cfg['LLM_MODEL']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["LLM_MODEL"], trust_remote_code=True)

    def _load_model(model_id, label):
        if cfg.get("LLM_4BIT", False) and torch.cuda.is_available():
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            m = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                device_map={"": cuda_device},
                attn_implementation="eager",  # bypass cuDNN SDPA — CUDNN_STATUS_NOT_INITIALIZED on this system
                trust_remote_code=True,
            )
        else:
            m = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map={"": cuda_device},
                attn_implementation="eager",  # bypass cuDNN SDPA — CUDNN_STATUS_NOT_INITIALIZED on this system
                trust_remote_code=True,
            )
        m.eval()
        m.generation_config = GenerationConfig(pad_token_id=tokenizer.eos_token_id)
        log.info(f"  {label} loaded on {cuda_device} | VRAM used: {_vram_gb():.1f} GB")
        return m

    log.info(f"Loading target model on {cuda_device} ...")
    model = _load_model(cfg["LLM_MODEL"], "Target")

    draft_model = None
    if cfg.get("DRAFT_MODEL"):
        log.info(f"Loading draft model: {cfg['DRAFT_MODEL']} ...")
        try:
            draft_model = _load_model(cfg["DRAFT_MODEL"], "Draft")
            log.info("✓ Speculative decoding enabled")
        except Exception as e:
            log.warning(f"Draft model failed ({e}) — standard decoding")
            draft_model = None
    else:
        log.info("  Speculative decoding disabled (DRAFT_MODEL=None)")

    free_gb  = torch.cuda.mem_get_info(cuda_idx)[0] / 1024**3
    total_gb = torch.cuda.mem_get_info(cuda_idx)[1] / 1024**3
    log.info(f"✓ Models ready | {cuda_device} VRAM: {_vram_gb():.1f} GB used, {free_gb:.1f} GB free / {total_gb:.1f} GB")
    return model, tokenizer, draft_model


def make_sampling_params(cfg: dict) -> dict:
    return {
        "max_new_tokens": cfg["MAX_NEW_TOKENS"],
        "do_sample":      False,
    }

print("✓ Section 5: init_llm defined (attn_implementation=eager)")


# ## Section 6 — Main Generation Loop
# 
# **Batched inference** (32 pairs per GPU call):
# 
# 1. For each `(unannotated note × feature)` pair:
#    - Retrieve 3 FAISS few-shot examples
#    - Build prompt (system + few-shot + full note + feature)
#    - Accumulate into batch of 32, run `model.generate()` (greedy, `do_sample=False`)
#    - Parse JSON spans, map to character offsets via exact/fuzzy matching
#    - Skip pairs already in checkpoint
# 
# 2. **Checkpoint**: saved atomically every 100 rows to `augmented_train_checkpoint.csv`
#    - Interrupt and rerun `main()` anytime — resumes automatically
# 
# 3. **Prompt context**: `max_length=4096` tokens — fits system prompt + 3 few-shot examples + full clinical note
# 
# 4. **On completion**: `augmented_train.csv` written to current directory
# 

# In[ ]:


def _load_checkpoint(cfg: dict) -> tuple:
    ckpt = Path(cfg["CHECKPOINT_FILE"])
    if ckpt.exists():
        df = pd.read_csv(ckpt)
        log.info(f"  Checkpoint found: {len(df)} rows already done — resuming.")
        return set(df["id"].tolist()), df.to_dict("records")
    return set(), []


def _save_checkpoint(rows: list, cfg: dict):
    dst = Path(cfg["CHECKPOINT_FILE"])
    tmp = dst.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(dst)


def _infer_batch(msgs_list: list, model, tokenizer, sampling_params: dict) -> list:
    from transformers import GenerationConfig

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for msgs in msgs_list
    ]

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True,
        max_length=4096,   # was 1024 — system+few-shot+full note needs up to 2K tokens; Qwen3-8B supports 128K
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    gen_cfg = GenerationConfig(
        max_new_tokens=sampling_params["max_new_tokens"],
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    out_ids = None
    try:
        with torch.no_grad():
            out_ids = model.generate(**inputs, generation_config=gen_cfg)
        return [
            tokenizer.decode(out_ids[i][input_len:], skip_special_tokens=True)
            for i in range(len(msgs_list))
        ]
    finally:
        del inputs
        if out_ids is not None:
            del out_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_spans(raw_output: str) -> list:
    if not raw_output:
        return []
    try:
        text = _strip_think_tokens(raw_output.strip())
        # Extract first JSON object — model sometimes repeats output
        m = re.search(r'\{[^{}]*"spans"\s*:\s*\[[^\]]*\][^{}]*\}', text, re.DOTALL)
        if m:
            text = m.group(0)
        return [s for s in json.loads(text).get("spans", [])
                if isinstance(s, str) and s.strip()]
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []


def generate_pseudo_labels(
    sample_notes, train_df, features_df,
    faiss_index, faiss_metadata, model, tokenizer, sampling_params, cfg,
    draft_model=None,
) -> pd.DataFrame:
    feat_map      = features_df.set_index(["case_num", "feature_num"])["feature_text"].to_dict()
    case_features = features_df.groupby("case_num")["feature_num"].agg(list).to_dict()
    batch_size    = cfg.get("BATCH_SIZE", 8)
    cuda_device   = cfg.get("CUDA_DEVICE", "cuda:0")

    done_ids, aug_rows = _load_checkpoint(cfg)
    rows_since_ckpt    = 0

    log.info(f"Loading sentence-transformer for FAISS retrieval on {cuda_device} ...")
    embed_model = SentenceTransformer(cfg["EMBED_MODEL"], device=cuda_device)

    total_pairs = sum(
        len(case_features.get(int(r["case_num"]), []))
        for _, r in sample_notes.iterrows()
        if isinstance(r["pn_history"], str) and r["pn_history"].strip()
    )
    log.info(f"  Total pairs: {total_pairs} | already done: {len(done_ids)} | batch_size: {batch_size}")

    pbar = tqdm(total=total_pairs, initial=len(done_ids), desc="Generating", unit="pair")

    pending = []

    def _flush_batch(items: list):
        nonlocal rows_since_ckpt
        if not items:
            return
        msgs_list = [item["msgs"] for item in items]
        try:
            outputs = _infer_batch(msgs_list, model, tokenizer, sampling_params)
        except Exception as exc:
            log.warning(f"  Batch inference failed ({exc}) — storing empty spans for {len(items)} pairs")
            outputs = [""] * len(items)

        for item, output in zip(items, outputs):
            spans     = _parse_spans(output)
            locations = find_span_locations(spans, item["pn_history"], cfg["FUZZY_SCORE_CUTOFF"])
            aug_rows.append({
                "id":          item["row_id"],
                "pn_num":      item["pn_num"],
                "feature_num": item["feature_num"],
                "case_num":    item["case_num"],
                "annotation":  str(spans)     if spans     else "[]",
                "location":    str(locations) if locations else "[]",
            })
            done_ids.add(item["row_id"])
            rows_since_ckpt += 1
            pbar.update(1)

        if rows_since_ckpt >= cfg["CHECKPOINT_EVERY"]:
            _save_checkpoint(aug_rows, cfg)
            rows_since_ckpt = 0

    with logging_redirect_tqdm():
        for _, note_row in sample_notes.iterrows():
            pn_num     = int(note_row["pn_num"])
            case_num   = int(note_row["case_num"])
            pn_history = note_row["pn_history"]
            if not isinstance(pn_history, str) or not pn_history.strip():
                continue

            for feature_num in case_features.get(case_num, []):
                feature_text = feat_map.get((case_num, feature_num), "")
                if not feature_text:
                    continue

                row_id = f"{pn_num:05d}_{feature_num:03d}"
                if row_id in done_ids:
                    # already done — don't double-count in pbar (initial already accounts for these)
                    continue

                try:
                    few_shot = retrieve_few_shot_examples(
                        feature_text, faiss_index, faiss_metadata, embed_model, cfg["TOP_K_EXAMPLES"]
                    )
                    msgs = build_messages(feature_text, pn_history, few_shot)
                except Exception as exc:
                    log.warning(f"  Skipped {row_id} (prompt build failed): {exc}")
                    continue

                pending.append({
                    "row_id": row_id, "pn_num": pn_num, "feature_num": feature_num,
                    "case_num": case_num, "pn_history": pn_history, "msgs": msgs,
                })

                if len(pending) >= batch_size:
                    _flush_batch(pending)
                    pending = []

        _flush_batch(pending)
        pending = []

    pbar.close()
    del embed_model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    _save_checkpoint(aug_rows, cfg)

    aug_df    = pd.DataFrame(aug_rows)
    non_empty = (aug_df["location"] != "[]").sum()
    fill_rate = 100.0 * non_empty / max(len(aug_df), 1)
    log.info(f"Done — {len(aug_df)} rows | non-empty: {non_empty} ({fill_rate:.1f}%)")
    return aug_df

print("✓ Section 6: generate_pseudo_labels defined (max_length=4096, pbar fix)")


# In[12]:


# Pre-cleanup + initialize CUDA context on the target device
# MUST call set_device() before any GPU ops — otherwise cuDNN fails on non-default GPUs
cuda_device = CONFIG["CUDA_DEVICE"]
cuda_idx    = int(cuda_device.split(":")[-1])
torch.cuda.set_device(cuda_idx)

# Force CUDA context init (allocates context, registers cuDNN, etc.)
_ = torch.zeros(1, device=cuda_device)
del _

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize(cuda_device)

free_gb  = torch.cuda.mem_get_info(cuda_idx)[0] / 1024**3
total_gb = torch.cuda.mem_get_info(cuda_idx)[1] / 1024**3
print(f"✓ CUDA context initialized on {cuda_device}")
print(f"  VRAM: {free_gb:.1f} / {total_gb:.1f} GB free")


# In[ ]:


def main():
    cfg         = CONFIG
    model       = None
    tokenizer   = None
    draft_model = None
    try:
        print("\n" + "="*65)
        print("  PHASE 1: Pseudo-Label Generation")
        print("="*65 + "\n")

        # Step 1: Load data
        print("▶ Step 1/4 — Loading and filtering data ...")
        sample_notes, train_df, features_df, pn_df = load_and_filter_data(cfg)
        print(f"  ✓ Notes: {len(sample_notes)} | Train: {len(train_df)} | Features: {len(features_df)}")

        # Step 2: FAISS index (cached to disk)
        print("\n▶ Step 2/4 — Building / loading FAISS index ...")
        faiss_index, faiss_metadata = build_faiss_index(train_df, pn_df, features_df, cfg)
        print(f"  ✓ Index: {faiss_index.ntotal} vectors")

        # Step 3: Load model (fp16, ~16 GB)
        model_id = cfg['LLM_MODEL']
        quant    = "4-bit NF4" if cfg.get("LLM_4BIT") else "fp16"
        print(f"\n▶ Step 3/4 — Loading model ({model_id}, {quant}) ...")
        model, tokenizer, draft_model = init_llm(cfg)
        sampling_params = make_sampling_params(cfg)

        # Step 4: Generate — auto-resumes from checkpoint
        ckpt = Path(cfg["CHECKPOINT_FILE"])
        if ckpt.exists():
            done = len(pd.read_csv(ckpt))
            print(f"\n▶ Step 4/4 — Resuming from checkpoint ({done} rows already done) ...")
        else:
            print("\n▶ Step 4/4 — Generating pseudo-labels ...")
            print(f"  Checkpoint will save to: {cfg['CHECKPOINT_FILE']}")

        aug_df = generate_pseudo_labels(
            sample_notes, train_df, features_df,
            faiss_index, faiss_metadata, model, tokenizer, sampling_params, cfg,
            draft_model=draft_model,
        )

        # Save final CSV
        out_path = cfg["OUTPUT_FILE"]
        aug_df.to_csv(out_path, index=False)
        non_empty = (aug_df["location"] != "[]").sum()
        fill_rate = 100.0 * non_empty / max(len(aug_df), 1)
        print(f"\n✓ Saved → {out_path}  shape={aug_df.shape}")
        print(f"  Non-empty labels: {non_empty} ({fill_rate:.1f}%)")

        # Delete checkpoint on clean completion
        if ckpt.exists():
            ckpt.unlink()
            print("  ✓ Checkpoint removed (run complete)")

        # Auto-download (Colab only — no-op locally)
        try:
            from google.colab import files
            print("\n▶ Downloading augmented_train.csv to your machine ...")
            files.download(str(out_path))
            print("  ✓ Download triggered")
        except Exception as e:
            print(f"  (Not Colab — file saved locally at: {out_path})")

        print("\n" + "="*65)
        print("  ✓ Phase 1 complete — augmented_train.csv ready for Phase 2")
        print("="*65)

    except (KeyboardInterrupt, Exception) as e:
        print(f"\n⚠ Interrupted/error: {e}")
        print("  Progress saved to checkpoint — rerun main() to resume.")
        if not isinstance(e, KeyboardInterrupt):
            raise
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if draft_model is not None:
            del draft_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("\n  ✓ Resources cleaned up")

main()

