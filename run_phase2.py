#!/usr/bin/env python
# coding: utf-8

# # Phase 2 — Train SLM Ensemble (QLoRA)
# 
# Fine-tunes three SLMs sequentially on `train.csv` + `augmented_train.csv` using 4-bit QLoRA.
# 
# | Model | Size | Precision | Adapter |
# |-------|------|-----------|---------|
# | `Qwen/Qwen3.5-9B` | 9B | BF16 | `qwen_35_9b_adapter/` |
# | `google/gemma-4-E4B-it` | 4B | BF16 | `gemma_4_e4b_adapter/` |
# | `Qwen/Qwen3.5-4B` | 4B | BF16 | `qwen_35_4b_adapter/` |
# 
# | | |
# |---|---|
# | **Output** | `adapters/` folder with 3 LoRA adapters |
# 
# > **Before running**: set `CUDA_VISIBLE_DEVICES=4` (or whichever GPU is free) in your kernel/terminal.

# In[1]:


import os

# ── GPU ISOLATION ──────────────────────────────────────────────────────────────
# Set this BEFORE any torch import to restrict to one physical GPU.
# Change "4" to whichever GPU index is free (check: !nvidia-smi).
# Must be set here in-notebook OR via `export CUDA_VISIBLE_DEVICES=4` before launching Jupyter.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
print(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

# ── HF TOKEN ───────────────────────────────────────────────────────────────────
# Accept Gemma 4 license first: https://huggingface.co/google/gemma-4-E4B-it
print("HF_TOKEN:", "set" if os.environ.get("HF_TOKEN") else "NOT SET — run: export HF_TOKEN=hf_...")


# In[2]:


# Install required packages (run once; restart kernel after)
import subprocess, sys
pkgs = [
    "transformers>=5.5.0",
    "trl>=1.0.0",
    "peft>=0.15.0",
    "bitsandbytes>=0.49.0",
    "datasets",
    "accelerate",
    "scikit-learn",
    "packaging",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
print("✓ Packages installed — restart kernel if this was the first install")


# In[3]:


import os
for f in ['features.csv', 'patient_notes.csv', 'train.csv']:
    print(f"  {f}: {'✓' if os.path.exists(f) else '✗ MISSING'}")
aug_status = '✓ found' if os.path.exists('augmented_train.csv') else '⚠ not found — will train on train.csv only'
print(f"  augmented_train.csv: {aug_status}")


# ## Configuration
# 
# | Key | Default | When to change |
# |-----|---------|----------------|
# | `AUG_SAMPLE_RATIO` | `0.15` | Set to `1.0` to use all augmented data; `0.05` for speed tests |
# | `MAX_STEPS` | `-1` | Set to `500` to cap training for Colab time limit |
# | `NUM_TRAIN_EPOCHS` | `2` | Increase for better convergence if time allows |
# | `PER_DEVICE_BATCH_SIZE` | `2` | Lower to `1` if OOM during training |

# In[4]:


import ast, gc, json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForCausalLM, AutoModelForImageTextToText,
    AutoTokenizer, BitsAndBytesConfig, set_seed,
)
from trl import SFTConfig, SFTTrainer

CONFIG = {
    "DATA_DIR":              Path("."),
    "ADAPTER_ROOT":          Path("./adapters"),
    "SEED":                  42,
    "AUG_SAMPLE_RATIO":      1.0,    # use all augmented data
    "N_FOLDS":               5,
    "VAL_FOLD":              4,
    "LORA_R":                16,
    "LORA_ALPHA":            32,
    "LORA_DROPOUT":          0.05,
    "LORA_TARGET_MODULES":   ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
    "PER_DEVICE_BATCH_SIZE": 8,      
    "GRADIENT_ACCUMULATION": 2,      # effective batch = 8×2 = 16
    "LEARNING_RATE":         2e-4,
    "NUM_TRAIN_EPOCHS":      3,
    "MAX_SEQ_LENGTH":        512,
    "WARMUP_RATIO":          0.05,
    "LR_SCHEDULER":          "cosine",
    "WEIGHT_DECAY":          0.01,
    "LOGGING_STEPS":         50,
    "SAVE_STEPS":            500,
    "EVAL_STEPS":            500,
    "EVAL_STRATEGY":         "steps",
    "SAVE_TOTAL_LIMIT":      1,
    "MAX_STEPS":             -1,     # full training (no time cap)
    "GPU_MEM_UTIL":          0.85,
}

MODEL_REGISTRY = [
    {
        "name":            "qwen_35_9b",
        "model_id":        "Qwen/Qwen3.5-9B",
        "model_class":     "causal_lm",
        "compute_dtype":   torch.bfloat16,  # native BF16
        "fp16":            False,
        "bf16":            True,
        "adapter_dir":     Path("./adapters/qwen_35_9b_adapter"),
        "enable_thinking": False,           # suppress <think> tokens in chat template
    },
    {
        "name":            "gemma_4_e4b",
        "model_id":        "google/gemma-4-E4B-it",
        "model_class":     "image_text_to_text",
        "compute_dtype":   torch.bfloat16,
        "fp16":            False,
        "bf16":            True,
        "adapter_dir":     Path("./adapters/gemma_4_e4b_adapter"),
        "enable_thinking": None,            # Gemma has no thinking mode
        "lora_target_modules": "model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",  # string = regex; skip vision_tower Gemma4ClippableLinear
    },
    {
        "name":            "qwen_35_4b",
        "model_id":        "Qwen/Qwen3.5-4B",
        "model_class":     "causal_lm",
        "compute_dtype":   torch.bfloat16,
        "fp16":            False,
        "bf16":            True,
        "adapter_dir":     Path("./adapters/qwen_35_4b_adapter"),
        "enable_thinking": False,
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

print("✓ CONFIG, MODEL_REGISTRY, SYSTEM_PROMPT loaded")


# ## Section 1 — Data Loading & Preparation
# 
# Loads `train.csv` and (optionally) `augmented_train.csv`, merges them, then joins with
# `patient_notes.csv` and `features.csv` to get the full `pn_history` and `feature_text`.
# 
# The `assistant_target` column is the JSON string the model should learn to produce:
# `{"spans": ["exact text"]}` for positive examples, `{"spans": []}` for negatives.

# In[5]:


def safe_parse_list(val) -> list:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def build_assistant_response(annotation_list: list) -> str:
    clean_spans = [s.strip() for s in annotation_list if isinstance(s, str) and s.strip()]
    return json.dumps({"spans": clean_spans}, ensure_ascii=False)


def load_and_merge_data(cfg: dict) -> pd.DataFrame:
    data_dir = cfg["DATA_DIR"]
    log.info("Loading CSVs ...")
    train_df    = pd.read_csv(data_dir / "train.csv")
    pn_df       = pd.read_csv(data_dir / "patient_notes.csv")
    features_df = pd.read_csv(data_dir / "features.csv")
    train_df["annotation"] = train_df["annotation"].apply(safe_parse_list)

    aug_path = data_dir / "augmented_train.csv"
    if aug_path.exists():
        aug_df = pd.read_csv(aug_path)
        aug_df["annotation"] = aug_df["annotation"].apply(safe_parse_list)
        ratio = cfg["AUG_SAMPLE_RATIO"]
        if ratio < 1.0:
            n_sample = max(1, int(len(aug_df) * ratio))
            aug_df   = aug_df.sample(n=n_sample, random_state=cfg["SEED"])
            log.info(f"  Augmented data sampled: {n_sample} / {len(pd.read_csv(aug_path))} ({100*ratio:.0f}%)")
        combined = pd.concat([train_df, aug_df], ignore_index=True)
        log.info(f"  Combined: {len(train_df)} (train) + {len(aug_df)} (augmented) = {len(combined)}")
    else:
        log.warning("augmented_train.csv not found — training on train.csv only.")
        combined = train_df.copy()

    pn_map   = pn_df.set_index("pn_num")["pn_history"].to_dict()
    feat_map = features_df.set_index(["case_num", "feature_num"])["feature_text"].to_dict()

    combined["pn_history"]       = combined["pn_num"].map(pn_map).fillna("")
    combined["feature_text"]     = combined.apply(lambda r: feat_map.get((r["case_num"], r["feature_num"]), ""), axis=1)
    combined["assistant_target"] = combined["annotation"].apply(build_assistant_response)

    before   = len(combined)
    combined = combined[combined["pn_history"].str.strip().ne("") & combined["feature_text"].str.strip().ne("")].reset_index(drop=True)
    log.info(f"  Dropped {before - len(combined)} empty rows. Remaining: {len(combined)}")
    return combined[["pn_num", "case_num", "feature_num", "pn_history", "feature_text", "annotation", "assistant_target"]]

print("✓ Section 1: safe_parse_list, build_assistant_response, load_and_merge_data defined")


# ## Section 2 — GroupKFold Split
# 
# Splits by `case_num` (10 unique cases → 5 folds of 2 cases each). This ensures no patient
# from the validation set appears in training — the clinically correct way to avoid leakage,
# since notes from the same case have correlated vocabulary.

# In[6]:


def make_train_val_datasets(df: pd.DataFrame, cfg: dict) -> tuple:
    gkf    = GroupKFold(n_splits=cfg["N_FOLDS"])
    groups = df["case_num"].values
    X      = np.arange(len(df))

    train_idx = val_idx = None
    for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X, groups=groups)):
        if fold == cfg["VAL_FOLD"]:
            train_idx, val_idx = tr_idx, vl_idx
            break

    if train_idx is None:
        all_idx   = set(range(len(df)))
        train_idx = np.array(sorted(all_idx - set(val_idx)))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    log.info(
        f"GroupKFold: train={len(train_df)} rows "
        f"(cases {sorted(train_df['case_num'].unique())}) | "
        f"val={len(val_df)} rows (cases {sorted(val_df['case_num'].unique())})"
    )
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)

print("✓ Section 2: make_train_val_datasets defined")


# ## Section 3 — Prompt Formatting
# 
# Converts each dataset row into a fully-formatted conversation string using the model's
# chat template. The `formatting_func` is called by `SFTTrainer` on each batch.
# 
# Loss is computed over the entire sequence (input + output) — a common choice for
# instruction-following fine-tuning when the input context is short relative to output.

# In[7]:


def make_formatting_func(tokenizer, model_spec: dict):
    enable_thinking = model_spec.get("enable_thinking")  # False for Qwen3.5, None for Gemma

    def _format_single(pn_history, feature_text, asst_response) -> str:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": (
                f"Note: \"{(pn_history or '').strip()}\"\n"
                f"Feature: {feature_text or ''}"
            )},
            {"role": "assistant", "content": asst_response or '{"spans": []}'},
        ]
        kwargs = dict(tokenize=False, add_generation_prompt=False)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return tokenizer.apply_chat_template(messages, **kwargs)

    def formatting_func(examples):
        # TRL 1.3.0 calls unbatched (scalars); older TRL calls batched (lists).
        if isinstance(examples["pn_history"], list):
            return [
                _format_single(
                    examples["pn_history"][i],
                    examples["feature_text"][i],
                    examples["assistant_target"][i],
                )
                for i in range(len(examples["pn_history"]))
            ]
        else:
            return _format_single(
                examples["pn_history"],
                examples["feature_text"],
                examples["assistant_target"],
            )

    return formatting_func

print("✓ Section 3: make_formatting_func defined")


# ## Section 4 — Model & Tokenizer Loading
# 
# Loads the base model in 4-bit NF4 quantization (QLoRA). Two model classes:
# - `AutoModelForCausalLM` for Qwen3.5-9B and Qwen3.5-4B
# - `AutoModelForImageTextToText` for Gemma 4 (multimodal arch, text-only training)
# 
# All models use `bfloat16` compute dtype — if system supports native BF16 tensor cores.
# 
# `device_map={"": "cuda:0"}` pins the model to a single GPU. Set `CUDA_VISIBLE_DEVICES`
# in your shell before launching Jupyter to control which physical GPU is used:
# ```bash
# export CUDA_VISIBLE_DEVICES=4
# jupyter notebook
# ```

# In[8]:


def build_bnb_config(compute_dtype: torch.dtype) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = compute_dtype,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_storage    = compute_dtype,
    )


def load_model_and_tokenizer(model_spec: dict, bnb_config: BitsAndBytesConfig):
    model_id    = model_spec["model_id"]
    model_class = model_spec["model_class"]
    log.info(f"Loading model: {model_id}  (class={model_class}) ...")

    load_kwargs = dict(
        pretrained_model_name_or_path = model_id,
        quantization_config           = bnb_config,
        torch_dtype                   = model_spec["compute_dtype"],
        device_map                    = {"": "cuda:0"},  # pin to single GPU (set CUDA_VISIBLE_DEVICES first)
        attn_implementation           = "eager",         # Qwen3.5 uses FLA arch; torch fallback breaks cuDNN
    )
    if model_class == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    elif model_class == "image_text_to_text":
        model = AutoModelForImageTextToText.from_pretrained(**load_kwargs)
    else:
        raise ValueError(f"Unknown model_class: {model_class!r}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("  pad_token set to eos_token")
    tokenizer.padding_side = "right"
    log.info(f"  vocab_size={tokenizer.vocab_size}  pad='{tokenizer.pad_token}'")
    return model, tokenizer

print("✓ Section 4: build_bnb_config, load_model_and_tokenizer defined")


# ## Section 5 — LoRA Adapter Setup
# 
# Injects LoRA adapters into all 7 projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
# `gate_proj`, `up_proj`, `down_proj`) with rank 16.
# 
# `prepare_model_for_kbit_training` enables gradient checkpointing and casts LayerNorm to
# float32 for stable training on top of 4-bit weights.

# In[9]:


def build_lora_config(cfg: dict, model_spec: dict = None) -> LoraConfig:
    return LoraConfig(
        r              = cfg["LORA_R"],
        lora_alpha     = cfg["LORA_ALPHA"],
        target_modules = (model_spec or {}).get("lora_target_modules") or cfg["LORA_TARGET_MODULES"],
        lora_dropout   = cfg["LORA_DROPOUT"],
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )


def apply_lora(model, lora_config: LoraConfig):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

print("✓ Section 5: build_lora_config, apply_lora defined")


# ## Section 6 — SFT Configuration
# 
# Builds the `SFTConfig` (a subclass of `TrainingArguments` from TRL v1.x).
# 
# Notable settings:
# - `optim="adamw_8bit"` — 8-bit Adam saves ~1 GB VRAM on T4
# - `group_by_length=True` — batches similar-length sequences to reduce padding waste
# - `report_to="none"` — disables W&B/TensorBoard (not needed on Colab Free)
# - `packing=False` — sequence packing off for stability

# In[10]:


def build_sft_config(model_spec: dict, adapter_dir: Path, cfg: dict) -> SFTConfig:
    # TRL 1.3.0 + transformers 5.7.0 compatible
    # max_seq_length → max_length in SFTConfig; group_by_length removed from TrainingArguments
    return SFTConfig(
        output_dir                   = str(adapter_dir / "checkpoints"),
        packing                      = False,
        max_length                   = cfg["MAX_SEQ_LENGTH"],
        num_train_epochs             = cfg["NUM_TRAIN_EPOCHS"],
        max_steps                    = cfg["MAX_STEPS"],
        per_device_train_batch_size  = cfg["PER_DEVICE_BATCH_SIZE"],
        per_device_eval_batch_size   = cfg["PER_DEVICE_BATCH_SIZE"],
        gradient_accumulation_steps  = cfg["GRADIENT_ACCUMULATION"],
        gradient_checkpointing       = True,
        learning_rate                = cfg["LEARNING_RATE"],
        weight_decay                 = cfg["WEIGHT_DECAY"],
        warmup_ratio                 = cfg["WARMUP_RATIO"],
        lr_scheduler_type            = cfg["LR_SCHEDULER"],
        optim                        = "adamw_8bit",
        fp16                         = model_spec["fp16"],
        bf16                         = model_spec["bf16"],
        eval_strategy                = cfg["EVAL_STRATEGY"],
        eval_steps                   = cfg["EVAL_STEPS"],
        save_strategy                = "steps",
        save_steps                   = cfg["SAVE_STEPS"],
        save_total_limit             = cfg["SAVE_TOTAL_LIMIT"],
        load_best_model_at_end       = False,
        logging_steps                = cfg["LOGGING_STEPS"],
        logging_dir                  = str(adapter_dir / "logs"),
        report_to                    = "none",
        seed                         = cfg["SEED"],
        data_seed                    = cfg["SEED"],
        remove_unused_columns        = False,
        dataloader_num_workers       = 0,
    )

print("✓ Section 6: build_sft_config defined")


# ## Section 7 — Training Pipeline (One Model)
# 
# Orchestrates the full QLoRA training loop for one SLM:
# 
# 1. Skip if adapter already saved (safe to re-run)
# 2. Load quantized model + tokenizer
# 3. Inject LoRA adapters
# 4. Train with `SFTTrainer`
# 5. Save adapter weights only (not the full model — they're tiny, ~100 MB)
# 6. Aggressive VRAM cleanup before the next model loads

# In[11]:


def train_one_model(model_spec: dict, train_dataset: Dataset, val_dataset: Dataset, cfg: dict) -> None:
    adapter_dir         = model_spec["adapter_dir"]
    model_name          = model_spec["name"]
    adapter_config_path = adapter_dir / "adapter_config.json"

    if adapter_config_path.exists():
        log.info(f"[{model_name}] Adapter already exists at {adapter_dir} — SKIPPING.")
        return

    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  Training: {model_name}  ({model_spec['model_id']})")
    print(f"{'='*65}")

    model = tokenizer = trainer = lora_config = bnb_config = sft_config = fmt_func = None
    try:
        log.info(f"[{model_name}] Loading quantized model ...")
        bnb_config         = build_bnb_config(model_spec["compute_dtype"])
        model, tokenizer   = load_model_and_tokenizer(model_spec, bnb_config)

        log.info(f"[{model_name}] Injecting LoRA adapters ...")
        lora_config = build_lora_config(cfg, model_spec)
        model       = apply_lora(model, lora_config)

        fmt_func   = make_formatting_func(tokenizer, model_spec)
        sft_config = build_sft_config(model_spec, adapter_dir, cfg)

        # TRL 1.3.0: SFTTrainer takes no max_seq_length — handled via SFTConfig.max_length
        trainer = SFTTrainer(
            model            = model,
            processing_class = tokenizer,
            args             = sft_config,
            train_dataset    = train_dataset,
            eval_dataset     = val_dataset,
            formatting_func  = fmt_func,
        )

        log.info(f"[{model_name}] Starting training ...")
        train_result = trainer.train()
        log.info(
            f"[{model_name}] Training complete — "
            f"loss={train_result.training_loss:.4f}  steps={train_result.global_step}"
        )

        log.info(f"[{model_name}] Saving LoRA adapter to {adapter_dir} ...")
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        metrics = {
            "model_name": model_name, "model_id": model_spec["model_id"],
            "training_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "train_samples": len(train_dataset), "val_samples": len(val_dataset),
        }
        with open(adapter_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Adapter saved → {adapter_dir}")

    finally:
        log.info(f"[{model_name}] Cleaning up VRAM ...")
        del trainer, model, tokenizer, fmt_func, sft_config, lora_config, bnb_config
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
            total_gb = torch.cuda.mem_get_info()[1] / 1024**3
            log.info(f"[{model_name}] VRAM after cleanup: {free_gb:.1f}/{total_gb:.1f} GB free")

print("✓ Section 7: train_one_model defined")


# ## Run Phase 2 — Sequential QLoRA Training
# 
# Trains all 3 models one after the other. Each model is fully
# unloaded from VRAM before the next one loads.
# 
# Expected log output per model:
# ```
# Training: qwen_35_9b  (Qwen/Qwen3.5-9B)
# Step 50/...  loss=X.XX  eval_loss=X.XX
# ...
# Training complete — loss=X.XX  steps=N
# ✓ Adapter saved → ./adapters/qwen_35_9b_adapter
# ```
# 
# **After completion**: copy `adapters/` to Kaggle as a private dataset (internet OFF inference).

# In[12]:


def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])

    import transformers
    from packaging.version import Version
    tf_ver = transformers.__version__
    if Version(tf_ver) < Version("5.5.0"):
        print(f"⚠ transformers=={tf_ver} — Gemma 4 requires >= 5.5.0.  Run: pip install -U transformers")
    else:
        print(f"✓ transformers=={tf_ver}")

    print("\n" + "="*65)
    print("  PHASE 2: SLM Ensemble QLoRA Training")
    print("="*65 + "\n")

    # Clear any residual VRAM before first model loads
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
        total_gb = torch.cuda.mem_get_info()[1] / 1024**3
        log.info(f"GPU ready: {free_gb:.1f}/{total_gb:.1f} GB free")

    print("▶ Step 1/3 — Loading and merging data ...")
    merged_df = load_and_merge_data(cfg)

    print("\n▶ Step 2/3 — GroupKFold split ...")
    train_dataset, val_dataset = make_train_val_datasets(merged_df, cfg)
    print(f"  Train: {len(train_dataset)} examples | Val: {len(val_dataset)} examples")

    cfg["ADAPTER_ROOT"].mkdir(parents=True, exist_ok=True)

    print(f"\n▶ Step 3/3 — Training {len(MODEL_REGISTRY)} models sequentially ...")
    for i, model_spec in enumerate(MODEL_REGISTRY):
        model_spec["adapter_dir"] = Path(model_spec["adapter_dir"]).resolve()
        print(f"\n  Model {i+1}/{len(MODEL_REGISTRY)}: {model_spec['name']}")
        train_one_model(model_spec, train_dataset, val_dataset, cfg)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n" + "="*65)
    print("  Phase 2 complete — adapter summary:")
    adapter_root = cfg["ADAPTER_ROOT"].resolve()
    for m in MODEL_REGISTRY:
        exists = (Path(m["adapter_dir"]) / "adapter_config.json").exists()
        print(f"    {'✓' if exists else '✗'} {m['name']:25s} → {m['adapter_dir']}")
    print("="*65)
    print(f"\n  Adapters saved at: {adapter_root}")
    print("  Next step: upload adapters/ to Kaggle as a private dataset for offline inference.")

main()

