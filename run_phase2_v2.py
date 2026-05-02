#!/usr/bin/env python3
"""
Phase 2 — Train SLM Ensemble (QLoRA)

Group 1 — GPU 0: Qwen3-1.7B, Qwen3-8B, LFM2.5-1.2B  (~7.3h)

Run in background:
    nohup python run_phase2_group1.py > logs/phase2_group1.log 2>&1 &

Monitor:
    tail -f logs/phase2_group1.log

Override GPU at runtime:
    GPU_INDEX=2 python run_phase2_group1.py
"""

import os
import subprocess
import sys

# ── MUST SET BEFORE ANY TORCH IMPORT ─────────────────────────────────────────
GPU_INDEX = os.environ.get("GPU_INDEX", "0")  # Group 1
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

N_THREADS = "8"
os.environ["OMP_NUM_THREADS"]        = N_THREADS
os.environ["MKL_NUM_THREADS"]        = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"]   = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"]    = N_THREADS
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import ast
import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

# ── LOGGING — stdout + file ───────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/phase2_group1.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── THREAD LIMITS (applied after torch import) ────────────────────────────────
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "DATA_DIR":              Path("."),
    "ADAPTER_ROOT":          Path("./adapters"),
    "AUG_CSV":               Path("./augmented_train.csv"),
    "SEED":                  42,
    "AUG_SAMPLE_RATIO":      1.0,
    "N_FOLDS":               5,
    "VAL_FOLD":              4,
    "LORA_R":                16,
    "LORA_ALPHA":            32,
    "LORA_DROPOUT":          0.05,
    "LORA_TARGET_MODULES":   ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
    "PER_DEVICE_BATCH_SIZE": 8,       # reduce to 4 if OOM
    "GRADIENT_ACCUMULATION": 2,       # effective batch = 8×2 = 16
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
    "MAX_STEPS":             -1,
}

MODEL_REGISTRY = [
    {
        "name":              "qwen3_1_7b",
        "model_id":          "Qwen/Qwen3-1.7B",
        "model_class":       "causal_lm",
        "compute_dtype":     torch.bfloat16,
        "fp16":              False,
        "bf16":              True,
        "adapter_dir":       Path("./adapters/qwen3_1_7b_adapter"),
        "enable_thinking":   False,
        "trust_remote_code": False,
    },
    {
        "name":              "qwen3_8b",
        "model_id":          "Qwen/Qwen3-8B",
        "model_class":       "causal_lm",
        "compute_dtype":     torch.bfloat16,
        "fp16":              False,
        "bf16":              True,
        "adapter_dir":       Path("./adapters/qwen3_8b_adapter"),
        "enable_thinking":   False,
        "trust_remote_code": False,
    },
    {
        "name":              "lfm2_5_1_2b",
        "model_id":          "LiquidAI/LFM2.5-1.2B-Instruct",
        "model_class":       "causal_lm",
        "compute_dtype":     torch.bfloat16,
        "fp16":              False,
        "bf16":              True,
        "adapter_dir":       Path("./adapters/lfm2_5_1_2b_adapter"),
        "enable_thinking":   None,
        "trust_remote_code": True,
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
    '{"spans": ["exact text 1", "exact text 2"]}'
)


# ── SECTION 1 — Data Loading ──────────────────────────────────────────────────
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

    aug_path = cfg.get("AUG_CSV", data_dir / "augmented_train.csv")
    if aug_path.exists():
        aug_df = pd.read_csv(aug_path)
        aug_df["annotation"] = aug_df["annotation"].apply(safe_parse_list)
        ratio = cfg["AUG_SAMPLE_RATIO"]
        if ratio < 1.0:
            n_sample = max(1, int(len(aug_df) * ratio))
            aug_df   = aug_df.sample(n=n_sample, random_state=cfg["SEED"])
            log.info(f"  Augmented data sampled: {n_sample} ({100*ratio:.0f}%)")
        combined = pd.concat([train_df, aug_df], ignore_index=True)
        log.info(f"  Combined: {len(train_df)} (train) + {len(aug_df)} (augmented) = {len(combined)}")
    else:
        log.warning("augmented_train.csv not found — training on train.csv only.")
        combined = train_df.copy()

    pn_map   = pn_df.set_index("pn_num")["pn_history"].to_dict()
    feat_map = features_df.set_index(["case_num", "feature_num"])["feature_text"].to_dict()

    combined["pn_history"]       = combined["pn_num"].map(pn_map).fillna("")
    combined["feature_text"]     = combined.apply(
        lambda r: feat_map.get((r["case_num"], r["feature_num"]), ""), axis=1
    )
    combined["assistant_target"] = combined["annotation"].apply(build_assistant_response)

    before   = len(combined)
    combined = combined[
        combined["pn_history"].str.strip().ne("") &
        combined["feature_text"].str.strip().ne("")
    ].reset_index(drop=True)
    log.info(f"  Dropped {before - len(combined)} empty rows. Remaining: {len(combined)}")
    return combined[["pn_num", "case_num", "feature_num", "pn_history",
                     "feature_text", "annotation", "assistant_target"]]


# ── SECTION 2 — GroupKFold Split ─────────────────────────────────────────────
def make_train_val_datasets(df: pd.DataFrame, cfg: dict) -> tuple:
    gkf       = GroupKFold(n_splits=cfg["N_FOLDS"])
    groups    = df["case_num"].values
    X         = np.arange(len(df))
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


# ── SECTION 3 — Prompt Formatting ────────────────────────────────────────────
def make_formatting_func(tokenizer, model_spec: dict):
    enable_thinking = model_spec.get("enable_thinking")

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
        if isinstance(examples["pn_history"], list):
            return [
                _format_single(
                    examples["pn_history"][i],
                    examples["feature_text"][i],
                    examples["assistant_target"][i],
                )
                for i in range(len(examples["pn_history"]))
            ]
        return _format_single(
            examples["pn_history"],
            examples["feature_text"],
            examples["assistant_target"],
        )

    return formatting_func


# ── SECTION 4 — Model & Tokenizer Loading ────────────────────────────────────
def build_bnb_config(compute_dtype: torch.dtype) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = compute_dtype,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_storage    = compute_dtype,
    )


def load_model_and_tokenizer(model_spec: dict, bnb_config: BitsAndBytesConfig):
    model_id          = model_spec["model_id"]
    trust_remote_code = model_spec.get("trust_remote_code", False)
    log.info(f"Loading model: {model_id} ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config = bnb_config,
        torch_dtype         = model_spec["compute_dtype"],
        device_map          = {"": "cuda:0"},
        attn_implementation = "eager",
        trust_remote_code   = trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("  pad_token set to eos_token")
    tokenizer.padding_side = "right"
    log.info(f"  vocab_size={tokenizer.vocab_size}  pad='{tokenizer.pad_token}'")
    return model, tokenizer


# ── SECTION 5 — LoRA Setup ────────────────────────────────────────────────────
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


# ── SECTION 6 — SFT Config ───────────────────────────────────────────────────
def build_sft_config(model_spec: dict, adapter_dir: Path, cfg: dict) -> SFTConfig:
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


# ── SECTION 7 — Train One Model ───────────────────────────────────────────────
def train_one_model(model_spec: dict, train_dataset: Dataset,
                    val_dataset: Dataset, cfg: dict) -> None:
    adapter_dir         = model_spec["adapter_dir"]
    model_name          = model_spec["name"]
    adapter_config_path = adapter_dir / "adapter_config.json"

    if adapter_config_path.exists():
        log.info(f"[{model_name}] Adapter already exists — SKIPPING.")
        return

    adapter_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"\n{'='*65}")
    log.info(f"  Training: {model_name}  ({model_spec['model_id']})")
    log.info(f"{'='*65}")

    model = tokenizer = trainer = lora_config = bnb_config = sft_config = fmt_func = None
    try:
        bnb_config       = build_bnb_config(model_spec["compute_dtype"])
        model, tokenizer = load_model_and_tokenizer(model_spec, bnb_config)

        lora_config = build_lora_config(cfg, model_spec)
        model       = apply_lora(model, lora_config)

        fmt_func   = make_formatting_func(tokenizer, model_spec)
        sft_config = build_sft_config(model_spec, adapter_dir, cfg)

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

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        metrics = {
            "model_name":    model_name,
            "model_id":      model_spec["model_id"],
            "training_loss": train_result.training_loss,
            "global_step":   train_result.global_step,
            "train_samples": len(train_dataset),
            "val_samples":   len(val_dataset),
        }
        with open(adapter_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        log.info(f"[{model_name}] Adapter saved → {adapter_dir}")

    finally:
        log.info(f"[{model_name}] Cleaning up VRAM ...")
        del trainer, model, tokenizer, fmt_func, sft_config, lora_config, bnb_config
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
            total_gb = torch.cuda.mem_get_info()[1] / 1024**3
            log.info(f"[{model_name}] VRAM after cleanup: {free_gb:.1f}/{total_gb:.1f} GB free")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    log.info(f"CUDA_VISIBLE_DEVICES = {GPU_INDEX}")
    log.info(f"CPU threads: OMP/MKL/OpenBLAS = {N_THREADS}, torch intra=8 inter=4")

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
         "--format=csv,noheader", f"--id={GPU_INDEX}"],
        capture_output=True, text=True,
    )
    idx, used, free = result.stdout.strip().split(", ")
    used_mib = int(used.replace(" MiB", ""))
    if used_mib > 1000:
        log.warning(f"GPU {GPU_INDEX} already has {used} used — consider switching GPU_INDEX")
    else:
        log.info(f"GPU {GPU_INDEX}: {used} used / {free} free — OK")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        log.info("HF_TOKEN: set")
    else:
        log.warning("HF_TOKEN not set — llama3_1_8b will fail (gated model)")

    import importlib.metadata as meta
    for pkg in ["transformers", "trl", "peft", "bitsandbytes", "accelerate"]:
        try:
            log.info(f"  {pkg}: {meta.version(pkg)}")
        except meta.PackageNotFoundError:
            log.warning(f"  {pkg}: NOT INSTALLED")

    for f in ["train.csv", "patient_notes.csv", "features.csv"]:
        path = cfg["DATA_DIR"] / f
        if not path.exists():
            log.error(f"Missing required file: {path}")
            sys.exit(1)
        log.info(f"  {f}: found")

    aug_path = cfg["AUG_CSV"]
    if aug_path.exists():
        log.info(f"  augmented_train.csv: found")
    else:
        log.warning("  augmented_train.csv: not found — training on train.csv only")

    # ── Clear VRAM ────────────────────────────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
        total_gb = torch.cuda.mem_get_info()[1] / 1024**3
        log.info(f"GPU ready: {free_gb:.1f}/{total_gb:.1f} GB free")

    log.info("\n" + "="*65)
    log.info("  PHASE 2 — Group 1: Qwen3-1.7B | Qwen3-8B | LFM2.5-1.2B")
    log.info(f"  Models: {[m['name'] for m in MODEL_REGISTRY]}")
    log.info("="*65)

    # ── Load data once — shared across all models ─────────────────────────────
    log.info("▶ Step 1/3 — Loading and merging data ...")
    merged_df = load_and_merge_data(cfg)

    log.info("▶ Step 2/3 — GroupKFold split ...")
    train_dataset, val_dataset = make_train_val_datasets(merged_df, cfg)
    log.info(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    cfg["ADAPTER_ROOT"].mkdir(parents=True, exist_ok=True)

    # ── Train sequentially ────────────────────────────────────────────────────
    log.info(f"▶ Step 3/3 — Training {len(MODEL_REGISTRY)} models sequentially ...")
    for i, model_spec in enumerate(MODEL_REGISTRY):
        model_spec["adapter_dir"] = Path(model_spec["adapter_dir"]).resolve()
        log.info(f"\n  Model {i+1}/{len(MODEL_REGISTRY)}: {model_spec['name']}")
        train_one_model(model_spec, train_dataset, val_dataset, cfg)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "="*65)
    log.info("  Phase 2 complete — adapter summary:")
    for m in MODEL_REGISTRY:
        exists = (Path(m["adapter_dir"]) / "adapter_config.json").exists()
        log.info(f"    {'✓' if exists else '✗'} {m['name']:25s} → {m['adapter_dir']}")
    log.info("="*65)
    log.info(f"  Adapters saved at: {cfg['ADAPTER_ROOT'].resolve()}")


if __name__ == "__main__":
    main()
