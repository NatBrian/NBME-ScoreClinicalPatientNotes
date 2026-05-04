#!/usr/bin/env python3
"""
Phase 5 — Full Fine-Tuning: Qwen3-1.7B

Full parameter fine-tuning (no LoRA/QLoRA) of Qwen/Qwen3-1.7B.
Saves complete model weights to ./models/qwen3_1_7b_finetuned/

Run in background:
    nohup python run_phase5.py > logs/phase5_qwen3_1_7b.log 2>&1 &

Monitor:
    tail -f logs/phase5_qwen3_1_7b.log

Override GPU at runtime:
    GPU_INDEX=1 python run_phase5.py
"""

# ── GPU SELECTION — must be set BEFORE any torch/cuda import ──────────────────
import os
import subprocess
import sys

GPU_INDEX = os.environ.get("GPU_INDEX", "2")   # GPU 2 idle by default
os.environ["CUDA_VISIBLE_DEVICES"]    = GPU_INDEX  # isolates to single GPU

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
from sklearn.model_selection import GroupKFold
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

# ── LOGGING — stdout + file ───────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/phase5_qwen3_1_7b.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── thread limits (after torch import) ────────────────────────────────────────
torch.set_num_threads(8)
torch.set_num_interop_threads(4)

# ────────────────────────────────────────────────────────────────────
import csv
import json
import psutil
import torch
from pathlib import Path
from transformers import TrainerCallback, TrainerState, TrainerControl

class ComprehensiveMetricsCallback(TrainerCallback):
    """
    Pure additive telemetry collector. Hooks into Trainer lifecycle events
    to export structured CSV logs and system resource snapshots.
    Zero modification to training dynamics, gradients, or model weights.
    """
    def __init__(self, output_dir: str):
        self.metrics_dir = Path(output_dir) / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.metrics_dir / "training_timeline.csv"
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "epoch", "loss", "eval_loss", "learning_rate",
                "grad_norm", "mean_token_accuracy", "eval_mean_token_accuracy",
                "entropy", "eval_entropy", "gpu_vram_used_gb", "gpu_vram_reserved_gb",
                "cpu_percent", "eval_runtime_s", "eval_samples_per_second"
            ])

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return

        # Capture VRAM safely
        vram_used, vram_reserved = 0.0, 0.0
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            vram_reserved = torch.cuda.memory_reserved() / 1024**3

        row = [
            state.global_step,
            state.epoch,
            logs.get("loss"),
            logs.get("eval_loss"),
            logs.get("learning_rate"),
            logs.get("grad_norm"),
            logs.get("mean_token_accuracy"),
            logs.get("eval_mean_token_accuracy"),
            logs.get("entropy"),
            logs.get("eval_entropy"),
            round(vram_used, 3),
            round(vram_reserved, 3),
            psutil.cpu_percent(interval=0.01),
            logs.get("eval_runtime"),
            logs.get("eval_samples_per_second"),
        ]

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        summary = {
            "run_id": Path(args.output_dir).name,
            "final_step": state.global_step,
            "final_epoch": state.epoch,
            "max_steps_planned": args.max_steps,
            "total_eval_steps": sum(1 for log in state.log_history if "eval_loss" in log),
            "metrics_csv": str(self.csv_path),
            "trainer_state_json": str(Path(args.output_dir).parent / "trainer_state.json")
        }
        with open(self.metrics_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG = {
    "MODEL_ID":              "Qwen/Qwen3-1.7B",
    "OUTPUT_DIR":            Path("./models/qwen3_1_7b_finetuned"),
    "DATA_DIR":              Path("."),
    "AUG_CSV":               Path("./augmented_train.csv"),
    "SEED":                  42,
    "AUG_SAMPLE_RATIO":      1.0,
    "N_FOLDS":               5,
    "VAL_FOLD":              4,
    "PER_DEVICE_BATCH_SIZE": 8,
    "GRADIENT_ACCUMULATION": 2,       # effective batch = 16
    "LEARNING_RATE":         2e-5,    
    "NUM_TRAIN_EPOCHS":      3,
    "MAX_SEQ_LENGTH":        1024,    # data p99=449 tok, max=524
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
def make_formatting_func(tokenizer):
    """LFM2.5 has no thinking mode — enable_thinking not passed."""

    def _format_single(pn_history, feature_text, asst_response) -> str:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": (
                f"Note: \"{(pn_history or '').strip()}\"\n"
                f"Feature: {feature_text or ''}"
            )},
            {"role": "assistant", "content": asst_response or '{"spans": []}'},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False
        )

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


# ── SECTION 4 — Model Loading (NO quantization, NO PEFT) ─────────────────────
def load_model_and_tokenizer(cfg: dict):
    model_id = cfg["MODEL_ID"]
    log.info(f"Loading model: {model_id} in bfloat16 (no quantization) ...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype         = torch.bfloat16,
        device_map          = {"": "cuda:0"},
        attn_implementation = "flash_attention_2",
        trust_remote_code   = False,
    )

    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("  pad_token set to eos_token")
    tokenizer.padding_side = "right"

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Total parameters:     {total_params:,} ({total_params/1e9:.2f}B)")
    log.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B) — 100%")
    log.info(f"  vocab_size={tokenizer.vocab_size}  pad='{tokenizer.pad_token}'")

    if torch.cuda.is_available():
        free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
        total_gb = torch.cuda.mem_get_info()[1] / 1024**3
        log.info(f"  VRAM after load: {total_gb - free_gb:.1f}/{total_gb:.1f} GB used")

    return model, tokenizer


# ── SECTION 5 — SFT Config ───────────────────────────────────────────────────
def build_sft_config(cfg: dict) -> SFTConfig:
    return SFTConfig(
        output_dir                   = str(cfg["OUTPUT_DIR"] / "checkpoints"),
        packing                      = False,
        max_length                   = cfg["MAX_SEQ_LENGTH"],
        num_train_epochs             = cfg["NUM_TRAIN_EPOCHS"],
        max_steps                    = cfg["MAX_STEPS"],
        per_device_train_batch_size  = cfg["PER_DEVICE_BATCH_SIZE"],
        per_device_eval_batch_size   = cfg["PER_DEVICE_BATCH_SIZE"],
        gradient_accumulation_steps  = cfg["GRADIENT_ACCUMULATION"],
        gradient_checkpointing       = True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate                = cfg["LEARNING_RATE"],
        weight_decay                 = cfg["WEIGHT_DECAY"],
        warmup_ratio                 = cfg["WARMUP_RATIO"],
        lr_scheduler_type            = cfg["LR_SCHEDULER"],
        optim                        = "adamw_torch_fused",
        bf16                         = True,
        fp16                         = False,
        eval_strategy                = cfg["EVAL_STRATEGY"],
        eval_steps                   = cfg["EVAL_STEPS"],
        save_strategy                = "steps",
        save_steps                   = cfg["SAVE_STEPS"],
        save_total_limit             = cfg["SAVE_TOTAL_LIMIT"],
        load_best_model_at_end       = False,
        logging_steps                = cfg["LOGGING_STEPS"],
        logging_dir                  = str(cfg["OUTPUT_DIR"] / "logs"),
        report_to                    = "none",
        seed                         = cfg["SEED"],
        data_seed                    = cfg["SEED"],
        remove_unused_columns        = False,
        dataloader_num_workers       = 0,
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    set_seed(cfg["SEED"])

    # ── Pre-flight ────────────────────────────────────────────────────────────
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

    # ── HF token ──────────────────────────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                    hf_token = line.split("=", 1)[1].strip()
                    os.environ["HF_TOKEN"] = hf_token
                    break
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        log.info("HF_TOKEN: set")
    else:
        log.warning("HF_TOKEN not set — may fail if Qwen3-1.7B is gated")

    # ── Package versions ──────────────────────────────────────────────────────
    import importlib.metadata as meta
    for pkg in ["transformers", "trl", "accelerate", "torch"]:
        try:
            log.info(f"  {pkg}: {meta.version(pkg)}")
        except meta.PackageNotFoundError:
            log.warning(f"  {pkg}: NOT INSTALLED")

    # ── Data files check ──────────────────────────────────────────────────────
    for f in ["train.csv", "patient_notes.csv", "features.csv"]:
        path = cfg["DATA_DIR"] / f
        if not path.exists():
            log.error(f"Missing required file: {path}")
            sys.exit(1)
        log.info(f"  {f}: found")
    if cfg["AUG_CSV"].exists():
        log.info(f"  augmented_train.csv: found")
    else:
        log.warning("  augmented_train.csv: not found — training on train.csv only")

    # ── Skip if already trained ───────────────────────────────────────────────
    done_marker = cfg["OUTPUT_DIR"] / "training_metrics.json"
    if done_marker.exists():
        log.info(f"Full fine-tuned model already exists at {cfg['OUTPUT_DIR']} — SKIPPING.")
        return

    # ── Clear VRAM ────────────────────────────────────────────────────────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
        total_gb = torch.cuda.mem_get_info()[1] / 1024**3
        log.info(f"GPU ready: {free_gb:.1f}/{total_gb:.1f} GB free")

    log.info("\n" + "="*65)
    log.info("  PHASE 5 — Full Fine-Tuning: Qwen3-1.7B")
    log.info("  (no LoRA, no quantization — all parameters trained)")
    log.info("="*65)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("▶ Step 1/4 — Loading and merging data ...")
    merged_df = load_and_merge_data(cfg)

    log.info("▶ Step 2/4 — GroupKFold split ...")
    train_dataset, val_dataset = make_train_val_datasets(merged_df, cfg)
    log.info(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("▶ Step 3/4 — Loading model ...")
    cfg["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(cfg)
    fmt_func         = make_formatting_func(tokenizer)
    sft_config       = build_sft_config(cfg)

    log.info(
        f"  SFT config: lr={cfg['LEARNING_RATE']}, epochs={cfg['NUM_TRAIN_EPOCHS']}, "
        f"eff_batch={cfg['PER_DEVICE_BATCH_SIZE'] * cfg['GRADIENT_ACCUMULATION']}"
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("▶ Step 4/4 — Training (all parameters, no LoRA/PEFT) ...")
    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        args             = sft_config,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        formatting_func  = fmt_func,
        callbacks        = [ComprehensiveMetricsCallback(str(cfg["OUTPUT_DIR"]))],
    )

    train_result = trainer.train()
    log.info(
        f"Training complete — "
        f"loss={train_result.training_loss:.4f}  steps={train_result.global_step}"
    )

    # ── Save full model ───────────────────────────────────────────────────────
    output_dir = cfg["OUTPUT_DIR"]
    log.info(f"Saving full model weights → {output_dir} ...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "model_id":      cfg["MODEL_ID"],
        "finetune_type": "full",
        "training_loss": train_result.training_loss,
        "global_step":   train_result.global_step,
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
        "learning_rate": cfg["LEARNING_RATE"],
        "epochs":        cfg["NUM_TRAIN_EPOCHS"],
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Full model saved → {output_dir.resolve()}")
    log.info("Files saved:")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / 1024**2
            log.info(f"  {p.name:40s}  {size_mb:8.1f} MB")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    log.info("Cleaning up VRAM ...")
    del trainer, model, tokenizer, fmt_func, sft_config
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_gb  = torch.cuda.mem_get_info()[0] / 1024**3
        total_gb = torch.cuda.mem_get_info()[1] / 1024**3
        log.info(f"VRAM after cleanup: {free_gb:.1f}/{total_gb:.1f} GB free")

    log.info("="*65)
    log.info(f"  Phase 5 complete — model at: {output_dir.resolve()}")
    log.info("="*65)


if __name__ == "__main__":
    main()
