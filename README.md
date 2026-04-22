# NBME — Score Clinical Patient Notes
## RAG-Augmented SLM Ensemble

---

## The Competition

**NBME – Score Clinical Patient Notes** is a Kaggle NLP competition where the task is to identify clinical feature expressions in free-text patient notes written by medical students.

Given:
- A **patient note** (free-text, ~50–300 words) written by a medical student describing a clinical encounter
- A **clinical feature** (short phrase, e.g. *"shortness of breath"*, *"family history of diabetes"*)

The goal is to find and return the **exact character spans** in the note that express that feature — even when the student used different phrasing than the feature label.

**Evaluation metric**: Micro-averaged character-level F1. Every character in the note is labelled 0 (outside span) or 1 (inside span). Precision and recall are computed at the character level across all notes.

---

## Our Approach — RAG-Augmented SLM Ensemble

The core challenge is **span extraction under paraphrase**: a feature like *"chest pain"* may appear as *"substernal pressure"* or *"left-sided chest tightness"* in the note. Classical NER models struggle here because the label vocabulary is small (143 features) but the surface form variation is large.

We tackle this with a three-stage pipeline:

### Stage 1 — Platinum Data Generation (Knowledge Distillation)
The competition provides only ~14,300 labelled rows across 42,146 patient notes. We use the unlabelled notes to create additional pseudo-labelled training data:

1. **FAISS few-shot retrieval**: For each (note, feature) pair we retrieve the 3 most semantically similar labelled examples from `train.csv` using a MiniLM sentence-transformer index. These examples become the few-shot context.
2. **Qwen3.5-9B-Instruct-AWQ**: A 4-bit AWQ-quantized 9B model runs via vLLM with XGrammar-constrained JSON decoding, guaranteeing valid `{"spans": [...]}` output every time. It reads the few-shot examples and extracts spans from the unlabelled note.
3. **Fuzzy span alignment**: Extracted text is mapped back to exact character offsets using exact match → case-insensitive match → rapidfuzz sliding window. This produces `augmented_train.csv` (~10,000 rows with pseudo-labels).

The 9B teacher model has enough clinical reasoning to produce high-quality pseudo-labels; the smaller student models below are trained to replicate this behaviour.

### Stage 2 — SLM Ensemble Training (QLoRA Fine-Tuning)
We fine-tune three compact SLMs on the combined `train.csv` + `augmented_train.csv`:

| Model | Size | Precision | Adapter |
|-------|------|-----------|---------|
| `Qwen/Qwen3.5-4B` | 4B | FP16 | `qwen_35_4b_adapter/` |
| `google/gemma-4-E2B-it` | 2B | BF16 | `gemma_4_e2b_adapter/` |
| `google/gemma-4-E4B-it` | 4B | BF16 | `gemma_4_e4b_adapter/` |

Each model is fine-tuned with **QLoRA** (4-bit NF4 quantization + LoRA rank 16) to fit on a single T4 16 GB GPU. Training uses `GroupKFold` by `case_num` to prevent data leakage between clinical cases.

The three models are architecturally diverse (different training data, tokenizers, and attention patterns), which is what makes their ensemble effective — errors from one model are unlikely to be shared by the others.

### Stage 3 — vLLM Ensemble Inference + Majority Voting
At inference time (Kaggle, internet OFF):

1. For each model, the LoRA adapter is **merged into the base weights on CPU** (no VRAM used during merge), then loaded into vLLM.
2. Each (note, feature) pair is decoded with a **per-note regex FSM constraint** built from the exact characters present in the note — the model physically cannot hallucinate text that isn't in the source.
3. After all 3 models predict, we apply **character-level majority voting**: a character position is included in the final span if ≥ 2 out of 3 models agree it falls inside a span. This is more robust than span-level voting because models may agree on the *region* even when their exact span boundaries differ.

```
train.csv (14,300 rows)
patient_notes.csv (42,146 rows)   Stage 1: FAISS + Qwen3.5-9B-AWQ
features.csv (143 features)    ──────────────────────────────────► augmented_train.csv (~10k rows)
                                                                           │
                               Stage 2: QLoRA fine-tuning                 │
                         train.csv + augmented_train.csv  ─────────────────► 3 LoRA adapters
                                                                           │
                               Stage 3: vLLM ensemble + char majority vote │
                         test.csv + adapters  ────────────────────────────► submission.csv
```

---

## File Overview

| Notebook | Phase | Runtime | Output |
|----------|-------|---------|--------|
| `1_generate_augmented_data.ipynb` | Pseudo-labelling | ~2–4 h on T4 | `augmented_train.csv` |
| `2_train_slm_ensemble.ipynb` | QLoRA training | ~60–90 min × 3 on T4 | `adapters/` |
| `3_kaggle_inference.ipynb` | Kaggle submission | ~2–3 h on T4 x2 | `submission.csv` |

---

## Prerequisites

### 1. Python environment

```bash
pip install \
  "vllm>=0.9.0" \
  "transformers>=5.5.0" \
  "trl>=1.0.0" \
  "peft>=0.15.0" \
  "bitsandbytes>=0.49.0" \
  sentence-transformers \
  faiss-gpu \
  rapidfuzz \
  xgrammar \
  pandas \
  numpy \
  scikit-learn \
  pydantic \
  tqdm \
  datasets \
  accelerate \
  packaging
```

> Gemma 4 requires `transformers >= 5.5.0`. Verify with `pip show transformers`.

### 2. Hugging Face access

```bash
huggingface-cli login
# Or use notebook_login() inside the notebook
```

Accept model licenses before downloading:
- https://huggingface.co/google/gemma-4-E2B-it
- https://huggingface.co/google/gemma-4-E4B-it

### 3. Competition data

```bash
kaggle competitions download -c nbme-score-clinical-patient-notes
unzip nbme-score-clinical-patient-notes.zip -d /path/to/brian/
```

Expected files in the working directory:
```
features.csv
patient_notes.csv
train.csv
test.csv
sample_submission.csv
```

---

## Running the Pipeline

### Phase 1 — Generate Platinum Data

1. Open `1_generate_augmented_data.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells (the notebook handles pip install and HF login)

**Key CONFIG options** (cell 5 of the notebook):
| Key | Default | Description |
|-----|---------|-------------|
| `SAMPLE_SIZE` | `10_000` | Unannotated notes to pseudo-label |
| `GPU_MEM_UTIL` | `0.82` | vLLM VRAM fraction; lower to `0.75` if OOM |
| `MAX_MODEL_LEN` | `4096` | Token context window; try `2048` if OOM |

**Output**: `augmented_train.csv` in the working directory. Download it — you need it for Phase 2.

---

### Phase 2 — Train the SLM Ensemble

1. Open `2_train_slm_ensemble.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Upload `augmented_train.csv` and the competition CSVs to the Colab session
4. Run all cells

**Key CONFIG options**:
| Key | Default | Description |
|-----|---------|-------------|
| `AUG_SAMPLE_RATIO` | `0.15` | Fraction of augmented data to use (1.0 = all) |
| `MAX_STEPS` | `-1` | `-1` = full epochs; set e.g. `500` for a smoke test |
| `NUM_TRAIN_EPOCHS` | `2` | Training epochs per model |

> If you hit the Colab 90-min session limit: set `AUG_SAMPLE_RATIO=0.05` and `MAX_STEPS=500`.

**Output**: `adapters/` folder with 3 subdirectories. Download the entire `adapters/` folder.

---

### Phase 3 — Kaggle Submission

#### Step A: Upload adapters as a Kaggle dataset

```bash
zip -r my-adapters.zip adapters/
```

1. Go to https://www.kaggle.com/datasets/new
2. Upload `my-adapters.zip`, name it `my-adapters`, set visibility to **Private**, click **Create**

#### Step B: Create the Kaggle notebook

1. Go to the competition: https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes
2. Click **Code** → **New Notebook**
3. In **Settings** (right panel):
   - **Accelerator**: GPU T4 x2 (or P100)
   - **Internet**: **OFF** (required for competition submission)
   - **Persistence**: Save & Run All
4. Add datasets:
   - Competition data auto-mounts at `/kaggle/input/nbme-score-clinical-patient-notes/`
   - Add your adapters: search `my-adapters` → attach → mounts at `/kaggle/input/my-adapters/`

#### Step C: Run inference

Upload `3_kaggle_inference.ipynb` as the notebook and run all cells.

The CONFIG is pre-set to Kaggle paths:
```python
"DATA_DIR":    Path("/kaggle/input/nbme-score-clinical-patient-notes"),
"ADAPTER_DIR": Path("/kaggle/input/my-adapters"),
"OUTPUT_DIR":  Path("/kaggle/working"),
```

**Output**: `/kaggle/working/submission.csv`

#### Step D: Submit

Click **Submit** in the Kaggle UI after the notebook completes.

---

## Adapter Directory Layout

```
/kaggle/input/my-adapters/
  qwen_35_4b_adapter/
    adapter_config.json
    adapter_model.safetensors
  gemma_4_e2b_adapter/
    adapter_config.json
    adapter_model.safetensors
  gemma_4_e4b_adapter/
    adapter_config.json
    adapter_model.safetensors
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| OOM during Phase 1 | Lower `GPU_MEM_UTIL` to `0.75` or `MAX_MODEL_LEN` to `2048` |
| OOM during Phase 2 | Lower `PER_DEVICE_BATCH_SIZE` to `1` or `MAX_SEQ_LENGTH` to `384` |
| OOM during Phase 3 | Lower `GPU_MEM_UTIL` to `0.80`; `ENFORCE_EAGER=True` is already set |
| Gemma import error | `pip install -U "transformers>=5.5.0"` |
| vLLM guided decoding error | Ensure `vllm>=0.9.0` and `xgrammar` is installed |
| Phase 3 missing adapters | Verify Kaggle dataset mount path matches `ADAPTER_DIR` in CONFIG |
| Colab session timeout | Set `MAX_STEPS=500` and `AUG_SAMPLE_RATIO=0.05` in Phase 2 |
| Qwen3.5 think tokens in output | Already handled: `/no_think` flag + regex strip in all notebooks |
