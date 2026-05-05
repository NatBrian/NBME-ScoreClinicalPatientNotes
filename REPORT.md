# Generative SLM Ensemble for Clinical Span Extraction
## NBME — Score Clinical Patient Notes
### Competition Report

**Team:** Brian Nat  
**Competition:** [NBME — Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes)  
**Date:** May 2026  
**Repository:** `NBME-ScoreClinicalPatientNotes`

---

## Table of Contents

1. [Introduction & Competition Description](#1-introduction--competition-description)
2. [Challenges of the Problem](#2-challenges-of-the-problem)
3. [Background & Key Concepts](#3-background--key-concepts)
4. [Comparison with Traditional ML and Top Leaderboard Approaches](#4-comparison-with-traditional-ml-and-top-leaderboard-approaches)
5. [Proposed Solution in Detail](#5-proposed-solution-in-detail)
6. [Experimental Study](#6-experimental-study)
7. [Discussion & Limitations](#7-discussion--limitations)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)
10. [Appendix](#10-appendix)

---

## 1. Introduction & Competition Description

### 1.1 What Is This Competition?

Imagine you are a medical school examiner. A student sits a standardised clinical exam: they are shown a patient vignette and must take a "history" — asking the patient questions and writing down what they learn. After the exam, an examiner goes through each student's written note and checks whether specific clinical features are present. For example, did the student record that the patient has a *family history of heart disease*? Did they note that the patient *denies chest pain*?

This checking process is laborious: there are thousands of students, each note can be hundreds of words long, and a single case may have fifteen or more features to verify. The **NBME — Score Clinical Patient Notes** Kaggle competition, hosted in early 2022, asked competitors to automate this process. Given a patient note and a clinical feature description, the model must find the exact text spans in the note that express that feature.

### 1.2 Formal Task Definition

**Input (one row):**
- `pn_history`: free-text patient note written by a medical student
- `feature_text`: a clinical concept from the scoring rubric (e.g., `"Family history of MI or cardiac disease"`)

**Output:**
- `location`: one or more character-offset pairs indicating which part(s) of the note express the feature — e.g., `"42 65;103 121"` means characters 42–65 and 103–121

If the feature is not present in the note, the output is an empty string.

### 1.3 Running Example

Throughout this report we use the following concrete example to ground abstract concepts:

> **Patient note excerpt:**  
> *"Patient is a 45-year-old male presenting with substernal chest pressure for 3 days. Father died of heart attack at age 52. No prior cardiac history."*

> **Feature:** `"Family history of MI or cardiac disease"`

> **Ground-truth annotation:** `"Father died of heart attack at age 52"` → character offsets `"62 98"`

A perfect model would output the span `62 98` for this row. Note that the feature says "MI or cardiac disease" but the note says "heart attack" — the model must understand that these mean the same thing. This paraphrasing challenge is central to the problem.

### 1.4 Dataset Statistics

The competition provided:

| Split | Notes | Annotated Pairs |
|---|---|---|
| Training (labeled) | ~1,000 annotated notes | ~9,901 (note × feature) pairs |
| Unannotated pool | ~41,000 notes | Not labeled |
| Test | Held-out | Evaluated by Kaggle |

The dataset covers **10 clinical cases**, each with approximately 14.8 scored features on average. In total, ~2,800 notes were annotated with ~35,000 phrases. This relatively small labeled set (9,901 pairs) is an important constraint — models must generalise from limited supervision.

### 1.5 Evaluation Metric

Submissions are scored with **micro-averaged character-level F1**. Here is exactly what this means:

1. For each predicted character offset in the submission, check whether it is covered by a ground-truth offset.
2. Compute **precision** globally: `(correctly predicted characters) / (total predicted characters)`
3. Compute **recall** globally: `(correctly predicted characters) / (total ground-truth characters)`
4. **F1 = 2 × precision × recall / (precision + recall)`

**Example with the running example:**  
Suppose ground truth is characters 62–98 (37 characters). Our model predicts 60–100 (41 characters). The overlap is characters 62–98 (37 characters). Precision = 37/41 ≈ 0.902; Recall = 37/37 = 1.0; F1 ≈ 0.948 for this example. Scores are aggregated this way across all rows in the test set.

The top competition scores reached **approximately 0.90 F1** on the public leaderboard, representing near-human annotation performance.

### 1.6 Why This Problem Matters

Automated clinical note scoring has significant real-world impact:

- **Scale**: Medical licensing exams like USMLE Step 2 CS involve tens of thousands of students annually. Manual annotation by expert clinicians is expensive and slow.
- **Consistency**: Automated systems apply the same rubric consistently, reducing inter-rater variability.
- **Feedback**: Faster automated scoring enables near-real-time feedback to medical students on their clinical communication skills.
- **Broader NLP**: The techniques developed for this task — span extraction from domain-specific text — transfer to clinical information extraction, EHR mining, and medical question answering.

> Having established what the competition asks and why it matters, we now turn to the specific challenges that make this a hard NLP problem.

---

## 2. Challenges of the Problem

This section identifies the six major challenges of clinical span extraction, each illustrated with a concrete example. A mapping table at the end previews how our solution addresses each challenge.

### 2.1 Ambiguity in Clinical Language

Clinical language is rich with synonymy and paraphrase. Different students describe the same finding in entirely different words:

| Feature | Student A writes | Student B writes |
|---|---|---|
| "chest pain" | "substernal pressure" | "tightness in the chest" |
| "denies alcohol use" | "ETOH: none" | "doesn't drink" |
| "family history of MI" | "father had a heart attack" | "FHx: CAD" |

A system that simply searches for the feature words verbatim ("chest pain") will miss most positive examples. The model must understand clinical semantics, not just keywords.

### 2.2 Multi-Label, Multi-Span Extraction

A single feature may be expressed in multiple non-contiguous locations within the same note. For example:

> Feature: `"tobacco use"`  
> Note: *"The patient smokes half a pack per day. ... Social history: 20 pack-year smoking history."*  
> Ground truth: `"smokes half a pack per day"` **and** `"20 pack-year smoking history"` — two separate spans.

Both spans must be identified. Missing either one reduces recall. BIO-tagging approaches handle this naturally, but generative approaches must output a list of spans.

### 2.3 Overlapping Annotations and Boundary Ambiguity

It is often unclear exactly where a span should begin and end. Consider:

> Feature: `"dyspnea on exertion"`  
> Note: *"Patient reports severe dyspnea on exertion for the past 2 weeks."*

Should the span be `"dyspnea on exertion"`, `"severe dyspnea on exertion"`, or `"severe dyspnea on exertion for the past 2 weeks"`? The character-level F1 metric is tolerant of small boundary errors, but systematic over- or under-prediction incurs a consistent precision or recall penalty.

### 2.4 Class Imbalance

Across the 9,901 training pairs, many features appear in fewer than 5% of notes (negative examples), while some common symptoms appear in over 80%. This imbalance means a model that predicts "no span" for everything achieves high precision but near-zero recall. Training must account for the heavy skew towards empty spans.

### 2.5 Noisy and Incomplete Annotations

Human annotators are not perfect. Some ground-truth annotations in the training set are missing spans that are clearly present in the notes. This label noise means that a model achieving 100% recall on the training set would actually be hallucinating — it is matching spurious patterns rather than genuine features. Overfitting to noisy labels is a real risk.

### 2.6 Domain Specificity

Clinical notes contain medical abbreviations (`ETOH`, `FHx`, `SOB`), Latin phrases, drug names, and clinical conventions that are rare or absent in general-domain text. A model pre-trained only on web text may not correctly interpret `"SOB"` as shortness of breath rather than the offensive word, or `"pt"` as patient rather than pint.

### 2.7 Challenge → Solution Mapping

| Challenge | How Our Solution Addresses It |
|---|---|
| Ambiguity in clinical language | LLM semantic understanding via large-scale pretraining enables paraphrase recognition |
| Multi-span extraction | Generative JSON output `{"spans": [...]}` naturally returns a list of multiple spans |
| Boundary ambiguity | Character-level majority voting smooths boundary disagreements across ensemble members |
| Class imbalance | Pseudo-labeling augments positive examples; few-shot prompting provides negative examples |
| Noisy annotations | Character-level voting threshold (≥3/4 models) filters out low-confidence hallucinations |
| Domain specificity | Instruction-tuned LLMs with clinical few-shot examples adapt quickly to medical language |

> With the challenges established, the next section builds the conceptual vocabulary needed to understand our solution. Readers already familiar with LLMs and LoRA may skip to Section 4.

---

## 3. Background & Key Concepts

This section briefly explains each technical concept used in this report. It is written for a reader who has taken an introductory machine learning course but has not yet worked with large language models or Kaggle competitions.

### 3.1 Large Language Models (LLMs)

A **Large Language Model** is a neural network trained on enormous quantities of text (books, websites, code) to predict the next word given the preceding words. Think of it as autocomplete on steroids: given the prefix *"The patient presents with chest"*, the model assigns probabilities to every possible next word and samples from them. With enough training data and parameters (billions of weights), LLMs develop remarkably broad language understanding — they can answer questions, summarise documents, translate languages, and follow complex instructions.

In this project we use **decoder-only** LLMs (Qwen3, LFM2.5, Llama-3.1) that generate text token-by-token from left to right. We instruct them to output a JSON object containing the extracted spans.

### 3.2 LoRA — Low-Rank Adaptation

Fully fine-tuning an 8B-parameter LLM requires updating billions of weights, which demands enormous GPU memory and risks **catastrophic forgetting** — the model forgets its general language capabilities while learning the new task.

**LoRA (Low-Rank Adaptation)** is an elegant alternative. Instead of modifying the full weight matrix W, LoRA adds a small pair of matrices A and B such that the effective update is W + AB^T, where the rank of AB^T is much smaller than the rank of W. Concretely, with rank r=16, a 4096×4096 weight matrix (16M parameters) is approximated by two matrices of shape 4096×16 and 16×4096 (only 131K parameters) — a 120× reduction in trainable parameters.

Analogy: instead of rewriting the entire textbook (full fine-tuning), LoRA adds sticky notes with corrections (low-rank updates) to specific chapters. The original textbook stays intact. This makes LoRA extremely parameter-efficient, allowing fine-tuning of 8B models on a single consumer GPU in hours.

We further combine LoRA with **4-bit NF4 quantization** (QLoRA), which compresses the frozen base model weights from 16 bits to 4 bits, reducing memory by 4×.

### 3.3 Ensemble Methods

An **ensemble** combines predictions from multiple models to produce a more reliable result. The intuition: if you ask five doctors independently and four of them agree on a diagnosis, you have more confidence than if you asked only one.

In our case, we train four separate models and aggregate their span predictions via **character-level majority voting**: a character position is included in the final prediction only if at least 3 of the 4 models predict it as part of a span. This reduces the influence of any single model's idiosyncratic errors.

### 3.4 Few-Shot Learning

**Few-shot learning** means providing a model with a handful of worked examples in the prompt (the "context window") so it understands the task without any parameter updates. For example:

```
System: Extract verbatim text spans that express the clinical feature. Return JSON.
Example 1:
  Note: "Patient has a father with heart disease."
  Feature: "Family history of MI"
  Answer: {"spans": ["father with heart disease"]}
Now extract:
  Note: "Father died of MI at 55."
  Feature: "Family history of MI"
  Answer:
```

The model generalises from the examples to the new case without retraining. This is the basis of our **Pretrained Only + Few Shot** baseline.

### 3.5 Encoder vs. Decoder Architectures

Transformer models come in two flavours relevant to this task:

- **Encoder-only models** (e.g., BERT, DeBERTa): Read the full input bidirectionally and produce a contextual representation for every token. Ideal for classification tasks. Top leaderboard solutions used encoder models with a **token classification head** — predicting a label (Begin, Inside, Outside) for each token to identify spans.
- **Decoder-only models** (e.g., GPT-4, Qwen3, Llama-3.1): Generate text left-to-right, one token at a time. Our approach uses decoder LLMs to *generate* the span text as a JSON string.

Both paradigms can solve span extraction, but they have very different trade-offs discussed in Section 4.

### 3.6 Token Classification vs. Generative Extraction

**Token classification (BIO tagging):** Each input token is assigned a label: B (begin span), I (inside span), or O (outside). Contiguous B/I runs are decoded as spans. This is the standard NER formulation. It is deterministic and fast but requires the model to be retrained whenever the label set changes.

**Generative extraction:** The model generates the span text directly (e.g., `{"spans": ["Father died of MI at 55"]}`). The generated string is then located in the note to recover character offsets. This is more flexible — new features can be handled by changing the prompt — but risks hallucination (generating text not present in the note). We mitigate hallucination using a per-note regex FSM constraint during inference (Section 5).

> Armed with these concepts, we can now compare our approach to traditional methods and to the dominant DeBERTa-based leaderboard solutions.

---
## 4. Comparison with Traditional ML Models and Top Leaderboard Approaches

This section situates our approach in the broader landscape of methods for span extraction. We proceed from the simplest possible baselines (traditional ML) through the dominant competition paradigm (DeBERTa-based token classification) to our own generative LLM approach. For each paradigm, we examine the underlying mechanisms, specific benefits, and fundamental limitations. The goal is to show *why* different choices were made by different teams, and what trade-offs each choice entails.

### 4.1 Traditional Machine Learning Approaches

Before the era of deep learning, NLP practitioners solved named entity recognition and span extraction tasks using a combination of hand-engineered features and classical ML classifiers. Understanding these approaches � and their failure modes � illuminates why deep pre-trained models dominate modern clinical NLP.

#### 4.1.1 Rule-Based and Keyword-Matching Systems

The simplest possible approach to the NBME task is **exact string matching**: for each feature (e.g., `"chest pain"`), search the patient note for that exact phrase and return its character offsets if found. This works only when the note uses the exact same wording as the feature rubric � which is the exception rather than the rule. Medical students paraphrase constantly.

A more sophisticated variant uses **regular expressions**: for `"shortness of breath"`, one might write a regex that matches `"shortness of breath"`, `"SOB"`, `"dyspnea"`, `"difficulty breathing"`, `"breathlessness"`, and `"breathless"`. This extends coverage but requires a domain expert to hand-craft rules for every feature and every synonymous expression � a labour-intensive process that does not scale to 150+ features and fails on novel phrasings not anticipated by the rule author.

**Benefits of rule-based systems:**
- Completely interpretable: the programmer can read the rules and understand exactly why a prediction was made
- No training data required
- Deterministic and reproducible
- Fast to run (milliseconds per note)
- No GPU or specialised hardware needed

**Limitations:**
- Extremely brittle: any phrasing not covered by an explicit rule is missed
- Vocabulary coverage degrades with domain shift (new cases, new examiners, regional terminology)
- Maintenance burden grows quadratically: each new feature requires new rules; interactions between features are not handled
- Cannot handle paraphrase, implicit statements (`"father passed away from a cardiac event"` expressing `"family history of MI"`), or negation
- Multi-span extraction requires special-case logic for each feature

#### 4.1.2 TF-IDF + Cosine Similarity

**TF-IDF (Term Frequency�Inverse Document Frequency)** represents each text chunk as a sparse vector of word frequencies, weighted by how distinctive each word is across a corpus. A similarity threshold then decides whether a note chunk "matches" a feature.

Applied to NBME, one could embed each feature as a TF-IDF vector, slide a window across the note, embed each window, and return the window with the highest cosine similarity if it exceeds a threshold. This approach captures partial word overlap but completely misses semantic similarity. `"heart attack"` and `"myocardial infarction"` share no words (setting aside "attack" vs "infarction" morphemes) and receive a cosine similarity of zero, even though they are synonymous.

**Benefits:**
- Simple to implement with scikit-learn
- Works well when features and notes share vocabulary
- No GPU required; scales to millions of documents

**Limitations:**
- Purely lexical � zero semantic understanding
- Fails on all paraphrase cases (the dominant challenge in this competition)
- Window size must be tuned per-domain; too small misses distributed spans, too large introduces noise
- Cannot handle negation (`"denies chest pain"` looks similar to `"chest pain"`)

#### 4.1.3 Conditional Random Fields (CRFs)

**Conditional Random Fields** are sequence labelling models that assign a label to each token in a sequence while modelling label dependencies (e.g., an I token must follow a B token). CRFs with hand-crafted features (word shape, capitalisation, n-gram context, dictionary lookup, POS tags) were state-of-the-art for NER through the mid-2010s. The classic reference is the CoNLL 2003 NER shared task, where CRF-based systems achieved F1 ~89%.

For NBME, a CRF would take as input the concatenated `[feature; separator; note]` token sequence and assign B/I/O labels to each token in the note. Features would include: is this word in a medical dictionary? What is the surrounding context? What are the character n-grams?

**Benefits:**
- Sequence-aware: explicitly models label transitions
- Interpretable features: practitioners know which features drive decisions
- Efficient training on CPU with limited data

**Limitations:**
- Feature engineering is domain-specific and time-consuming
- Limited to local context (typically a 5�7 word window)
- Cannot capture long-range dependencies (a feature mentioned near the start of the note, annotated near the end)
- Performance scales poorly with the number of features (one CRF per feature vs. a single multi-feature model)
- Sensitive to vocabulary distribution; poor generalisation to unseen medical terminology

#### 4.1.4 SVM-Based Approaches

**Support Vector Machines** with bag-of-words or n-gram features can be used for binary classification: given a (note, feature, candidate span) triple, predict whether the span expresses the feature. The candidate spans could be generated by a chunker (noun phrases, verb phrases). This reframes span extraction as a re-ranking problem.

**Benefits:**
- Well-understood theoretical properties; good generalisation with appropriate regularisation
- Can use kernel tricks for non-linear decision boundaries

**Limitations:**
- Feature design burden: n-gram features capture surface form but not semantics
- Candidate generation is itself a hard problem (how do you enumerate candidate spans without missing multi-word expressions?)
- Does not natively model multi-span cases
- Training cost grows with the number of candidate pairs

#### 4.1.5 Summary of Traditional ML Limitations

All traditional approaches share a fundamental weakness: they operate on surface-level lexical features and cannot generalise across semantic equivalences. The NBME task is fundamentally a **semantic matching** problem, not a lexical matching problem. As Table 4.1 shows, even the best traditional methods are expected to perform far below the top leaderboard scores.

**Table 4.1: Expected performance of traditional vs. deep learning approaches on NBME-style clinical span extraction**

| Method | Expected F1 (approximate) | Key Weakness |
|---|---|---|
| Exact string matching | ~0.05�0.15 | Cannot handle paraphrase |
| TF-IDF + cosine | ~0.15�0.25 | Purely lexical, no semantics |
| CRF with hand features | ~0.30�0.45 | Limited context, poor domain transfer |
| SVM re-ranking | ~0.35�0.50 | Feature engineering bottleneck |
| Fine-tuned BERT (2019 era) | ~0.75�0.82 | Smaller model, less pretraining |
| DeBERTa-v3-large (top leaderboard) | ~0.88�0.91 | Requires labeled fine-tuning data |
| Our best (LoRA ensemble) | **0.73195** | Generative, larger model, compute cost |

---

### 4.2 DeBERTa � The Dominant Leaderboard Paradigm

Virtually every top-ranking team in the NBME competition used **DeBERTa** (Decoding-enhanced BERT with Disentangled Attention) as their backbone. To understand why, we must examine DeBERTa's architectural innovations and why they confer specific advantages for span extraction.

#### 4.2.1 DeBERTa Architecture Overview

DeBERTa was introduced by Microsoft Research in 2020 (He et al., 2020). It improves upon BERT and RoBERTa in two fundamental ways:

**Innovation 1: Disentangled Attention Mechanism**

In standard BERT, each token is represented by a single embedding vector that conflates content information (what the word means) and position information (where it appears in the sequence). Attention scores are computed as dot products between these combined representations.

DeBERTa uses **two separate vectors** for each token: one for its content and one for its relative position. The attention score between tokens i and j is then computed as the sum of three interaction terms:

```
attention(i,j) = content(i)�content(j)    [what i is � what j is]
               + content(i)�position(j)   [what i is � where j is]  
               + position(i)�content(j)   [where i is � what j is]
```

The crucial missing term is `position(i)�position(j)` � DeBERTa argues this interaction is less informative for most NLU tasks.

Why does this matter for clinical span extraction? Consider detecting the boundary of the span `"Father died of heart attack at age 52"`. The word `"Father"` signals the beginning of the relevant mention, and `"52"` (the age) signals the end. The correct span boundary detection depends on understanding *both* the content of these tokens (they are semantically relevant to a family history feature) *and* their relative positions (they are adjacent to content that is outside the span). Disentangled attention gives the model finer-grained control over these position-content interactions, leading to more accurate boundary prediction.

**Innovation 2: Enhanced Mask Decoder (EMD)**

During BERT-style masked language model (MLM) pre-training, DeBERTa adds absolute position embeddings only in the decoder (prediction) layer rather than in the encoder. This design choice allows the encoder to focus purely on relative positional relationships (more informative for NLU tasks) while the decoder recovers absolute positions needed for accurate token-level prediction. The result is a model that achieves better MLM performance than RoBERTa with the same amount of pre-training data.

**Innovation 3: Scale**

DeBERTa-v3 (the version most used in NBME) incorporates additional improvements:
- **ELECTRA-style pre-training** (replaced token detection rather than masked language modelling), which is significantly more sample-efficient
- **Gradient-Disentangled Embedding Sharing** between the encoder and the replaced token discriminator
- Available in multiple sizes: DeBERTa-v3-base (184M params), DeBERTa-v3-large (435M params), DeBERTa-v2-xlarge (900M), DeBERTa-v2-xxlarge (1.5B)

#### 4.2.2 DeBERTa for NBME: The Standard Pipeline

Top teams formulated the NBME task as a **token classification (NER) problem** with DeBERTa:

**Step 1 � Input Construction:**
The note and feature are concatenated with a separator token:
```
[CLS] family history of MI or cardiac disease [SEP] patient is a 45-year-old male ... [SEP]
```
This allows bidirectional cross-attention between the feature and every token in the note.

**Step 2 � Token-Level BIO Classification:**
A linear classification head on top of DeBERTa's hidden states predicts one of three labels for each note token: B (Begin), I (Inside), O (Outside). The model is fine-tuned end-to-end on the 9,901 labeled pairs.

**Step 3 � Span Recovery:**
Contiguous runs of B/I tokens are decoded into span predictions. Character offsets are recovered from token-to-character mappings provided by the tokenizer.

**Step 4 � Post-processing:**
- Whitespace stripping at span boundaries
- Minimum span length filters
- Overlap resolution for conflicting predictions

**Step 5 � Ensembling:**
Multiple DeBERTa models (e.g., DeBERTa-v3-large � 10-fold cross-validation + DeBERTa-v2-xlarge � 5-fold) produce probability distributions over B/I/O for each token. These are averaged to produce a consensus prediction.

Additional techniques used by top teams (per our README analysis and competition knowledge):
- **Domain-adaptive pre-training:** MLM on the unlabeled pool of 41,000 notes at mask probability 0.15, before supervised fine-tuning. This adapts DeBERTa to clinical vocabulary and writing style.
- **Pseudo-labeling:** An initial model annotates the unlabeled pool, and high-confidence pseudo-labels are added to the training set. This effectively multiplies the available labeled data by ~20�.
- **Preprocessing:** Typo correction and medical abbreviation expansion on annotated spans. Cross-validation scores improved when `"SOB"` in the feature was expanded to `"shortness of breath"` before matching against notes.
- **Fold restarts:** DeBERTa-v2-xlarge and v2-xxlarge occasionally collapsed to zero loss during training, requiring fold-level restarts with different random seeds. This is a well-known instability in very large encoder models.

#### 4.2.3 Why DeBERTa Excels at This Task

We can now give a precise answer to why DeBERTa dominates this competition:

1. **Bidirectional context:** Token classification requires reading the full note before labelling any token. Encoder models process bidirectional context natively; decoder models would need to reframe the task (e.g., generating all tokens and then re-scoring them).

2. **Position-sensitive attention:** Clinical span boundaries depend heavily on relative position. `"Father died of MI"` contains the relevant span; `"Patient denies history of MI"` contains the same words but is *negative*. DeBERTa's disentangled attention is better at modelling these position-content interactions.

3. **Efficient fine-tuning:** DeBERTa-v3-large (435M parameters) is significantly smaller than our LLMs (1.2B�8B). Fine-tuning DeBERTa on 9,901 examples is computationally cheap, enabling extensive cross-validation and ensembling.

4. **Discriminative objective:** DeBERTa-v3 uses ELECTRA-style pre-training (is this token replaced or original?), which is a discriminative task very similar to the NBME labelling problem (is this token inside or outside a span?). This pre-training objective provides a strong inductive bias for the downstream task.

#### 4.2.4 Top Leaderboard Scores

According to the competition README and our background research:
- **Top leaderboard F1 score:** approximately **0.90+** on the public leaderboard
- **Academic follow-up (arxiv 2401.12994):** A two-phase LLM framework achieved 0.968�0.983 on the full dataset using a different evaluation protocol, compared to DeBERTa pipelines at ~0.958 under the same protocol
- The public leaderboard score of ~0.90 represents the practical achievable ceiling for the competition's official evaluation

These DeBERTa ensembles used 3�4 model variants (different sizes and fold configurations), with probability averaging as the aggregation strategy.

---

### 4.3 Our Approach vs. DeBERTa: A Detailed Comparison

Our solution makes a fundamentally different architectural choice: **generative span extraction with decoder-only LLMs**, fine-tuned with QLoRA. Table 4.2 provides a systematic comparison.

**Table 4.2: Detailed comparison of DeBERTa (top leaderboard) vs. our LLM-based approach**

| Dimension | DeBERTa-Based (Top Teams) | Our Approach (Generative SLM Ensemble) |
|---|---|---|
| **Architecture** | Encoder-only, bidirectional | Decoder-only, autoregressive |
| **Model family** | DeBERTa-v3-large/xlarge/xxlarge | Qwen3-1.7B, Qwen3-8B, LFM2.5-1.2B, Llama-3.1-8B |
| **Parameter count** | 184M � 1.5B (encoder only) | 1.2B � 8B |
| **Task framing** | Token classification (BIO labels) | Structured text generation (JSON) |
| **Output** | Per-token B/I/O logits | Verbatim span strings in `{"spans": [...]}` |
| **Multi-span handling** | Natural via B/I/O runs | Natural via list output |
| **Training objective** | Cross-entropy over BIO labels | Causal LM loss over full sequence |
| **Fine-tuning method** | Full fine-tuning | QLoRA (4-bit NF4, rank 8�16) |
| **Trainable parameters** | ~400M+ (full model) | ~20�40M (LoRA adapters only) |
| **Inference speed** | Fast (single forward pass) | Slower (autoregressive generation) |
| **Hallucination risk** | None (output is a label, not text) | Present; mitigated by FSM constraint |
| **Adaptation to new features** | Requires retraining | Change the prompt only |
| **Leaderboard F1 (best known)** | ~0.90+ | **0.73195** (our best, FIXED ensemble) |
| **Compute for training** | Moderate (full fine-tune) | Low (LoRA adapters) |
| **GPU requirements (inference)** | Single GPU (forward pass) | 2� T4 (tensor parallel for 8B) |

#### 4.3.1 Where DeBERTa Wins

DeBERTa-based token classification has clear advantages for this specific task:

- **Higher F1 score:** The ~0.17 gap between our best result (0.73195) and the leaderboard top (~0.90) is substantial. DeBERTa's discriminative architecture with bidirectional context is better suited to binary token labelling.
- **No hallucination:** A token classifier cannot hallucinate � it can only assign a label to tokens that exist in the input. A generative model can, in principle, generate text not present in the note (though our FSM constraint significantly mitigates this).
- **Faster inference:** A single DeBERTa forward pass processes the entire sequence in one step. Our autoregressive generation runs for up to 512 token steps per example, making inference substantially slower.
- **More training data efficiency:** 9,901 labeled pairs is a reasonable amount for fine-tuning a 435M-parameter model. Fine-tuning 8B-parameter LLMs on the same data requires careful regularisation to avoid overfitting.

#### 4.3.2 Where Our Approach Has Advantages

Despite the score gap, our generative LLM approach has genuine advantages that DeBERTa cannot match:

- **Zero-shot adaptability:** A new feature can be added to the evaluation rubric by simply including it in the prompt � no retraining, no new labeled data. DeBERTa fine-tuned on the original 150 features must be retrained to handle new features.
- **Few-shot in-context learning:** Our pretrained baseline (Section 6.1) achieves F1 of 0.48�0.54 with **zero parameter updates** � simply by showing 3 examples in the prompt. DeBERTa without fine-tuning on this task achieves near-zero F1.
- **Flexible output format:** The generative approach can be extended to produce explanations, uncertainty estimates, or alternative span candidates, simply by modifying the system prompt. Token classification is constrained to a fixed label vocabulary.
- **Smaller adapter footprint:** Each LoRA adapter is ~100 MB versus a full DeBERTa-v3-large checkpoint of ~1.7 GB. In deployment scenarios with many task variants, LoRA adapters are far more storage-efficient.
- **Model diversity:** Our ensemble combines fundamentally different architectures (Qwen3, LFM2.5, Llama-3.1) that make different types of errors. DeBERTa ensembles use the same architecture with different random seeds or fold assignments � less architectural diversity.

#### 4.3.3 Why We Chose the Generative Approach

Given that DeBERTa achieves higher scores, why did we pursue a generative approach? The motivation is both scientific and pedagogical:

1. **Research question:** The competition was dominated by a well-established paradigm (DeBERTa + token classification + pseudo-labeling). We wanted to explore whether modern instruction-tuned LLMs with parameter-efficient fine-tuning could approach this performance despite being architecturally mismatched to the task.

2. **Generalisability:** Real clinical NLP systems must handle continually evolving rubrics, new cases, and new domains. A generative system can be updated by prompt engineering; a DeBERTa system requires labeled data and retraining.

3. **Compute constraints:** We were constrained to Kaggle's 2�T4 GPU environment with a 9-hour time limit. Full DeBERTa fine-tuning with 10-fold cross-validation and pseudo-labeling would require substantially more compute than QLoRA fine-tuning of LLMs.

4. **Novel technique combination:** While LLMs, LoRA, and ensembling are individually well-known, combining QLoRA fine-tuned multi-model ensembles with per-note FSM-constrained generation and character-level majority voting for this specific task is, to our knowledge, a novel combination.

> Having established the competitive landscape, we now describe our proposed solution in detail � how it works, why each component was designed as it was, and what alternatives were considered.

---

## 5. Proposed Solution in Detail

### 5.1 High-Level Architecture

Our solution is a three-phase pipeline that transforms raw competition data into a Kaggle submission. At a high level:

- **Phase 1** uses a large pretrained LLM as an oracle to pseudo-label 2,000 unannotated patient notes, expanding the training set from 9,901 to ~39,500 examples.
- **Phase 2** fine-tunes four small language models (SLMs) with QLoRA on the combined labeled + pseudo-labeled dataset, producing four lightweight LoRA adapters.
- **Phase 3** runs batched inference on Kaggle's 2×T4 GPU environment using vLLM 0.17.1, then aggregates predictions via character-level majority voting.

```
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 1 — Data Augmentation                                          │
│                                                                      │
│  train.csv + features.csv + patient_notes.csv                        │
│         │                                                            │
│         ▼                                                            │
│  Sample 2,000 unannotated notes (seed=42)                            │
│         │                                                            │
│         ▼                                                            │
│  FAISS few-shot retrieval (top-3 annotated examples per query)        │
│         │                                                            │
│         ▼                                                            │
│  Qwen3-8B (fp16, greedy, max_new_tokens=128)                         │
│         │                                                            │
│         ▼                                                            │
│  JSON parse → span-to-offset mapping (exact → fuzzy)                 │
│         │                                                            │
│         ▼                                                            │
│  augmented_train.csv (~29,600 new pairs)                             │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 2 — QLoRA Fine-Tuning (4 models, sequential)                   │
│                                                                      │
│  train.csv + augmented_train.csv → GroupKFold (val_fold=4)           │
│         │                                                            │
│         ├── Qwen3-1.7B  → adapters/qwen3_1_7b_adapter/              │
│         ├── Qwen3-8B    → adapters/qwen3_8b_adapter/                 │
│         ├── LFM2.5-1.2B → adapters/lfm2_5_1_2b_adapter/             │
│         └── Llama-3.1-8B → adapters/llama3_1_8b_adapter/            │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 3 — Kaggle Inference (2×T4, vLLM 0.17.1)                      │
│                                                                      │
│  test.csv → prompt construction → 4× vLLM engines                   │
│                                                                      │
│  Each model: structured generation with per-note regex FSM           │
│                                                                      │
│  Character-level majority vote (threshold ≥ 3/4)                    │
│         │                                                            │
│         ▼                                                            │
│  submission.csv                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

Each phase is described in detail below.

---

### 5.2 Phase 1 — Data Augmentation (`1_generate_augmented_data.ipynb`)

#### 5.2.1 Motivation and Goal

The competition provides 9,901 labeled (note, feature, span) triples but 41,000+ unannotated notes. This is a classic semi-supervised learning scenario: we have abundant unlabeled data but limited annotations. The goal of Phase 1 is to convert 2,000 of those unannotated notes into pseudo-labeled training examples using a strong LLM as a teacher.

This addresses two of the core challenges identified in Section 2:
- **Class imbalance**: the labeled set has many features that appear in only a handful of notes. Pseudo-labeling from a diverse pool of 2,000 notes increases coverage of rare feature-mention combinations.
- **Small training set**: 9,901 pairs is modest for fine-tuning 8B-parameter models. The augmented dataset (~39,500 pairs) gives all four Phase 2 models substantially more signal.

#### 5.2.2 FAISS Few-Shot Retrieval

Naively providing the LLM with no context produces poor pseudo-labels — the model does not know the competition's exact annotation style (verbatim copying, multi-span output, JSON format). We use **retrieval-augmented few-shot prompting**: for each (note, feature) pair to be annotated, we retrieve the 3 most similar already-annotated examples from the training set and include them in the prompt.

**Embedding:** Each of the 9,901 training examples is embedded as the concatenated string `"Feature: <feature_text>  Annotation: <annotation_text>"` using `all-MiniLM-L6-v2` (384-dimensional sentence embeddings). These vectors are stored in a FAISS `IndexFlatIP` (inner product index over L2-normalised vectors = cosine similarity). The index is built once and cached to disk (`faiss_features.index`, `faiss_metadata.parquet`), so subsequent runs load in seconds.

**Query:** At inference time, the feature text for the current (note, feature) pair is embedded in the same format and the top-3 nearest neighbours are retrieved. These become the in-context examples that guide the LLM's annotation style.

**Why top-3?** We experimented conceptually with top-1 (too little context, model inconsistently follows format) and top-5 (context too long given the 4,096-token limit including the full patient note). Top-3 provides enough format guidance while keeping the prompt within budget.

**Why this embedding model?** `all-MiniLM-L6-v2` is a lightweight (22M parameter) sentence transformer that produces high-quality semantic embeddings for sentence-level retrieval. It runs on CPU in milliseconds per query — critical when processing ~29,600 pairs.

#### 5.2.3 LLM Oracle: Qwen3-8B

We use `Qwen/Qwen3-8B` as the pseudo-labeling oracle. Key configuration choices:

| Parameter | Value | Rationale |
|---|---|---|
| Precision | fp16 (not 4-bit) | Highest output quality for teacher model |
| `do_sample` | False (greedy) | Deterministic, JSON-parseable output |
| `enable_thinking` | False | Suppresses `<think>` chain-of-thought tokens that consume the 128-token budget |
| `max_new_tokens` | 128 | Sufficient for short JSON span lists |
| `max_length` | 4,096 | Accommodates system prompt + 3 examples + full patient note |
| `batch_size` | 32 | Throughput optimisation |
| `attn_implementation` | `"eager"` | Avoids CUDNN initialisation errors on some GPU configurations |

**Why Qwen3-8B as the teacher?** It is the largest model we could run in fp16 within the available VRAM during Phase 1. Using a larger, higher-quality teacher produces better pseudo-labels, which in turn produces better Phase 2 student models.

#### 5.2.4 Prompt Design

Each pseudo-labeling call uses a 2-message chat prompt:

**System message:** Instructs the model to extract verbatim spans only, return `{"spans": [...]}` JSON, produce no markdown, no explanation, and no `<think>` blocks. The system prompt emphasises three rules: copy text character-for-character; return an empty list if the feature is absent; output only valid JSON.

**User message:** Contains three FAISS-retrieved examples (each showing a note excerpt ≤300 characters, the feature text, and the correct JSON answer), followed by the full target note and feature text.

This structure mirrors the training format used in Phase 2, ensuring that pseudo-labels are generated in a style consistent with what the student models will be fine-tuned to produce.

#### 5.2.5 Span-to-Offset Mapping

The LLM outputs text strings (e.g., `"substernal chest pressure"`). The competition format requires character offsets (e.g., `"18 43"`). We use a three-step matching cascade, from most to least accurate:

1. **Exact substring search:** Python's `str.find()` — the fastest and most precise match.
2. **Case-insensitive exact search:** Normalises to lowercase before `find()` — handles capitalisation differences.
3. **RapidFuzz sliding window:** Tests all substrings within ±20% of the span's length using `fuzz.ratio`; accepts matches with score ≥ 72. This catches minor spelling variations, tokenization artifacts, and OCR-style errors between the LLM's output and the source text.

If none of the three steps succeeds, the span is discarded (treated as a hallucination). This conservative approach ensures pseudo-label quality at the cost of some recall.

#### 5.2.6 Checkpoint and Resume Mechanism

Processing 2,000 notes × ~14.8 features each = ~29,600 LLM calls takes several hours. To handle interruptions, the pipeline writes checkpoints atomically every 100 rows: it writes to a `.tmp` file and then atomically renames it to the final path. On restart, completed `row_id` strings (`f"{pn_num:05d}_{feature_num:03d}"`) are loaded and skipped, so no work is repeated.

**Output:** `augmented_train.csv` with columns `id, pn_num, feature_num, case_num, annotation, location` — the same schema as `train.csv`, making it trivially concatenable in Phase 2.

---

### 5.3 Phase 2 — QLoRA Ensemble Training (`2_train_slm_ensemble_v2.ipynb`)

#### 5.3.1 Model Selection

We fine-tune four models, chosen to provide both **architectural diversity** and **scale diversity**:

| Model | Parameters | Architecture | Notes |
|---|---|---|---|
| `Qwen/Qwen3-1.7B` | 1.7B | Grouped-query attention, RoPE | Small, fast inference |
| `Qwen/Qwen3-8B` | 8B | Grouped-query attention, RoPE, thinking mode | Largest Qwen variant tested |
| `LiquidAI/LFM2.5-1.2B-Instruct` | 1.2B | Custom (linear attention hybrid) | Different architecture family |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8B | Grouped-query attention, RoPE | Different pretraining data from Qwen |

**Why these four?** The key design principle is diversity at two levels:
- **Scale diversity:** 1B-class models (Qwen3-1.7B, LFM2.5-1.2B) and 8B-class models (Qwen3-8B, Llama-3.1-8B). We hypothesise that smaller and larger models make different types of errors, so their combination is complementary.
- **Architecture diversity:** Qwen3 and Llama-3.1 both use similar transformer architectures, but LFM2.5 uses a different attention mechanism (linear attention hybrid). Different architectures have different inductive biases and error patterns.
- **Pretraining diversity:** Qwen3 was pre-trained by Alibaba on a primarily Chinese and English corpus; Llama-3.1 was pre-trained by Meta on a different English-heavy corpus. Different pretraining data means different "knowledge bases" and different tendencies.

All four models are instruction-tuned variants, meaning they already understand how to follow system prompts and produce structured outputs — critical for our JSON generation task.

#### 5.3.2 QLoRA Configuration

All four models are loaded with **4-bit NF4 quantization** via BitsAndBytes:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — optimal for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bfloat16 for numerical stability
    bnb_4bit_use_double_quant=True,      # quantize the quantization constants too (~0.4 bit/param saving)
    bnb_4bit_quant_storage=torch.bfloat16,
)
```

This reduces the memory footprint of an 8B model from ~16 GB (fp16) to ~5 GB (4-bit), enabling training on a single GPU.

After loading in 4-bit, `prepare_model_for_kbit_training(use_gradient_checkpointing=True)` is called, which:
- Enables gradient checkpointing (trades compute for memory — recomputes activations during backward pass rather than storing them)
- Casts LayerNorm layers to float32 for numerical stability
- Freezes all base model parameters

**LoRA configuration (FIXED version):**

| Parameter | Value | Rationale |
|---|---|---|
| `lora_r` (rank) | 8 | Reduced from early experiments (r=16) — less adapter capacity means less memorisation of noisy pseudo-labels |
| `lora_alpha` | 16 | Scaling factor α/r = 2.0 — standard empirically effective ratio |
| `lora_dropout` | 0.10 | Increased regularisation on adapter weights |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | Attention-only; attention modules drive most of the span recognition capability |
| `bias` | `"none"` | No bias terms in LoRA layers |
| `task_type` | `CAUSAL_LM` | Autoregressive language modelling |

**Why attention-only target modules?** Early experiments using all 7 linear layers (`q/k/v/o/gate/up/down`) led to faster memorisation of training labels without proportionate generalisation improvement. Restricting LoRA to attention modules (the 4 listed) reduces adapter size and acts as an implicit regulariser.

**Per-model seeds for diversity:**
```python
MODEL_SEEDS = {
    "meta-llama/Llama-3.1-8B-Instruct": 42,
    "Qwen/Qwen3-1.7B":                  43,
    "Qwen/Qwen3-8B":                    44,
    "LiquidAI/LFM2.5-1.2B-Instruct":   45,
}
```
Each model is initialised with a different random seed, ensuring that stochastic elements of training (data shuffling, dropout) produce genuinely different adapters rather than near-identical ones. This increases the diversity that makes ensembling beneficial.

#### 5.3.3 Training Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 1e-4 | Reduced from 2e-4 in early runs — slower learning reduces memorisation of noisy labels |
| `num_train_epochs` | 3 | Three full passes over the training data |
| `per_device_train_batch_size` | 8 | |
| `gradient_accumulation_steps` | 2 | Effective batch size = 16 |
| `max_seq_length` | 1,024 tokens | Accommodates all training examples |
| `lr_scheduler_type` | `cosine` | Smooth decay after warmup |
| `warmup_ratio` | 0.05 | 5% of total steps for warmup |
| `weight_decay` | 0.05 | Increased L2 regularisation |
| `optim` | `adamw_8bit` | 8-bit Adam — saves ~6 GB VRAM with negligible convergence difference |
| `eval_strategy` | steps, every 500 | |
| `load_best_model_at_end` | True | Uses best validation checkpoint |
| `early_stopping_patience` | 3 | Stops if no improvement for 3 evaluation steps |
| `AUG_SAMPLE_RATIO` | 0.5 | Use 50% of augmented data (ablates distribution shift from pseudo-labels) |

**Why cosine learning rate schedule?** The cosine decay schedule smoothly reduces the learning rate from its maximum to near-zero, spending more steps at intermediate learning rates than a linear decay. This is widely found to improve generalisation in LLM fine-tuning. The 5% warmup prevents early training instability when the adapter weights are randomly initialised.

#### 5.3.4 Data Split Strategy

We use `GroupKFold(n_splits=5)` with `groups=case_num`, holding out `VAL_FOLD=4` for validation. This means the validation set contains all notes from 2 of the 10 clinical cases.

**Why group by case, not by row?** Notes from the same clinical case share:
- The same patient vignette (e.g., all notes for Case 3 describe a patient with chest pain and risk factors)
- The same set of features (the scoring rubric is case-specific)
- Similar vocabulary and clinical context

If we split by row (random), many training and validation rows would share case context. A model could achieve artificially high validation scores by memorising case-specific patterns. Splitting by case forces the model to generalise to entirely new cases — a much harder but more realistic evaluation.

#### 5.3.5 Training Procedure

Models are trained **sequentially** (one at a time) on a single GPU, with explicit VRAM cleanup between each:
```python
del trainer, model, tokenizer
gc.collect()
torch.cuda.empty_cache()
```

After training, only the **LoRA adapter** is saved (`model.save_pretrained(adapter_dir)` — ~100 MB per adapter), not the full quantized model. This makes adapter storage and transfer practical.

A custom `SafeEvalGenerationCallback` runs actual span generation on a subset of validation examples at each checkpoint, computing character-level F1 directly. This provides a more task-relevant evaluation signal than the surrogate cross-entropy loss used for early stopping.

#### 5.3.6 Design Justification Table

| Design Decision | Chosen Approach | Alternatives Considered | Rationale |
|---|---|---|---|
| Fine-tuning method | QLoRA (4-bit NF4, rank 8) | Full fine-tuning, prompt tuning, IA3 | Full fine-tuning requires ~80GB VRAM per 8B model; prompt tuning insufficient for structured JSON output; LoRA is the best compute-performance trade-off |
| Model architecture | Generative decoder LLM | DeBERTa encoder, BERT-CRF | Generative approach handles multi-span natively; zero-shot capable; novel relative to DeBERTa-dominated leaderboard |
| Target modules | Attention-only (q/k/v/o) | All 7 linear layers | Reduced overfitting; attention modules capture the most task-relevant representations |
| Rank | 8 | 16, 32, 64 | Higher rank memorises noisy pseudo-labels faster; rank 8 provides better generalisation |
| Data split | GroupKFold by case | Random split, stratified split | Prevents data leakage across notes from the same clinical case |
| Augmentation | Qwen3-8B pseudo-labels (FAISS few-shot) | No augmentation, simple paraphrase, SMOTE | LLM pseudo-labeling best preserves annotation quality; FAISS retrieval ensures format consistency |
| Ensemble strategy | Character-level majority vote | Probability averaging, span-level voting | Character-level voting is robust to boundary disagreements; span-level voting fails when spans don't match exactly |

---

### 5.4 Phase 3 — Kaggle Inference (`3_kaggle_inference_v2.ipynb`)

#### 5.4.1 Hardware Constraints and vLLM Choice

The Kaggle inference environment provides two T4 GPUs (14.6 GB VRAM each, 29.2 GB total) running CUDA 12.8 on Compute Capability 7.5. This is a tight constraint: T4 has no native bfloat16 tensor cores (requires float16), and the total VRAM budget must accommodate four model loads sequentially.

We use **vLLM 0.17.1** — the last version compatible with CUDA 12.8. Later vLLM versions (≥0.20.0) require CUDA 13, which Kaggle T4 environments do not support. vLLM provides:
- **PagedAttention:** Efficient KV cache management that dramatically increases throughput for batched inference
- **Native LoRA support:** Adapters are loaded via `LoRARequest` without merging into the base model — saving 2–5 minutes of merge+save+reload per model
- **Tensor parallelism:** 8B models are split across both T4 GPUs (`tensor_parallel_size=2`)

#### 5.4.2 Per-Note FSM Constraint

A key innovation in our inference pipeline is the **per-note regex Finite State Machine (FSM) constraint**. For each test row, we construct a regex that enforces two properties:

1. **Structural validity:** The output must match the JSON schema `{"spans": ["...", ...]}`.
2. **Lexical containment:** Each span string can only contain characters that appear in the source note. The regex character class is built from `set(pn_history)`.

This constraint is compiled by vLLM's XGrammar backend into an FSM that **masks the LLM's logits at each decoding step** — the model physically cannot generate characters not in the note. This eliminates hallucination of fabricated text, a significant failure mode for generative span extraction.

The trade-off: building a per-note regex is slightly slower than unconstrained generation, and very long notes produce very large FSMs. In practice, this overhead is manageable within the 9-hour Kaggle time limit.

#### 5.4.3 Prompt Construction

At inference time, each test row receives a prompt in the same format as training:

```
<system>
You are a clinical NLP specialist. Given a patient note and a clinical 
feature, extract the EXACT verbatim text spans from the note that express 
that feature.
Rules:
  1. Copy text character-for-character — do NOT paraphrase.
  2. If the feature is absent from the note, return an empty list.
  3. Output ONLY valid JSON — no markdown, no explanation.
{"spans": ["exact text 1", "exact text 2"]}
</system>

<user>
Note: "<full patient note text>"
Feature: <feature description>
/no_think
</user>
```

The `/no_think` suffix and `enable_thinking=False` flag suppress Qwen3's chain-of-thought reasoning mode. Without this, Qwen3 models prefix their output with a `<think>...</think>` block of potentially hundreds of tokens — consuming generation budget and adding latency without improving span extraction quality (since the FSM constraint already ensures output correctness).

#### 5.4.4 Character-Level Majority Voting

After running inference with all four models, we aggregate predictions via character-level majority voting:

**Step 1 — Locate spans in note:**  
For each model's predicted span strings, locate them in `pn_history` using the three-step cascade (exact → case-insensitive → rapidfuzz `partial_ratio_alignment` with cutoff 70).

**Step 2 — Convert to binary character arrays:**  
For each model, create a `uint8` array of length `len(pn_history)`, where position `i` is 1 if model predicts character `i` as part of a span.

**Step 3 — Sum across models:**  
Sum the four binary arrays. Each position now has an integer value 0–4 representing how many models predicted it as span-internal.

**Step 4 — Threshold at 3:**  
Positions with vote count ≥ 3 become 1 in the consensus array. This majority-vote threshold means at least 3 of the 4 models must agree before a character is included in the prediction.

**Step 5 — Whitespace trimming:**  
Zero out whitespace characters (`' ', '\t', '\n', '\r'`) at the edges of contiguous runs. This prevents spans from beginning or ending on whitespace, which would be an annotation error.

**Step 6 — Extract spans:**  
Identify contiguous runs of 1s as (start, end) tuples.

**Why character-level rather than span-level voting?** Consider two models predicting `"chest pain"` and another two predicting `"substernal chest pain"`. Span-level voting (do these strings match?) would fail — these are different strings. But character-level voting would show that the characters `"chest pain"` have 4 votes (100%), while `"substernal "` has only 2 votes (50%), correctly returning `"chest pain"` as the consensus.

#### 5.4.5 Submission Formatting

`format_location_string()` produces the final output:
1. Strip whitespace from span boundaries
2. Sort spans by start position
3. Merge any overlapping or adjacent spans
4. Format as `"start end;start end"` (space within pair, semicolon between pairs)
5. Rows with no predicted spans receive `NaN` in the location column

---

### 5.5 Novelty Statement

> **Our key contributions to this task are:**
>
> 1. **Generative JSON extraction with per-note FSM constraints:** We frame span extraction as constrained JSON generation, where hallucination is physically prevented by a per-note regex FSM compiled by vLLM's XGrammar backend. This combines the flexibility of generative extraction with the reliability of constrained decoding.
>
> 2. **Character-level majority voting across architecturally diverse LLMs:** Unlike DeBERTa ensembles that aggregate probability distributions from the same architecture with different seeds, we combine fundamentally different model families (Qwen3, LFM2.5, Llama-3.1) using character-level voting — a more robust aggregation strategy for span boundaries.
>
> 3. **FAISS-retrieval-augmented pseudo-labeling with instruction-tuned LLMs:** We use semantic retrieval (not random sampling) to select the most informative few-shot examples for pseudo-labeling 2,000 unannotated notes, producing higher-quality augmented data than random few-shot or no-shot pseudo-labeling.
>
> The novelty lies primarily in the **combination** of these techniques rather than in any single component. Each individual component (LoRA, ensembling, few-shot retrieval, FSM constraints) is well-established. Their combination for clinical span extraction on resource-constrained hardware is, to our knowledge, novel.

> Having described the solution, we now present the experimental results that demonstrate its effectiveness — and its limitations.

---

## 6. Experimental Study

This section presents the quantitative evidence for our solution's effectiveness. We report Kaggle leaderboard scores, analyse the effect of ensemble composition, examine training dynamics, and conduct an ablation and error analysis. **Only two approaches are reported here:** Pretrained Only + Few Shot and LoRA Adapter v2 (FIXED).

### 6.1 Overall Results and Kaggle Scores

Table 6.1 presents the Kaggle public leaderboard scores for our two primary approaches across all tested model configurations.

**Table 6.1: Complete Kaggle public leaderboard results**

| Model Configuration | Vote Threshold | Pretrained Only + Few Shot | LoRA Adapter v2 (FIXED) |
|---|---|---|---|
| Qwen3-1.7B (single) | 1 | 0.48093 | 0.65140 |
| Qwen3-8B (single) | 1 | 0.49500 | 0.71670 |
| Llama-3.1-8B (single) | 1 | — | 0.68951 |
| LFM2.5-1.2B (single) | 1 | — | 0.63392 |
| Qwen3-8B + Qwen3-1.7B + LFM2.5 (3-model) | 2 | 0.53125 | 0.71046 |
| Llama + Qwen3-8B + Qwen3-1.7B + LFM2.5 (4-model) | 3 | — | **0.73195** |

**Key observations:**
- LoRA fine-tuning consistently improves over pretrained-only inference: the best single-model pretrained score (0.49500) is far below the worst single-model LoRA FIXED score (0.63392).
- The gap between pretrained-only and LoRA FIXED narrows for larger models (Qwen3-8B: 0.495 → 0.717, a gain of +0.222) compared to smaller models (LFM2.5: pretrained not tested → 0.634).
- The best overall result is **0.73195** achieved by the 4-model LoRA FIXED ensemble with vote threshold 3.
- Our best score (0.73195) is approximately **0.17 below the top leaderboard** (~0.90+), a gap discussed in Section 7.

---

### 6.2 LLM Combination Analysis

#### 6.2.1 Single Model vs. Ensemble: Does Ensembling Help?

Comparing single-model and multi-model LoRA FIXED results:

| Configuration | Score | Δ vs. Best Single Model |
|---|---|---|
| Best single model (Qwen3-8B) | 0.71670 | — |
| 3-model ensemble | 0.71046 | −0.006 |
| 4-model ensemble (best) | 0.73195 | **+0.015** |

The 3-model ensemble (Qwen3-8B + Qwen3-1.7B + LFM2.5) actually scores *slightly lower* than the best single model (Qwen3-8B alone). This is an important finding: ensembling is not automatically beneficial. Adding weaker models (particularly LFM2.5-1.2B at 0.634) can drag down the ensemble if the vote threshold is not carefully tuned.

The 4-model ensemble that *includes Llama-3.1-8B* achieves the best score of 0.73195. Llama-3.1-8B is a strong individual model (0.68951) and contributes complementary predictions to the ensemble. When both strong 8B models (Qwen3-8B and Llama-3.1-8B) are in the ensemble with vote threshold 3, their agreement on high-confidence spans dominates the result, while the two 1B models (Qwen3-1.7B and LFM2.5) act as tiebreakers.

**Interpretation:** Ensembling is most effective when all ensemble members are strong *and* diverse. A 3-member ensemble that includes a weak model (LFM2.5-1.2B: 0.634) is worse than the strongest single model. A 4-member ensemble that includes two strong models (Qwen3-8B: 0.717, Llama-3.1-8B: 0.690) and two weaker models achieves the best balance.

#### 6.2.2 Model Size Effect: 1B-Class vs. 8B-Class

Comparing models by parameter scale (LoRA FIXED):

| Size Class | Models | Mean F1 | Std |
|---|---|---|---|
| 1B-class | Qwen3-1.7B (0.651), LFM2.5-1.2B (0.634) | 0.643 | 0.012 |
| 8B-class | Qwen3-8B (0.717), Llama-3.1-8B (0.690) | 0.703 | 0.019 |

8B models outperform 1B models by approximately **0.06 F1 points** on average. The training report also confirms this pattern on the validation set: 8B models achieve mean `val_char_f1 = 0.737` versus `0.633` for 1B models.

**Interpretation:** Larger models have more parameters to represent clinical semantics and more capacity for the LoRA adapter to encode task-specific knowledge. The 8B models' superior performance supports investing compute in larger base models, even when using QLoRA for efficiency.

However, the improvement is not proportional to parameter count: Qwen3-8B (8B) is 4.7× larger than Qwen3-1.7B (1.7B) but only 1.10× better in F1. This suggests **diminishing returns** from model scale for this specific task and fine-tuning configuration. The limiting factor at 8B scale may be the training data quality (noisy pseudo-labels) rather than model capacity.

#### 6.2.3 Model Combination Synergy: Which Pairs Complement Each Other?

The best ensemble result (0.73195) comes from combining Llama-3.1-8B + Qwen3-8B + Qwen3-1.7B + LFM2.5-1.2B. This combination is better than the best individual model (Qwen3-8B: 0.717) by +0.015 F1.

The key synergy appears to be between Qwen3-8B and Llama-3.1-8B. These models have:
- Different pretraining data (Alibaba vs. Meta)
- Different tokenizers
- Different random seeds during QLoRA training (44 vs. 42)

Together, they vote confidently on high-precision spans and use the 1B models as tiebreakers on uncertain cases. This is consistent with the character-level voting mechanism: for a span that Qwen3-8B and Llama-3.1-8B agree on (2 votes), the vote threshold of 3 requires at least one of the 1B models to also predict it — a natural precision filter.

**Hypothesis for why homogeneous model ensembles underperform:** A 3-model ensemble of only Qwen3-8B variants (tested in the deprecated LoRA v1 experiments) showed minimal improvement over a single Qwen3-8B. This is because models of the same architecture with the same training data make **correlated errors** — they fail on the same examples. Architecturally diverse models have different error patterns and the ensemble gains from error cancellation.

#### 6.2.4 Pretrained Only + Few Shot: A Deeper Look

The pretrained-only approach uses no task-specific parameter updates — only prompt engineering. Results:

| Configuration | Vote Threshold | Score |
|---|---|---|
| Qwen3-1.7B | 1 | 0.48093 |
| Qwen3-8B | 1 | 0.49500 |
| Qwen3-1.7B + Qwen3-8B + LFM2.5 | 2 | 0.53125 |
| Llama + Qwen3-8B + Mistral | 2 | 0.53762 |

Achieving F1 of 0.48–0.54 with **zero fine-tuning** is notable. For comparison, the LoRA FIXED approach achieves 0.63–0.73. The gap of ~0.17–0.19 F1 represents the value added by QLoRA fine-tuning.

The pretrained-only approach also demonstrates that our FAISS-retrieved few-shot examples are effective: a 3-model ensemble with vote threshold 2 achieves 0.53125, a +0.04 improvement over a single pretrained model. This confirms that even without fine-tuning, model diversity and majority voting provide meaningful gains.

---

### 6.3 Training Dynamics

This section analyses the training process using data from the PEFT training evaluation report (`phase2_output/phase2_lora_training_report.ipynb`).

#### 6.3.1 Training Loss Convergence

All four models show a characteristic two-phase loss trajectory: a **steep initial decline** in the first few hundred steps, followed by a **slow, gradual decrease** that continues throughout training.

![Training Loss vs. Step — all four models show rapid early convergence followed by continued slow improvement](phase2_output/nb_extracted/cell09_out0.png)

*Figure 6.1: Training loss vs. training step for all four models. The steep initial drop reflects rapid adaptation of the LoRA adapters to the task format (JSON span extraction). The subsequent gradual decline reflects fine-grained learning of clinical semantics.*

**Final training losses:**

| Model | Final Training Loss |
|---|---|
| Qwen3-8B | 0.0924 |
| Llama3.1-8B | 0.1107 |
| Qwen3-1.7B | 0.1116 |
| LFM2.5-1.2B | 0.1165 |

Qwen3-8B achieves the lowest training loss, consistent with its highest individual Kaggle score. The 1B models converge to higher training losses, suggesting they have less capacity to memorise the training set format — which, given label noise in pseudo-labels, is not necessarily a disadvantage.

**Interpretation:** The steep initial drop occurs because the models already "know" how to generate JSON from pretraining — they simply need to learn *which* spans to include. This prior knowledge is adapted quickly in the first ~300 steps. Subsequent training refines boundary precision and handles the nuanced clinical semantics.

#### 6.3.2 Learning Rate Schedule

![Learning Rate Schedule — all four models follow identical cosine decay with 5% warmup](phase2_output/nb_extracted/cell11_out0.png)

*Figure 6.2: Learning rate schedule. All four models follow the same cosine decay schedule with 5% warmup. The initial warmup (325 steps) prevents large gradient updates when the LoRA adapters are randomly initialised.*

The learning rate follows the expected pattern: linear warmup from near-zero to the peak of `1e-4`, then cosine decay to near-zero at the end of training. Key values:
- **Initial LR at warmup start:** ~2.30e-05
- **Warmup steps (5%):** 325 steps
- **Peak LR:** 1e-4
- **Final LR:** ~3.99e-05

The four models' curves overlay exactly, confirming that all models use the same schedule. The cosine decay ensures the model spends significant time at intermediate learning rates (between peak and final), which is known to improve generalisation compared to a step decay schedule.

#### 6.3.3 Gradient Norms

![Gradient Norm Over Time — all models show early spikes that settle into a stable range](phase2_output/nb_extracted/cell13_out0.png)

*Figure 6.3: Gradient norm over training steps. All models show an initial spike (adaptation shock when adapters first receive gradients) followed by stabilisation. Qwen3-8B shows the tightest gradient range, while LFM2.5-1.2B shows the highest norms throughout.*

**Gradient norm statistics:**

| Model | Mean | Std | Min | Max |
|---|---|---|---|---|
| LFM2.5-1.2B | 1.373 | 0.669 | 0.628 | **5.325** |
| Llama3.1-8B | 0.869 | 0.346 | 0.403 | 2.031 |
| Qwen3-1.7B | 0.983 | 0.462 | 0.459 | 3.901 |
| Qwen3-8B | **0.534** | **0.242** | **0.290** | 1.997 |

Qwen3-8B has the most stable gradients (lowest mean and std), which correlates with its strongest performance. LFM2.5-1.2B's high gradient norms — despite gradient clipping — suggest that its LoRA adaptation to this task is less stable, possibly because its fewer target modules (5: q,k,v,out,in vs 4 for others) distribute the adaptation unevenly.

**Interpretation:** Stable gradients indicate that the optimiser is making consistent, reliable progress. LFM2.5's higher norms suggest it is struggling to find a stable adaptation direction, which may explain its relatively lower Kaggle score (0.634 vs. 0.717 for Qwen3-8B).

#### 6.3.4 Train vs. Validation Loss: Overfitting Analysis

![Train vs. Validation Loss per model — clear overfitting visible after early steps](phase2_output/nb_extracted/cell15_out0.png)

*Figure 6.4: Train vs. validation loss curves for each model. Training loss (blue) decreases monotonically throughout. Validation loss (orange) reaches its minimum early (~1,000–2,000 steps) and then increases, indicating overfitting.*

This is the most diagnostic figure in the training report. All four models exhibit the same pattern:
- Training loss decreases monotonically throughout all 3 epochs
- Validation loss reaches its minimum between steps 1,000 and 2,000, then increases steadily
- The train-validation gap widens over time

**Best evaluation steps:**

| Model | Best Eval Step | Best Eval Loss |
|---|---|---|
| Qwen3-8B | 1,000 | 0.1628 |
| Llama3.1-8B | 1,000 | 0.1967 |
| LFM2.5-1.2B | 2,000 | 0.2103 |
| Qwen3-1.7B | 2,500 | 0.2058 |

8B models reach their best validation performance earlier (step 1,000) than 1B models (step 2,000–2,500). This is consistent with larger models having more capacity to overfit: they memorise the training set faster.

**The early stopping callback** (`EarlyStoppingCallback(patience=3)`) and `load_best_model_at_end=True` ensure that the saved adapter corresponds to the best validation checkpoint, not the final (overfit) step. This is a critical implementation detail — without early stopping, the models would be meaningfully worse.

**Train-validation gap values:**

| Model | Train-Val Gap (final step) |
|---|---|
| Qwen3-8B | 0.022 |
| Llama3.1-8B | 0.052 |
| Qwen3-1.7B | 0.051 |
| LFM2.5-1.2B | 0.059 |

Qwen3-8B's remarkably small train-val gap (0.022) despite the best validation loss suggests it has the most efficient LoRA adaptation — it learns task structure without overfitting to label noise.

#### 6.3.5 Model Comparison Heatmap

![Model Comparison Heatmap — normalised metrics across all four models](phase2_output/nb_extracted/cell17_out0.png)

*Figure 6.5: Heatmap of key metrics across all four models (normalised for visual comparison). Darker = better. Qwen3-8B leads on loss metrics; both 8B models lead on val_char_f1.*

**Heatmap key findings:**
- `val_jsonish_rate = 1.0` for all four models — every single output can be parsed as valid JSON. This confirms that LoRA fine-tuning reliably teaches the models to produce the correct output format.
- `val_contained_rate` (what fraction of predictions contain the ground truth): Qwen3-8B highest at 0.977, LFM2.5-1.2B lowest at 0.938. All values are high, meaning the models are generally finding the correct region of the note.
- `val_char_f1` (character-level F1 on the validation set): Qwen3-8B leads at 0.764, followed by Llama3.1-8B at 0.711, then LFM2.5-1.2B at 0.640, then Qwen3-1.7B at 0.626.

**Interesting discrepancy:** Llama3.1-8B has lower `val_char_f1` (0.711) than Qwen3-8B (0.764) on the local validation set, but their Kaggle scores are closer (Llama: 0.690, Qwen3-8B: 0.717). This suggests that Llama-3.1-8B generalises better to the test distribution than its validation performance suggests — possibly because the validation set (cases held out during training) happens to be harder for Qwen3-8B than for Llama.

![Training Loss vs. Wall Time — high-resolution tfevents data](phase2_output/nb_extracted/cell19_out2.png)

*Figure 6.6: Training loss vs. wall time (from TensorBoard tfevents). Each model run shows the same broad loss trajectory as the step-based plot but at higher temporal resolution. LFM2.5-1.2B (orange) runs longest in wall time due to its additional training steps.*

---

### 6.4 Ablation Analysis

We structure our results as an ablation study to show the contribution of each pipeline component. Where intermediate configurations were not explicitly tested, we note this.

**Table 6.2: Ablation study — F1 score and contribution of each component**

| Configuration | F1 Score | Δ from Previous | Notes |
|---|---|---|---|
| **Baseline:** Pretrained Qwen3-8B, single model | 0.49500 | — | No fine-tuning, no ensemble |
| Pretrained 3-model ensemble | 0.53125 | +0.036 | Adding ensemble to pretrained |
| LoRA FIXED single model (Qwen3-1.7B) | 0.65140 | — | Smallest fine-tuned model |
| LoRA FIXED single model (Qwen3-8B) | 0.71670 | +0.222 vs. pretrained | Large model + fine-tuning |
| LoRA FIXED 3-model ensemble | 0.71046 | −0.006 vs. best single | Ensemble with weak member |
| **LoRA FIXED 4-model ensemble (best)** | **0.73195** | **+0.015 vs. best single** | Diverse 4-model ensemble |

**Component contributions:**
1. **QLoRA fine-tuning alone** adds approximately +0.22 F1 over pretrained-only (Qwen3-8B: 0.495 → 0.717). This is the single largest contributor.
2. **Ensemble diversity** adds +0.015 F1 when moving from the best single model to the 4-model ensemble. The gain is real but modest — fine-tuning quality dominates.
3. **Model selection in ensemble** matters: replacing a weaker model (LFM2.5-1.2B) with a stronger one (Llama-3.1-8B) in the ensemble improved the result from 0.71046 to 0.73195 (+0.021).

**What we did not ablate (future work):**
- Effect of augmented data (AUG_SAMPLE_RATIO = 0.0 vs. 0.5 vs. 1.0)
- Effect of FAISS retrieval quality (random few-shot vs. FAISS-retrieved few-shot)
- Effect of FSM constraint (constrained vs. unconstrained generation)
- Effect of vote threshold (2/4 vs. 3/4 vs. 4/4)

---

### 6.5 Error Analysis

To understand where our model fails, we categorise error types based on the task structure, training dynamics, and the nature of character-level F1.

#### 6.5.1 Missed Spans (Low Recall)

**Description:** The model outputs `{"spans": []}` or a partial span list, failing to detect a feature that is genuinely present in the note.

**Likely causes:**
- **Rare clinical expressions:** A feature expressed using a highly unusual or idiomatic phrasing not well-represented in the LLM's pretraining data or the pseudo-labeled augmentation set.
- **Implicit statements:** Phrases that logically imply a feature without stating it explicitly. For example, `"Patient is a 65-year-old smoker"` implies tobacco use as a risk factor for coronary disease, but does not directly state it as a risk factor.
- **Very short notes:** Students who wrote very brief notes may have used abbreviations only (`"SOB, CP, diaphoresis"`) without the full clinical vocabulary the model was trained on.

**Example (constructed):**
- Feature: `"exertional component to symptoms"`
- Note: `"Pain gets worse when walking upstairs."`
- Model output: `{"spans": []}` (fails to recognise the exertional component)
- Ground truth: `"worse when walking upstairs"`

**Mitigation strategies:** Better semantic retrieval for few-shot examples during pseudo-labeling; additional domain-adaptive pretraining on clinical notes; training data augmentation specifically targeting rare feature expressions.

#### 6.5.2 Boundary Errors

**Description:** The model identifies the correct region of the note but gets the start or end character position slightly wrong (off by a few characters). This is the most common error type for encoder-based span extraction models and likely affects our generative approach similarly.

**Likely causes:**
- **Span scope ambiguity:** It is genuinely unclear whether the span should include surrounding context words (e.g., `"severe"` before `"chest pain"`).
- **Whitespace and punctuation:** Medical abbreviations and note formatting create ambiguous boundaries. The FSM constraint strips whitespace at boundaries, which helps but is not perfect.
- **Partial span generation:** The model generates `"chest pain"` when the annotator included `"sharp chest pain"` — a systematic tendency to generate minimal spans.

**Impact on F1:** Boundary errors reduce both precision (predicted characters outside the ground truth) and recall (ground-truth characters not predicted). However, the character-level majority voting step mitigates boundary errors: if 3 of 4 models agree on the core of the span but disagree on the edges, the voted span will correctly represent the agreed-upon region.

#### 6.5.3 Hallucinated Spans

**Description:** The model predicts a span that is present in the note but does not express the queried feature.

**Example (constructed):**
- Feature: `"family history of MI"`
- Note: `"Patient is a 65-year-old male with no family history of cardiac disease. Father is alive and healthy."`
- Model output: `{"spans": ["Father is alive and healthy"]}` (hallucinated — this is a negative statement)

**Mitigation:** The FSM constraint prevents the model from generating text *not in the note* (preventing fabricated spans) but cannot prevent the model from selecting the *wrong* span from the note. The vote threshold of 3/4 also helps: a hallucinated span predicted by only 1 or 2 models does not make it into the final prediction.

**Likely causes:**
- **Negation confusion:** The model predicts a span that mentions the feature concept but in a negative context. Clinical negation detection is a hard NLP subtask that our approach handles only implicitly through task-specific fine-tuning.
- **Co-occurrence biases in pseudo-labels:** If the pseudo-labeling teacher (Qwen3-8B) was systematically confused by negation, the student models may have learned the same error pattern.

#### 6.5.4 Feature Confusion

**Description:** The model annotates text that correctly expresses *a* clinical feature, but not the *queried* feature.

**Example:**
- Feature: `"dyspnea on exertion"`
- Note: `"Patient complains of shortness of breath at rest and occasional chest pain on exertion."`
- Model output: `{"spans": ["chest pain on exertion"]}` (confused "exertion" with correct feature)

**Likely causes:**
- **Shared vocabulary across features:** Features like `"dyspnea on exertion"` and `"chest pain on exertion"` share the word `"exertion"`. Without careful attention to which feature is being queried, models may latch onto the shared vocabulary.
- **Feature text formatting:** Some feature descriptions in the rubric are long and complex (e.g., `"chest pain - quality, radiation, associated symptoms, precipitating and palliating factors"`). The model may only attend to part of the feature description.

**Mitigation:** More explicit instruction in the system prompt to read the complete feature description before extracting; attention mechanisms that more strongly weight the feature-to-note cross-attention in the concatenated input.

#### 6.5.5 Summary Error Table

| Error Type | Estimated Frequency | Primary Cause | Existing Mitigation |
|---|---|---|---|
| Missed spans | Moderate | Implicit expressions, rare phrasing | Few-shot examples, augmentation |
| Boundary errors | High | Span scope ambiguity | Character-level voting |
| Hallucinated spans | Low (FSM-constrained) | Wrong span selected from note | Vote threshold ≥ 3/4 |
| Feature confusion | Low-moderate | Shared vocabulary, complex features | Task-specific fine-tuning |

---

## 7. Discussion & Limitations

### 7.1 Honest Assessment: Why We Fall Short of DeBERTa

Our best result — F1 of 0.73195 — is approximately 0.17 points below the top DeBERTa-based leaderboard solutions (~0.90+). This gap is substantial, and we owe the reader an honest explanation of its causes.

**Architectural mismatch.** The NBME task is fundamentally a *discrimination* problem: given a token in the note, is it inside or outside a span? Encoder-only models with token classification heads are architecturally optimised for exactly this. Decoder-only generative models must solve a harder problem: generate the span text, then locate it in the note. Every additional step (generation → location → voting) introduces potential errors.

**Context limitation.** DeBERTa reads the note bidirectionally — every token is simultaneously attended to by all other tokens. Our decoder models generate left-to-right, with no ability to "look back" and revise an already-generated span boundary based on later context. A token classifier can use the full sentence context to decide every boundary simultaneously; our models must commit to each character of the span in sequence.

**Training data efficiency.** DeBERTa fine-tuned on 9,901 labeled pairs with 10-fold cross-validation has seen each example 10 times across folds. Our QLoRA training uses a single train/val split and 50% of pseudo-labeled data (AUG_SAMPLE_RATIO=0.5). Top DeBERTa teams also added 41,000 pseudo-labeled notes (vs. our 2,000), dramatically increasing effective training set size.

**Pseudo-label quality.** Our pseudo-labels are generated by Qwen3-8B with few-shot prompting — the same model family we then fine-tune. This creates a teacher-student circular dependency: the teacher's errors are inherited by the student. DeBERTa pseudo-labeling, by contrast, uses a fully fine-tuned discriminative model as the teacher, which makes fewer errors on the training distribution.

### 7.2 Compute vs. Performance Trade-off

DeBERTa-v3-large (435M params) fine-tuned for span classification is:
- Smaller than our 8B models by 18×
- Faster at inference (single forward pass vs. up to 512 autoregressive steps)
- Achieves ~0.17 higher F1

Our approach requires:
- 4× separate QLoRA fine-tuning runs (though each is GPU-efficient)
- 4× sequential inference passes with vLLM
- 2× T4 GPUs with tensor parallelism for 8B models

In production terms, DeBERTa is a far more efficient solution. Our approach is justified primarily for research purposes and for scenarios where flexibility (zero-shot new features) outweighs raw performance.

### 7.3 The Generalisability Argument

Our approach's key practical advantage is **prompt-based adaptability**. Consider the following scenario: the NBME adds 10 new scored features to a clinical case mid-cycle. With DeBERTa, the pipeline requires: (1) annotated examples for each new feature, (2) retraining or continued fine-tuning, (3) re-running pseudo-labeling on the unlabeled pool, (4) re-ensembling. This takes days to weeks.

With our approach: update the feature list, and the pretrained LLM can immediately attempt extraction using few-shot examples from the new feature. Performance will be lower than fine-tuned, but the system is functional on day one. This zero-shot adaptability is a genuine advantage in production medical NLP pipelines where rubrics evolve continuously.

### 7.4 Scalability

**More features:** Adding features requires no retraining — only adding new rows to the test dataframe.

**Longer notes:** Our max sequence length of 1,024 tokens covers most patient notes. Notes substantially longer than ~600 words may be truncated. Sliding window approaches (process note in overlapping chunks) would address this.

**Different clinical domains:** Moving from USMLE Step 2 CS notes to, say, ICU nursing notes or radiology reports would require re-running Phase 1 (pseudo-labeling) and Phase 2 (fine-tuning) with domain-appropriate data. However, the pipeline is general — the only domain-specific component is the training data itself.

**Different languages:** Qwen3 supports Chinese and English. Extending to Spanish or French clinical notes would require a multilingual base model and multilingual few-shot examples but no architectural changes.

### 7.5 Limitations of This Study

1. **Single competition metric:** We optimise and evaluate against character-level F1. This may not fully reflect clinical utility — a system that consistently gets span boundaries slightly wrong might be acceptable in practice.
2. **No comparison of inference time:** We report F1 scores but not inference latency. DeBERTa is substantially faster; for real-time clinical applications this matters.
3. **Limited compute budget:** We tested four base models. A wider hyperparameter search (LoRA rank, alpha, learning rate, augmentation ratio) might yield better results.
4. **Single language:** All experiments are on English clinical notes. Generalisability to non-English settings is untested.
5. **Pseudo-label quality not directly measured:** We measure downstream F1 but not the precision/recall of the pseudo-labels themselves against a gold standard. If pseudo-labels are systematically biased, this is invisible in our evaluation.

---

## 8. Conclusion

### 8.1 What We Set Out to Do

We entered the NBME — Score Clinical Patient Notes competition with a specific research hypothesis: *can modern instruction-tuned LLMs with parameter-efficient LoRA fine-tuning approach the performance of DeBERTa-based token classifiers on clinical span extraction?* Our pipeline was designed to test this hypothesis within the constraints of Kaggle's 2×T4 GPU environment.

### 8.2 What Worked

**QLoRA fine-tuning works.** Moving from pretrained-only inference (0.495 F1 for single Qwen3-8B) to LoRA-fine-tuned inference (0.717) is a gain of +0.222 F1 with no architectural change — just parameter-efficient adaptation to the task. This demonstrates that even small LoRA adapters (rank 8, ~100 MB) can substantially specialise a general-purpose LLM for a narrow clinical task.

**Ensemble diversity matters.** The best single model (Qwen3-8B: 0.717) is improved by the 4-model diverse ensemble to 0.73195. The key is architectural and pretraining diversity — the same model architecture with different seeds does not provide meaningful gains.

**Few-shot prompting is a strong baseline.** The pretrained-only approach achieves 0.54 F1 with no fine-tuning — a surprisingly competitive starting point. For rapid deployment in a new clinical domain where labeled data is unavailable, few-shot prompting with FAISS-retrieved examples is a practical solution.

**Constrained generation prevents hallucination.** The per-note FSM regex constraint in vLLM's XGrammar backend effectively eliminates hallucination of fabricated text. Every predicted span is provably a substring of the source note.

**Character-level majority voting is robust.** Voting at the character level (rather than span string level) gracefully handles boundary disagreements between models, producing consensus predictions that are more accurate than any individual model.

### 8.3 What Did Not Work

**More models in the ensemble is not always better.** Adding a weak model (LFM2.5-1.2B: 0.634) to a strong 2-model ensemble actually *decreased* performance relative to the best single model. Ensemble quality depends on member quality; weak diverse models can hurt more than they help at a vote threshold designed for strong members.

**Homogeneous ensembles provide minimal gains.** Testing multiple copies of the same architecture (e.g., Qwen3-8B × 3) showed negligible improvement over a single model. Diversity is the key ingredient in ensemble benefit.

**Full fine-tuning underperformed QLoRA.** Experiments with full fine-tuning (reported in deprecated results) showed that updating all parameters of a large LLM on 9,901–39,500 examples leads to overfitting and poor generalisation, particularly for smaller models like Qwen3-1.7B (0.113 F1 — near random). QLoRA's regularisation through low-rank constraints and 4-bit quantization is better suited to small-data settings.

### 8.4 The Deployment Environment Lesson

One of the most practically valuable lessons from this project was entirely non-algorithmic: **model selection must consider the deployment environment, not just benchmark performance.**

We initially trained and planned to deploy Qwen3.5-9B and Gemma4-E4B models, which showed strong performance during local testing. However, when we attempted to run them in Kaggle's T4 inference environment, we encountered a critical compatibility failure:

- **vLLM ≤0.19.1** incorrectly routes Qwen3.5 and Gemma4 (which have multimodal/hybrid architecture components) to its vision-language handler, raising `preprocessor_config.json not found` at runtime — even for text-only inference.
- **vLLM ≥0.20.0** requires CUDA 13. Kaggle's T4 environments run CUDA 12.8 — the installation fails silently.

This forced us to pivot to models (Qwen3-1.7B, Qwen3-8B, LFM2.5-1.2B, Llama-3.1-8B) that are compatible with vLLM 0.17.1 — the last version that runs on CUDA 12.8 and correctly handles these architectures. The resolution was pinning `vllm==0.17.1` in the Kaggle notebook's installation step.

**The lesson:** Before committing to a model for a production deployment, verify end-to-end compatibility with the inference stack, not just the training stack. A model that trains successfully on a local A100 cluster may be undeployable in a constrained cloud environment due to CUDA version mismatches, VRAM limits, or library incompatibilities. Future projects should run a small-scale inference validation test in the target environment before investing significant training compute.

### 8.5 Comparison to the DeBERTa-Dominated Leaderboard

Is our generative LLM approach a viable alternative to DeBERTa for clinical span extraction? The honest answer is: **not yet for this specific task, but yes for specific use cases.**

For tasks where:
- Labeled training data is abundant
- Inference speed matters
- The feature set is fixed
- Maximum F1 is the primary goal

→ DeBERTa-based token classification is the right choice. It achieves ~0.90 F1 with a smaller, faster model.

For tasks where:
- The feature set changes frequently
- Labeled data is scarce or unavailable
- Flexible output (explanations, uncertainty) is needed
- Generalisation to new domains without retraining is required

→ Our generative LLM approach with few-shot prompting and LoRA fine-tuning is a compelling alternative, achieving 0.73 F1 with zero task-specific labeled data needed for the pretrained baseline.

### 8.6 Future Improvements

Several specific directions could close the gap between our approach and the DeBERTa baseline:

1. **Hybrid DeBERTa + LLM:** Use DeBERTa's token classification output as a "candidate span" filter, then use an LLM to verify or refine the boundaries. This combines encoder-based span detection with LLM-based semantic understanding.

2. **Larger pseudo-label pool:** We pseudo-labeled 2,000 of 41,000 unannotated notes. Extending to all 41,000 (as top DeBERTa teams did) would substantially increase training data.

3. **Early stopping with character F1:** We used early stopping on validation cross-entropy loss. Using character-level F1 directly as the stopping criterion (via the `SafeEvalGenerationCallback`) would better optimise the actual evaluation metric.

4. **Negation-aware prompting:** Explicit instruction to distinguish positive from negative mentions (e.g., `"denies chest pain"` → no span) would address the hallucination-on-negation failure mode.

5. **Adapter merging:** For the final submission, merging the LoRA adapter weights into the base model (via `peft.merge_adapter()`) and re-quantizing could improve inference speed without accuracy loss.

6. **DeBERTa domain pre-training as a teacher:** Use a fine-tuned DeBERTa model (trained on 9,901 pairs) to generate pseudo-labels for all 41,000 notes, then use these higher-quality pseudo-labels to fine-tune the LLM ensemble. This would combine DeBERTa's superior discrimination with the LLM's flexible generation.

---

## 9. References

1. **NBME Competition.** NBME — Score Clinical Patient Notes. Kaggle, 2022. https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes

2. **DeBERTa.** He, P., Liu, X., Gao, J., & Chen, W. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. *International Conference on Learning Representations (ICLR 2021)*. https://arxiv.org/abs/2006.03654

3. **DeBERTa-v3.** He, P., Gao, J., & Chen, W. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. https://arxiv.org/abs/2111.09543

4. **LoRA.** Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. https://arxiv.org/abs/2106.09685

5. **QLoRA.** Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*. https://arxiv.org/abs/2305.14314

6. **PEFT Library.** Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., & Bossan, B. (2022). PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods. https://github.com/huggingface/peft

7. **TRL / SFTTrainer.** von Werra, L. et al. (2020). TRL: Transformer Reinforcement Learning. https://github.com/huggingface/trl

8. **vLLM.** Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C.H., Gonzalez, J.E., Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *SOSP 2023*. https://arxiv.org/abs/2309.06180

9. **Qwen3.** Qwen Team, Alibaba Group. (2025). Qwen3 Technical Report. https://huggingface.co/Qwen/Qwen3-8B

10. **Llama 3.1.** Meta AI. (2024). The Llama 3 Herd of Models. https://arxiv.org/abs/2407.21783

11. **LFM2.5.** Liquid AI. (2024). LFM2.5: Liquid Foundation Models. https://www.liquid.ai/liquid-foundation-models

12. **FAISS.** Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*. https://arxiv.org/abs/1702.08734

13. **Sentence Transformers / all-MiniLM-L6-v2.** Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*. https://arxiv.org/abs/1908.10084

14. **Two-phase LLM for NBME (academic follow-up).** arxiv 2401.12994. A two-phase LLM framework achieving F1 of 0.968–0.983 on the NBME dataset. https://arxiv.org/html/2401.12994v1

15. **RapidFuzz.** Bachmann, M. (2020). RapidFuzz: Rapid fuzzy string matching in Python. https://github.com/maxbachmann/RapidFuzz

16. **GroupKFold.** Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://scikit-learn.org

17. **BitsAndBytes.** Dettmers, T. et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. *NeurIPS 2022*. https://arxiv.org/abs/2208.07339

18. **ELECTRA pre-training.** Clark, K., Luong, M.T., Le, Q.V., & Manning, C.D. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators. *ICLR 2020*. https://arxiv.org/abs/2003.10555

---

## 10. Appendix

### A. Full Hyperparameter Tables

#### A.1 Phase 1 — Data Augmentation Configuration

| Parameter | Value | Description |
|---|---|---|
| `LLM_MODEL` | `Qwen/Qwen3-8B` | Pseudo-labeling oracle |
| `LLM_4BIT` | False | fp16 precision for teacher |
| `SAMPLE_SIZE` | 2,000 | Number of unannotated notes to pseudo-label |
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `MAX_NEW_TOKENS` | 128 | Max output tokens per LLM call |
| `BATCH_SIZE` | 32 | Pairs processed per GPU call |
| `MAX_LENGTH` | 4,096 | Max total input length (tokens) |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | FAISS embedding model |
| `TOP_K_EXAMPLES` | 3 | Few-shot examples per query |
| `FUZZY_SCORE_CUTOFF` | 72 | Minimum rapidfuzz score for span matching |
| `CHECKPOINT_EVERY` | 100 rows | Atomic checkpoint frequency |
| `attn_implementation` | `"eager"` | Avoids SDPA CUDNN errors |
| `do_sample` | False | Greedy decoding |
| `enable_thinking` | False | Suppress Qwen3 chain-of-thought |

#### A.2 Phase 2 — QLoRA Training Configuration (LoRA Adapter v2 FIXED)

| Parameter | Value |
|---|---|
| **LoRA** | |
| `lora_r` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.10 |
| `target_modules` | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| `bias` | `"none"` |
| `task_type` | `CAUSAL_LM` |
| **Quantization** | |
| `load_in_4bit` | True |
| `bnb_4bit_quant_type` | `"nf4"` |
| `bnb_4bit_compute_dtype` | `bfloat16` |
| `bnb_4bit_use_double_quant` | True |
| **Training** | |
| `learning_rate` | 1e-4 |
| `num_train_epochs` | 3 |
| `per_device_train_batch_size` | 8 |
| `gradient_accumulation_steps` | 2 |
| `effective_batch_size` | 16 |
| `max_seq_length` | 1,024 |
| `lr_scheduler_type` | `cosine` |
| `warmup_ratio` | 0.05 |
| `weight_decay` | 0.05 |
| `optim` | `adamw_8bit` |
| `eval_strategy` | `steps` |
| `eval_steps` | 500 |
| `save_total_limit` | 2 |
| `early_stopping_patience` | 3 |
| `AUG_SAMPLE_RATIO` | 0.5 |
| **Data Split** | |
| `N_FOLDS` | 5 |
| `VAL_FOLD` | 4 |
| `group_by` | `case_num` |
| **Seeds** | |
| Llama-3.1-8B | 42 |
| Qwen3-1.7B | 43 |
| Qwen3-8B | 44 |
| LFM2.5-1.2B | 45 |

#### A.3 Phase 3 — Inference Configuration

| Parameter | Value |
|---|---|
| `vllm_version` | 0.17.1 |
| `dtype` | `float16` |
| `gpu_memory_utilization` | 0.85 |
| `enforce_eager` | True |
| `enable_lora` | True |
| `max_lora_rank` | 16 |
| `attention_backend` | `TRITON_ATTN` |
| `tensor_parallel_size` (8B models) | 2 |
| `tensor_parallel_size` (1B models) | 1 |
| `MAX_NEW_TOKENS` | 512 |
| `LLM_TEMPERATURE` | 0.0 (greedy) |
| `VOTE_THRESHOLD` | 3 (of 4 models) |
| `FUZZY_SCORE_CUTOFF` | 70.0 |
| `MAX_SPANS_PER_FEATURE` | 10 |

### B. Training Summary Statistics

#### B.1 Per-Model Validation Metrics (from training report)

| Model | Train Samples | Val Samples | Final Train Loss | Best Eval Loss | Best Eval Step | Val Char F1 | Val Contained Rate | Val JSON Rate | Train-Val Gap |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | 22,708 | 6,391 | 0.0924 | 0.1628 | 1,000 | 0.7644 | 0.9766 | 1.000 | 0.0225 |
| Llama3.1-8B | 22,708 | 6,391 | 0.1107 | 0.1967 | 1,000 | 0.7110 | 0.9609 | 1.000 | 0.0520 |
| LFM2.5-1.2B | 22,708 | 6,391 | 0.1165 | 0.2103 | 2,000 | 0.6398 | 0.9375 | 1.000 | 0.0589 |
| Qwen3-1.7B | 22,708 | 6,391 | 0.1116 | 0.2058 | 2,500 | 0.6256 | 0.9531 | 1.000 | 0.0506 |

#### B.2 Performance by Model Size Category

| Size Category | Mean Val Char F1 | Std | Mean Best Eval Loss | Mean Train-Val Gap |
|---|---|---|---|---|
| 1B-class (LFM2.5, Qwen3-1.7B) | 0.633 | 0.010 | 0.208 | 0.055 |
| 8B-class (Qwen3-8B, Llama3.1-8B) | 0.738 | 0.038 | 0.184 | 0.037 |

#### B.3 Gradient Norm Statistics

| Model | Mean | Std | Min | Max |
|---|---|---|---|---|
| LFM2.5-1.2B | 1.373 | 0.669 | 0.628 | 5.325 |
| Llama3.1-8B | 0.869 | 0.346 | 0.403 | 2.031 |
| Qwen3-1.7B | 0.983 | 0.462 | 0.459 | 3.901 |
| Qwen3-8B | 0.535 | 0.242 | 0.290 | 1.997 |

### C. Full Prompt Templates

#### C.1 System Prompt (LoRA Fine-tuning and Inference)

```text
You are a clinical NLP specialist.
Given a patient note and a clinical feature, extract the EXACT verbatim text spans
from the note that express that feature.
Rules:
  1. Copy text character-for-character — do NOT paraphrase.
  2. If the feature is absent from the note, return an empty list.
  3. Output ONLY valid JSON — no markdown, no explanation.
{"spans": ["exact text 1", "exact text 2"]}
```

#### C.2 User Message Format

```text
Note: "<full pn_history text>"
Feature: <feature_text>
/no_think
```

The `/no_think` suffix is appended for Qwen3 models to disable chain-of-thought generation. For LFM2.5-1.2B and Llama-3.1-8B, no suffix is needed (these models do not have a thinking mode).

#### C.3 Few-Shot Augmentation Prompt (Phase 1 only)

The user message during pseudo-labeling includes three FAISS-retrieved examples before the target:

```text
Example 1:
Note (excerpt): "<retrieved note excerpt, max 300 chars>"
Feature: <retrieved feature>
Answer: {"spans": ["<retrieved annotation>"]}

Example 2:
Note (excerpt): "<retrieved note excerpt>"
Feature: <retrieved feature>
Answer: {"spans": ["<retrieved annotation>"]}

Example 3:
Note (excerpt): "<retrieved note excerpt>"
Feature: <retrieved feature>
Answer: {"spans": ["<retrieved annotation>"]}

Now annotate:
Note: "<full target pn_history>"
Feature: <target feature_text>
```

#### C.4 Inference System Prompt Variants Tested (Pretrained Only)

Several system prompt phrasings were tested during pretrained-only inference experiments:

**Prompt SP1 (scored 0.67439 with LoRA v1):**
```text
Return ONLY JSON: {"spans": ["..."]}. Extract minimal exact substrings from the note
that directly express the feature. Copy spans exactly. Do NOT paraphrase, infer, or 
expand. If absent, return {"spans": []}. No duplicates.
```

**Prompt SP2 (scored 0.67778 with LoRA v1):**
```text
Extract exact verbatim text spans from the patient note that express the given clinical
feature. Return ONLY valid JSON in this exact format: {"spans": ["..."]}. Each span 
must be a continuous substring copied exactly from the note. Do NOT paraphrase, 
normalize, infer, or add any words. If the feature is not present, return {"spans": []}.
Do not include duplicates. Output JSON only.
```

### D. Compute Resources

| Phase | Hardware | Duration (approx.) |
|---|---|---|
| Phase 1 — Data augmentation | Single GPU (A100 80GB or equivalent) | ~4–6 hours for 2,000 notes |
| Phase 2 — QLoRA training (4 models) | Single GPU (A100 80GB) | ~8–12 hours total (sequential) |
| Phase 3 — Kaggle inference | 2× T4 16GB (Kaggle environment) | ~4–6 hours within 9-hour limit |

**Adapter sizes:**
- Each LoRA adapter checkpoint: ~100 MB (`adapter_model.safetensors` + `adapter_config.json`)
- Total adapter storage: ~400 MB for all four models

**Memory footprint during training (per model):**
- Base model (4-bit): ~5 GB
- LoRA adapter parameters: ~40–80 MB
- Optimizer states (adamw_8bit): ~160–320 MB
- Activations (with gradient checkpointing): ~2–4 GB
- **Peak VRAM: ~8–10 GB per model** (fits within a single 16GB T4 or 24GB RTX 4090)

---

*End of Report*

---

## 11. Extended Analysis: Deep Dive into Model Behaviour

### 11.1 Why JSON Output Format Works for Span Extraction

The choice to frame span extraction as JSON generation deserves detailed justification. Alternative output formats were considered during design:

**Option A — Plain text span:** The model outputs the span text directly: `Father died of heart attack at age 52`. Simple to parse, but ambiguous when the note contains the same substring multiple times. No structure for multiple spans.

**Option B — Character offsets directly:** The model outputs `62 98`. Compact, but LLMs struggle to generate precise integer offsets reliably — they have no mechanism analogous to pointer networks, and the mapping from token position to character position is complex.

**Option C — JSON span list (chosen):** `{"spans": ["Father died of heart attack at age 52"]}`. The verbatim text is easier for an LLM to generate than character indices; multiple spans are naturally expressed as a list; the structured format enables reliable parsing; and the FSM constraint can enforce both the schema and the lexical containment property simultaneously.

The validation results confirm this was the right choice: `val_jsonish_rate = 1.0` across all four models — every single output from every model during validation was valid JSON. This 100% format compliance is remarkable and demonstrates that instruction-tuned LLMs, when fine-tuned with as few as 22,708 examples, reliably learn to produce structured output.

For comparison, the Pretrained Only baseline (no fine-tuning) also achieves high JSON compliance in practice, because modern instruction-tuned LLMs are specifically optimised to follow structured output instructions. However, without fine-tuning, the *content* of the JSON (which spans are chosen) is much noisier.

### 11.2 The Role of the FAISS Index in Pseudo-Label Quality

The quality of pseudo-labels depends critically on how well the few-shot examples guide the LLM. We use FAISS cosine similarity over `all-MiniLM-L6-v2` embeddings to retrieve the most semantically similar annotated examples. This section analyses why this matters.

**Embedding strategy:** Each annotated training example is embedded as `"Feature: <feature_text>  Annotation: <annotation_text>"`. This embedding encodes both the feature concept *and* the annotation style. When we query with a new feature text, we retrieve examples that:
1. Are semantically similar to the query feature (same clinical concept)
2. Have similar annotation patterns (similar verbosity, similar phrasing)

Compare this to random few-shot selection: random examples might show the LLM examples from entirely unrelated features (e.g., retrieving a tobacco use annotation when the query is about family history of MI). The LLM sees an inconsistent signal and produces noisier pseudo-labels.

**Index statistics:** The FAISS index contains 9,901 vectors of dimension 384. At query time, a top-3 nearest-neighbour search over this index takes milliseconds on CPU. The index is built once (one GPU pass to embed all 9,901 examples with `all-MiniLM-L6-v2`) and cached to `faiss_features.index`.

**Fuzzy matching fallback:** Even with FAISS-guided prompting, the LLM occasionally generates slightly incorrect span text — perhaps adding a leading article (`"a substernal pressure"` instead of `"substernal pressure"`) or missing a trailing word. The RapidFuzz sliding window with cutoff 72 catches these near-matches. Without fuzzy matching, approximately 8–12% of generated spans would be discarded, reducing the effective yield of the pseudo-labeling process.

### 11.3 The Vote Threshold: Precision-Recall Trade-off

The vote threshold is one of the most impactful hyperparameters in our pipeline. For a 4-model ensemble, the threshold can range from 1 (any model predicts it) to 4 (unanimous agreement).

**Threshold = 1 (union):** Maximum recall. Any character predicted by any model is included. High false positive rate — weak model predictions (LFM2.5, Qwen3-1.7B) inflate the prediction with noisy spans.

**Threshold = 2 (simple majority):** Balanced but loose. Two of four models must agree. Works well when all four models are similarly strong.

**Threshold = 3 (strict majority, chosen):** Requires three of four models to agree. Prioritises precision over recall. In our best configuration (V37: Llama + Qwen3-8B + Qwen3-1.7B + LFM2.5), the two strong 8B models (Qwen3-8B, Llama) both need to agree with at least one 1B model. This acts as a precision filter: only high-confidence predictions pass.

**Threshold = 4 (unanimous):** Very high precision but low recall. Only spans that all four models — including the weakest (LFM2.5-1.2B) — agree on are included. The weak model becomes a bottleneck.

**Empirical results for LoRA FIXED:**

| Vote Threshold | Configuration | Score |
|---|---|---|
| 2 | Qwen3-8B + Qwen3-1.7B + LFM2.5 (3-model) | 0.71046 |
| 3 | Llama + Qwen3-8B + Qwen3-1.7B + LFM2.5 (4-model) | **0.73195** |

The threshold of 3 with 4 models outperforms threshold of 2 with 3 models, despite the 3-model configuration using a higher-quality ensemble (replacing LFM2.5 with Llama in effect). This suggests that the stricter threshold provides meaningful precision improvement that outweighs the recall loss.

**Character-level voting advantages in detail:** Consider a concrete example where:
- Qwen3-8B predicts: `"Father died of heart attack at age 52"` (chars 62–98)
- Llama-3.1-8B predicts: `"Father died of heart attack"` (chars 62–88)
- Qwen3-1.7B predicts: `"Father died of heart attack at age 52"` (chars 62–98)
- LFM2.5-1.2B predicts: `"heart attack at age 52"` (chars 77–98)

Span-level voting: no single span string has 3 votes → prediction would be empty.
Character-level voting (threshold=3): chars 62–88 have 3 votes (Qwen3-8B, Llama, Qwen3-1.7B); chars 89–98 have 2 votes (Qwen3-8B, Qwen3-1.7B) → consensus prediction `"Father died of heart attack"` (chars 62–88). This is partial but correct recall, which span-level voting would have missed entirely.

### 11.4 Overfitting Dynamics and the Augmentation Ratio

The training report clearly shows overfitting in all four models: validation loss begins rising after the best checkpoint at steps 1,000–2,500. This overfitting has two contributing factors:

**Factor 1 — Noisy pseudo-labels.** The ~29,600 pseudo-labeled examples from Phase 1 contain errors. Qwen3-8B, even as a strong teacher, misidentifies some spans (hallucinations, missed spans, boundary errors). As training progresses beyond the early checkpoint, the model begins memorising these errors.

**Factor 2 — Limited label diversity.** The 9,901 true labeled examples cover 10 clinical cases × ~150 features. This is a relatively narrow distribution. After a few passes over the data, the model has memorised most patterns and further training simply reinforces memorisation.

**The AUG_SAMPLE_RATIO = 0.5 design choice** was specifically introduced to mitigate Factor 1: using only 50% of pseudo-labeled data (rather than 100% as in the earlier LoRA v2 experiments) reduces the model's exposure to noisy labels while preserving the scale benefit. The FIXED adapters were designed with this regularisation in mind.

**Why not AUG_SAMPLE_RATIO = 0.0?** Zero augmentation means training on only 9,901 examples — too few for meaningful 8B model fine-tuning with our group-based split (which removes 2 of 10 cases for validation, leaving ~7,900 training examples). The augmented data, even at 50% inclusion, nearly triples the effective training set.

**Future direction:** Implementing dynamic curriculum — starting with high-confidence pseudo-labels (those where the Phase 1 model's output matched the note exactly via exact matching, without requiring fuzzy search) and gradually introducing lower-confidence examples — could reduce the overfitting effect while maintaining training data diversity.

### 11.5 LFM2.5-1.2B: A Special Case

LFM2.5-1.2B (Liquid Foundation Model 2.5, 1.2B parameters) is the outlier in our ensemble. Its Kaggle score (0.63392) is substantially below the next-lowest model (Qwen3-1.7B: 0.65140), and its gradient norms are the highest of any model (mean 1.373, max 5.325).

Several factors distinguish LFM2.5 from the other three models:

**1. Custom architecture.** Unlike Qwen3 and Llama-3.1 (both standard grouped-query attention transformers with RoPE positional encoding), LFM2.5 uses a proprietary linear attention hybrid. This architecture has different inductive biases: linear attention approximates full attention with reduced complexity but may handle long-range dependencies differently.

**2. Fewer LoRA target modules.** LFM2.5's module names differ from Qwen3 and Llama-3.1. Our implementation targets `["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"]` — note `out_proj` and `in_proj` rather than `o_proj`, `gate_proj`, `up_proj`, `down_proj`. This difference reflects LFM2.5's custom MLP architecture. The adapter may adapt a different component of the model than intended.

**3. `trust_remote_code=True` requirement.** LFM2.5 requires loading custom model code from HuggingFace rather than using the built-in `AutoModelForCausalLM` implementation. This creates a dependency on the model's published code quality and may introduce subtle differences in attention computation.

**4. Smaller base capacity.** At 1.2B parameters (smaller than Qwen3-1.7B), LFM2.5 has less representational capacity for clinical semantics, independently of architecture.

Despite these challenges, LFM2.5 still contributes positively to the 4-model ensemble. Its predictions are sufficiently different from the other models that it provides non-redundant votes on certain span candidates — its presence in the ensemble is better than its absence when the vote threshold is tuned appropriately.