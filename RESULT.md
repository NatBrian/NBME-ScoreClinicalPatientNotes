# Results

| Model / Combination                                            | Pretrained Only + Few Shot | LoRA Adapter       | LoRA Adapter v2   | With LoRA Adapter v2 (FIXED) | Full Fine Tuning |
| -------------------------------------------------------------- | --------------------------: | ------------------: | ----------------: | ----------------------------: | ----------------: |
| Qwen3-1.7B                                                     | 0.48093 (V5)               | 0.60103 (V16)      | 0.64721 (V25)     | 0.65140 (V41)                 | 0.11256 (V4)      |
| Qwen3-8B                                                       | 0.49500 (V3)               | 0.66226 (V10)      | 0.66552 (V24)     | 0.71670 (V40)                 | 0.50343 (V5)      |
| Llama-3.1-8B                                                   | -                          | 0.64465 (V11)      | 0.66515 (V23)     | 0.68951 (V39)                 | -                 |
| LFM2.5-1.2B                                                    | -                          | 0.43482 (V17)      | 0.62867 (V26)     | 0.63392 (V42)                 | 0.49203 (V3)      |
| Qwen3-1.7B + Qwen3-8B + LFM2.5-1.2B                            | 0.53125 (V6)               | 0.67778 (V6 SP2)   | 0.67900 (V22)     | 0.71046 (V38)                 | 0.62569 (V2)      |
| Qwen3-8B + Llama-3.1-8B + Qwen3-1.7B + LFM2.5-1.2B             | -                          | 0.70143 (V15)      | 0.69971 (V21)     | 0.73195 (V37)                 | -                 |
| Qwen3-8B x2 + Llama-3.1-8B x2 + Qwen3-1.7B + LFM2.5-1.2B       | -                          | **0.70180 (V14)**  | 0.69419 (V29)     | 0.7X (temp)                            | -                 |
| Qwen3-8B x2 + Llama-3.1-8B x2 + Qwen3-1.7B x2 + LFM2.5-1.2B x2 | -                          | 0.70160 (V18)      | 0.69909 (V27)     | 0.7X (temp)                            | -                 |
| Llama-3.1-8B + Qwen/Qwen3-8B + Mistral-7B                      | 0.53762 (V2)               | -                  | -                 | -                            | -                 |

## Pretrained Only Model with Few Shot

| # | Models | VOTE_THRESHOLD | Score | Submission | 
|---|--------|----------------|-------|------------| 
| 1 | Qwen3-1.7B | 1 | 0.48093 | Version 5 | 
| 2 | Qwen3-8B | 1 | 0.49500 | Version 3 | 
| 3 | Qwen3-1.7B, Qwen3-8B, LFM2.5-1.2B | 2 | 0.53125 | Version 6 | 
| 4 | Llama-3.1-8B, Qwen3-8B, Mistral-7B | 2 | 0.53762 | Version 2 |

1.
   - **Models:** Qwen/Qwen3-1.7B, Qwen/Qwen3-8B, LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 2
   - **Score:** 0.53125
   - **Submission:** `3_kaggle_inference_pretrained_only-nbme-score - Version 6`

2.
   - **Models:** meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, Mistral-7B-Instruct-v0.3
   - **VOTE_THRESHOLD:** 2
   - **Score:** 0.53762
   - **Submission:** `3_kaggle_inference_pretrained_only-nbme-score_v2 - Version 2`

3.
   - **Models:** Qwen/Qwen3-8B
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.49500
   - **Submission:** `3_kaggle_inference_pretrained_only-nbme-score_v2 - Version 3`

4.
   - **Models:** Qwen/Qwen3-1.7B
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.48093
   - **Submission:** `3_kaggle_inference_pretrained_only-nbme-score_v2 - Version 5`

## With LoRA Adapter

| # | Models | VOTE_THRESHOLD | Score | Submission |
|---|--------|----------------|-------|------------|
| 1 | LFM2.5-1.2B | 1 | 0.43482 | Ver 17 |
| 2 | Qwen3-1.7B | 1 | 0.60103 | Ver 16 |
| 3 | Qwen3-8B | 1 | 0.66226 | Ver 10 |
| 4 | Llama-3.1-8B | 1 | 0.64465 | Ver 11 |
| 5 | Qwen3-8B x3 | 2 | 0.66215 | Ver 5 |
| 6 | Qwen3-1.7B + Qwen3-8B + LFM2.5-1.2B | 2 | 0.67146 + 0.67173 | Ver 3 + 4 |
| 7 | Qwen3-1.7B + Qwen3-8B + LFM2.5-1.2B (SP1) | 2 | 0.67439 | Ver 7 |
| 8 | Qwen3-1.7B + Qwen3-8B + LFM2.5-1.2B (SP2) | 2 | 0.67778 | Ver 6 |
| 9 | Qwen3-1.7B + Qwen3-8B + LFM2.5-1.2B + Few Shot | 2 | 0.66659 | Ver 8 |
| 10 | Qwen3-8B + Llama-3.1-8B + Qwen3-1.7B + LFM2.5-1.2B | 3 | 0.70143 | Ver 15 |
| 11 | Qwen3-8B x2 + Llama-3.1-8B x2 + Qwen3-1.7B + LFM2.5-1.2B | 4 | 0.70180 | Ver 14 |
| 12 | Qwen3-8B x3 + Llama-3.1-8B x3 | 4 | 0.66286 | Ver 19 |
| 13 | Qwen3-8B x2 + Llama-3.1-8B x2 + Qwen3-1.7B x2 + LFM2.5-1.2B x2 | 5 | 0.70160 | Ver 18 |
| 14 | Qwen3-8B x4 + Llama-3.1-8B x4 | 5 | 0.66261 | Ver 20 |

1.
   - **Models:** Qwen/Qwen3-1.7B, Qwen/Qwen3-8B, LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 2
   - **Score:** 0.67146, 0.67173
   - **Submission:**
     - `3_kaggle_inference-nbme-score-clinical-v2 - Version 3`
     - `3_kaggle_inference-nbme-score-clinical-v2 - Version 4`

1.
   - **Models:** Qwen/Qwen3-8B + Qwen/Qwen3-8B + Qwen/Qwen3-8B
   - **VOTE_THRESHOLD:** 2
   - **Score:** 0.66215
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 5`

2.
   - **Models:** Qwen/Qwen3-1.7B, Qwen/Qwen3-8B, LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 2
   - **Score:** 0.67439
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 7`
   - **SYSTEM_PROMPT:**

```text
SYSTEM_PROMPT = (
    "Return ONLY JSON: {\"spans\": [\"...\"]}. "
    "Extract minimal exact substrings from the note that directly express the feature. "
    "Copy spans exactly. Do NOT paraphrase, infer, or expand. "
    "If absent, return {\"spans\": []}. "
    "No duplicates."
)
```

3.
   - **Models:** Qwen/Qwen3-1.7B, Qwen/Qwen3-8B, LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 2
   - **Score:** 0.67778
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 6`
   - **SYSTEM_PROMPT:**

```text
SYSTEM_PROMPT = (
    "Extract exact verbatim text spans from the patient note that express the given clinical feature. "
    "Return ONLY valid JSON in this exact format: {\"spans\": [\"...\"]}. "
    "Each span must be a continuous substring copied exactly from the note. "
    "Do NOT paraphrase, normalize, infer, or add any words. "
    "If the feature is not present, return {\"spans\": []}. "
    "Do not include duplicates. "
    "Output JSON only."
)
```

4. Few Shot
   - **Models:** Qwen/Qwen3-1.7B, Qwen/Qwen3-8B, LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 2
   - **Score:** 0.66659
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 8`
   - **SYSTEM_PROMPT:**

```text
SYSTEM_PROMPT = (
    "Extract exact verbatim spans from the note that express the feature.\n"
    "Return only JSON: {\"spans\": [\"...\"]}.\n"
    "Rules: copy text exactly; no paraphrase; no explanation; no duplicates; empty list if absent.\n\n"
    "Example:\n"
    "Note: \"Patient reports chest pain and nausea.\"\n"
    "Feature: chest pain\n"
    "{\"spans\": [\"chest pain\"]}"
)
```

5.
   - **Models:** Qwen/Qwen3-8B
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.66226
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 10`

5.
   - **Models:** meta-llama/Llama-3.1-8B-Instruct
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.64465
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 11`

5.
   - **Models:** Qwen/Qwen3-8B x2 + meta-llama/Llama-3.1-8B-Instruct x2 + Qwen/Qwen3-1.7B + LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 4
   - **Score:** 0.70180
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 14`

5.
   - **Models:** Qwen/Qwen3-8B + meta-llama/Llama-3.1-8B-Instruct + Qwen/Qwen3-1.7B + LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 3
   - **Score:** 0.70143
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 15`

5.
   - **Models:** Qwen/Qwen3-1.7B
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.60103
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 16`

5.
   - **Models:** LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.43482
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 17`

6.
   - **Models:** Qwen/Qwen3-8B x2 + meta-llama/Llama-3.1-8B-Instruct x2 + Qwen/Qwen3-1.7B x2 + LiquidAI/LFM2.5-1.2B-Instruct x2
   - **VOTE_THRESHOLD:** 5
   - **Score:** 0.70160
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 18`

7.
   - **Models:** Qwen/Qwen3-8B x3 + meta-llama/Llama-3.1-8B-Instruct x3
   - **VOTE_THRESHOLD:** 4
   - **Score:** 0.66286
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 19`

8.
   - **Models:** Qwen/Qwen3-8B x4 + meta-llama/Llama-3.1-8B-Instruct x4
   - **VOTE_THRESHOLD:** 5
   - **Score:** 0.66261
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 20`

## With LoRA Adapter v2

| # | Models | VOTE_THRESHOLD | Score | Submission |
|---|--------|----------------|-------|------------|
| 1 | LFM2.5 | 1 | 0.62867 | Ver 26 |
| 2 | Qwen3-1.7B | 1 | 0.64721 | Ver 25 |
| 3 | Qwen3-8B | 1 | 0.66552 | Ver 24 |
| 4 | Llama-3.1-8B | 1 | 0.66515 | Ver 23 |
| 5 | Qwen3-8B + Qwen3-1.7B + LFM2.5 | 2 | 0.67900 | Ver 22 |
| 6 | Qwen3-8B + Llama-3.1-8B + Qwen3-1.7B + LFM2.5 | 3 | 0.69944, 0.69971 | Ver 21 |
| 7 | Qwen3-8B x2 + Llama-3.1-8B x2 + Qwen3-1.7B + LFM2.5 | 4 | 0.69419 | Ver 29 |
| 8 | Qwen3-8B x2 + Llama-3.1-8B x2 + Qwen3-1.7B x2 + LFM2.5 x2 | 5 | 0.69909 | Ver 27 |
| 9 | Qwen3-8B x3 + Llama-3.1-8B x3 + Qwen3-1.7B x3 + LFM2.5 x3 | 7 | 0.69930 | Ver 28 |

1.
   - **Models:** Qwen/Qwen3-8B + meta-llama/Llama-3.1-8B-Instruct + Qwen/Qwen3-1.7B + LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 3
   - **Score:** 0.69944, score: 0.69971
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 21`

2.
   - **Models:** Qwen/Qwen3-8B + Qwen/Qwen3-1.7B + LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 2
   - **Score:** 0.67900
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 22`

3.
   - **Models:** meta-llama/Llama-3.1-8B-Instruct
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.66515
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 23`

4.
   - **Models:** Qwen/Qwen3-8B
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.66552
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 24`

5.
   - **Models:** Qwen/Qwen3-1.7B
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.64721
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 25`

6.
   - **Models:** LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.62867
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 26`

7.
   - **Models:** Qwen/Qwen3-8B x2 + meta-llama/Llama-3.1-8B-Instruct x2 + Qwen/Qwen3-1.7B x2 + LiquidAI/LFM2.5-1.2B-Instruct x2
   - **VOTE_THRESHOLD:** 5
   - **Score:** 0.69909
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 27`

8.
   - **Models:** Qwen/Qwen3-8B x3 + meta-llama/Llama-3.1-8B-Instruct x3 + Qwen/Qwen3-1.7B x3 + LiquidAI/LFM2.5-1.2B-Instruct x3
   - **VOTE_THRESHOLD:** 7
   - **Score:** 0.69930
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 28`

9.
   - **Models:** Qwen/Qwen3-8B x2 + meta-llama/Llama-3.1-8B-Instruct x2 + Qwen/Qwen3-1.7B + LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 4
   - **Score:** 0.69419
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 29`


## With LoRA Adapter v2 (FIXED)
1.
   - **Models:** Qwen/Qwen3-8B + Qwen/Qwen3-1.7B + LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 2
   - **Score:** 0.71046
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 38`

2.
   - **Models:** meta-llama/Llama-3.1-8B-Instruct + Qwen/Qwen3-8B + Qwen/Qwen3-1.7B + LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 3
   - **Score:** 0.73195
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 37`

3.
   - **Models:** meta-llama/Llama-3.1-8B-Instruct
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.68951
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 39`

4.
   - **Models:** Qwen/Qwen3-8B
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.71670
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 40`

5.
   - **Models:** Qwen/Qwen3-1.7B
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.65140
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 41`

6.
   - **Models:** LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.63392
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 42`

7.
   - **Models:** meta-llama/Llama-3.1-8B-Instruct x2 + Qwen/Qwen3-8B x2 + Qwen/Qwen3-1.7B + LiquidAI/LFM2.5-1.2B-Instruct
   - **VOTE_THRESHOLD:** 4
   - **Score:** 
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 43`

8.
   - **Models:** meta-llama/Llama-3.1-8B-Instruct x2 + Qwen/Qwen3-8B x2 + Qwen/Qwen3-1.7B x2 + LiquidAI/LFM2.5-1.2B-Instruct x2
   - **VOTE_THRESHOLD:** 5
   - **Score:**
   - **Submission:** `3_kaggle_inference-nbme-score-clinical-v2 - Version 44`


## Full Fine Tuning

| # | Models | VOTE_THRESHOLD | Score | Submission |
|---|--------|----------------|-------|------------|
| 1 | Qwen3-8B + Qwen3-1.7B + LFM2.5 | 2 | 0.62569 | Ver 2 |
| 2 | LFM2.5 | 1 | 0.49203 | Ver 3 |
| 3 | Qwen3-1.7B | 1 | 0.11256 | Ver 4 |
| 4 | Qwen3-8B | 1 | 0.50343 | Ver 5 |

1.
   - **Models:** Qwen/Qwen3-8B-finetuned + Qwen/Qwen3-1.7B-finetuned + LiquidAI/LFM2.5-1.2B-Instruct-finetuned
   - **VOTE_THRESHOLD:** 2
   - **Score:** 0.62569
   - **Submission:** `3-kaggle-inference-finetuned-nbme-score - Version 2`

2.
   - **Models:** LiquidAI/LFM2.5-1.2B-Instruct-finetuned
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.49203
   - **Submission:** `3-kaggle-inference-finetuned-nbme-score - Version 3`

3.
   - **Models:** Qwen/Qwen3-1.7B-finetuned
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.11256
   - **Submission:** `3-kaggle-inference-finetuned-nbme-score - Version 4`

4.
   - **Models:** Qwen/Qwen3-8B
   - **VOTE_THRESHOLD:** 1
   - **Score:** 0.50343
   - **Submission:** `3-kaggle-inference-finetuned-nbme-score - Version 5`