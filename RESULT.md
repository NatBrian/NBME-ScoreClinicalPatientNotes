Results:

### Pretrained Only Model with Few Shot

1. 
Models: Qwen3-1.7B, Qwen3-8B, LiquidAI/LFM2.5-1.2B-Instruct
Score: 0.53125

2.
Models: Meta-Llama-3.1-8B-Instruct, Qwen3-8B, Mistral-7B-Instruct-v0.3
Score: 0.53762

### With LoRA Adapater

1. 
Models: Qwen3-1.7B, Qwen3-8B, LiquidAI/LFM2.5-1.2B-Instruct
Score: 0.67146, 0.67173, 
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

2. 
Models: Qwen3-1.7B, Qwen3-8B, LiquidAI/LFM2.5-1.2B-Instruct
Score: 0.67439
SYSTEM_PROMPT = (
    "Return ONLY JSON: {\"spans\": [\"...\"]}. "
    "Extract minimal exact substrings from the note that directly express the feature. "
    "Copy spans exactly. Do NOT paraphrase, infer, or expand. "
    "If absent, return {\"spans\": []}. "
    "No duplicates."
)

3. 
Models: Qwen3-1.7B, Qwen3-8B, LiquidAI/LFM2.5-1.2B-Instruct
Score: 0.67778
SYSTEM_PROMPT = (
    "Extract exact verbatim text spans from the patient note that express the given clinical feature. "
    "Return ONLY valid JSON in this exact format: {\"spans\": [\"...\"]}. "
    "Each span must be a continuous substring copied exactly from the note. "
    "Do NOT paraphrase, normalize, infer, or add any words. "
    "If the feature is not present, return {\"spans\": []}. "
    "Do not include duplicates. "
    "Output JSON only."
)

4. Few Shot
Models: Qwen3-1.7B, Qwen3-8B, LiquidAI/LFM2.5-1.2B-Instruct
Score: 0.66659
SYSTEM_PROMPT = (
    "Extract exact verbatim spans from the note that express the feature.\n"
    "Return only JSON: {\"spans\": [\"...\"]}.\n"
    "Rules: copy text exactly; no paraphrase; no explanation; no duplicates; empty list if absent.\n\n"
    "Example:\n"
    "Note: \"Patient reports chest pain and nausea.\"\n"
    "Feature: chest pain\n"
    "{\"spans\": [\"chest pain\"]}"
)
