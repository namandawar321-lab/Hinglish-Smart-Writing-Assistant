# Hinglish Smart Writing Assistant: Start Guide (CS3235 Mini-Project #4)

Goal: Roman-script Hinglish -> native script (Devanagari) -> polished phrasing -> optional English translation, with a Gradio UI and 30-50 test cases.

Pipeline:

    Hinglish text -> IndicXlit (transliterate) -> LLM polish -> IndicTrans2 (to English) -> Gradio UI

---

## Step 0: Project setup (10 min, no GitHub needed)

    mkdir hinglish-assistant && cd hinglish-assistant
    python -m venv .venv
    # Windows: .venv\Scripts\activate    |  Mac/Linux: source .venv/bin/activate
    mkdir src data results
    touch src/smoke_test.py

Open the folder in Cursor and select the `.venv` interpreter.

## Step 1: Install dependencies

Install PyTorch for your CUDA version first (https://pytorch.org/get-started/locally/), then:

    pip install transformers sentencepiece accelerate gradio jiwer sacrebleu pandas
    pip install ai4bharat-transliteration
    pip install IndicTransToolkit

Notes:
- `ai4bharat-transliteration` (IndicXlit) depends on fairseq, which can break on newer Python versions. Python 3.9-3.10 is the safest. If it fails after ~30 min, use the fallback in Step 3.
- If `IndicTransToolkit` is not found on PyPI, install from GitHub: `pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git`
- Verify checkpoint names and licenses on Hugging Face before relying on them (as the handout says).

## Step 2: Smoke test (paste into `src/smoke_test.py`)

    import torch

    # ---------- 1. Transliteration (IndicXlit) ----------
    def test_xlit():
        from ai4bharat.transliteration import XlitEngine
        engine = XlitEngine("hi", beam_width=10, src_script_type="en")
        sents = [
            "kal meeting hai please send the report",
            "mujhe ye assignment samajh nahi aaya",
            "aaj raat movie dekhne chalte hain",
        ]
        for s in sents:
            print(s, "->", engine.translit_sentence(s))

    # ---------- 2. Translation (IndicTrans2, Hindi -> English) ----------
    def test_translate():
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError:
            from IndicTransToolkit.processor import IndicProcessor

        ckpt = "ai4bharat/indictrans2-indic-en-dist-200M"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ckpt, trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        ip = IndicProcessor(inference=True)

        sents = ["कल मीटिंग है, कृपया रिपोर्ट भेज दो।"]
        batch = ip.preprocess_batch(sents, src_lang="hin_Deva", tgt_lang="eng_Latn")
        inputs = tok(batch, truncation=True, padding="longest",
                     return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, num_beams=5, max_length=256)
        dec = tok.batch_decode(out, skip_special_tokens=True)
        print(ip.postprocess_batch(dec, lang="eng_Latn"))

    if __name__ == "__main__":
        print("GPU:", torch.cuda.is_available())
        test_xlit()
        test_translate()

Run: `python src/smoke_test.py`

Success = Devanagari output for the Hinglish sentences and an English sentence for the Hindi one.

## Step 3: If IndicXlit won't install (fallback)

Use the LLM for transliteration instead and treat it as the baseline. Prompt idea:

    Convert this Hinglish (Roman script) to Devanagari Hindi. Keep English words
    like "meeting", "report" in English (Latin) script. Output only the converted text.

Document this choice in the README and compare it against IndicXlit if you get it working later.

## Step 4: Pick the polish model

Try BharatGen Param1 from https://huggingface.co/bharatgenai. Check its size and license first. If it is too heavy for 12 GB VRAM, use a small instruct model (4-bit quantized if needed) and say so in the README.

## Step 5: Build order after the smoke test passes

1. `src/pipeline.py`: functions `transliterate()`, `polish()`, `translate()`
2. English-word detection (leave English words untouched). This is your improved pipeline vs the transliterate-everything baseline.
3. `app.py`: Gradio UI with one input box and three output panels
4. `data/test_cases.csv`: 40 rows with columns `hinglish, reference_devanagari, reference_english`
5. `src/evaluate.py`: CER and word accuracy (jiwer), chrF/BLEU (sacrebleu), baseline vs improved table
6. `results/failure_analysis.md`: 8-10 failures grouped by cause (English words, ambiguous spellings, slang/SMS spellings)
7. `README.md`: setup, demo screenshot, results table, limitations

## Test case tips

- Write about 40 Hinglish sentences across categories: casual chat, work messages, mixed English words, SMS-style spelling, numbers/dates.
- Have an LLM draft the Devanagari references, then fix them by hand. Do not trust them blindly.

## Rubric checklist

- [ ] Problem definition and relevance
- [ ] Model/dataset selection (justify each model)
- [ ] Working end-to-end app
- [ ] GenAI integration (LLM polish step)
- [ ] 30-50 test cases, quantitative results, failure analysis
- [ ] Demo, documentation, repository

## Cursor tip

Paste this file into Cursor chat and say: "Follow START_HERE.md and implement Step 5 file by file, starting with pipeline.py."
