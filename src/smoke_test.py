import gc
import os
import sys
from pathlib import Path

import torch

# Keep Hugging Face downloads on D:. The system drive is short on space.
_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HOME", str(_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_ROOT / ".cache" / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(_ROOT / ".cache" / "torch"))
if str(_ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(_ROOT / "vendor"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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

    from src.pipeline import TRANSLATION_CANDIDATES, _patch_indictrans_tokenizer

    _patch_indictrans_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = model = None
    for ckpt in TRANSLATION_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                ckpt, trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            print("Translation checkpoint:", ckpt)
            break
        except Exception as exc:
            print(f"Could not load {ckpt}: {type(exc).__name__}: {exc}")
    if model is None:
        raise RuntimeError("No IndicTrans2 checkpoint could be loaded.")
    ip = IndicProcessor(inference=True)

    sents = ["कल मीटिंग है, कृपया रिपोर्ट भेज दो।"]
    batch = ip.preprocess_batch(sents, src_lang="hin_Deva", tgt_lang="eng_Latn")
    inputs = tok(batch, truncation=True, padding="longest",
                 return_tensors="pt").to(device)
    with torch.no_grad():
        # This checkpoint predates the Cache API. An empty DynamicCache
        # makes past_key_values[0][0] None and crashes the decoder.
        out = model.generate(**inputs, num_beams=5, max_length=256, use_cache=False)
    dec = tok.batch_decode(out, skip_special_tokens=True)
    print(ip.postprocess_batch(dec, lang="eng_Latn"))


def test_xlit_fallback(sentences):
    """Step 3 fallback when fairseq/IndicXlit cannot be installed."""
    print("IndicXlit is unavailable. Using the LLM transliteration fallback.")
    from src.pipeline import get_assistant

    assistant = get_assistant()
    for sentence in sentences:
        print(sentence, "->", assistant.transliterate(sentence, preserve_english=False))


if __name__ == "__main__":
    print("GPU:", torch.cuda.is_available())
    samples = [
        "kal meeting hai please send the report",
        "mujhe ye assignment samajh nahi aaya",
        "aaj raat movie dekhne chalte hain",
    ]
    # Translation first: it is the smaller model, and the LLM fallback
    # below keeps its own copy loaded afterwards.
    test_translate()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        test_xlit()
    except Exception as exc:
        print(f"IndicXlit failed: {type(exc).__name__}: {exc}")
        test_xlit_fallback(samples)
