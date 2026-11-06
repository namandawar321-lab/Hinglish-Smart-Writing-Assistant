"""Hinglish writing pipeline.

Roman Hinglish -> transliteration -> LLM polish -> English translation.

transliterate() has two modes:
- baseline: IndicXlit over the whole sentence, English words included
- improved: SMS normalization, then IndicXlit only on Hindi spans
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
VENDOR = ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))

import torch

from src.lexicon import DEVANAGARI_RE, apply_transliteration

# Param-1-2.9B is the handout model. It is optional via POLISH_MODEL because
# it is too slow and heavy on an 8 GB GPU. Qwen 1.5B is the default.
DEFAULT_POLISH = "Qwen/Qwen2.5-1.5B-Instruct"
OPTIONAL_POLISH = "bharatgenai/Param-1-2.9B-Instruct"
# Official Indic-En distilled checkpoint, then the public RoPE release of the
# same size and direction. The official repo is gated on Hugging Face.
TRANSLATION_CANDIDATES = [
    "ai4bharat/indictrans2-indic-en-dist-200M",
    "prajdabre/rotary-indictrans2-indic-en-dist-200M",
]

POLISH_INSTRUCTION = (
    "You are a careful Hindi editor. The user gives one sentence that is "
    "already written in Devanagari, sometimes mixed with English words in "
    "Latin script. Rewrite it so the grammar and punctuation sound natural. "
    "Do not change the meaning. Do not translate the sentence into English. "
    "Keep English words such as meeting, report, email, and deadline in "
    "Latin script. Output only the rewritten sentence."
)
TRANSLIT_KEEP_INSTRUCTION = (
    "Convert this Hinglish (Roman script) to Devanagari Hindi. Keep English "
    "words like meeting, report, email, and deadline in English (Latin) "
    "script. Output only the converted text."
)
TRANSLIT_ALL_INSTRUCTION = (
    "Convert this entire Hinglish sentence into Devanagari, including "
    "English words. Output only the converted text."
)


def _patch_indictrans_tokenizer() -> None:
    """Let older IndicTrans tokenizers construct under Transformers 5.

    Their __init__ assigns unk_token before PreTrainedTokenizerBase creates
    _special_tokens_map. Transformers 5 routes that assignment through the map.
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if getattr(PreTrainedTokenizerBase, "_hinglish_tokenizer_patch", False):
        return
    original = PreTrainedTokenizerBase.__setattr__

    def patched(self, key, value):
        if isinstance(key, str):
            bare = key.removesuffix("_ids").removesuffix("_id") if key.endswith(("_id", "_ids")) else key
            names = getattr(type(self), "SPECIAL_TOKENS_ATTRIBUTES", ())
            if bare in names or bare == "extra_special_tokens":
                if "_special_tokens_map" not in self.__dict__:
                    self.__dict__["_special_tokens_map"] = {}
        return original(self, key, value)

    PreTrainedTokenizerBase.__setattr__ = patched
    PreTrainedTokenizerBase._hinglish_tokenizer_patch = True


def _coerce_xlit(result) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        value = result.get("hi", next(iter(result.values())))
        return _coerce_xlit(value)
    if isinstance(result, (list, tuple)):
        return _coerce_xlit(result[0]) if result else ""
    return str(result).strip()


def _clean_generation(text: str, fallback: str) -> str:
    text = (text or "").strip().strip("`").strip()
    if not text:
        return fallback
    lines = []
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(
            (
                "polished",
                "output",
                "rewritten",
                "devanagari",
                "hindi:",
                "note:",
                "sure,",
                "sure.",
                "here is",
                "here's",
                "the rewritten",
                "converted",
            )
        ):
            continue
        lines.append(line)
    candidate = lines[0] if lines else text.splitlines()[0].strip()
    if len(candidate) > 400:
        candidate = candidate[:400].rsplit(" ", 1)[0]
    return candidate or fallback


class HinglishAssistant:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.xlit = None
        self.xlit_backend = None
        self.polish_model = None
        self.polish_tok = None
        self.polish_name = None
        self.polish_quant = None
        self.trans_model = None
        self.trans_tok = None
        self.trans_ip = None
        self.trans_device = None
        self.translation_name = None
        self.notes: list[str] = []
        self._loaded = False
        self._loading = False

    def load(self) -> None:
        if self._loaded or self._loading:
            return
        self._loading = True
        try:
            CACHE.mkdir(parents=True, exist_ok=True)
            self._load_xlit()
            self._load_polish()
            self._load_translation()
            self._loaded = True
            self.describe()
        finally:
            self._loading = False

    def describe(self) -> None:
        vram = 0.0
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.is_available()} ({vram:.1f} GB)")
        print(f"Transliteration: {self.xlit_backend}")
        print(f"Polish: {self.polish_name} [{self.polish_quant}]")
        print(f"Translation: {self.translation_name} on {self.trans_device}")
        for note in self.notes:
            print(f"Note: {note}")

    def info(self) -> dict:
        return {
            "gpu": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "transliteration": self.xlit_backend,
            "polish_model": self.polish_name,
            "polish_quantization": self.polish_quant,
            "translation_model": self.translation_name,
            "translation_device": self.trans_device,
            "notes": list(self.notes),
        }

    def _load_xlit(self) -> None:
        try:
            from ai4bharat.transliteration import XlitEngine
        except Exception as exc:
            self.xlit_backend = "llm"
            self.notes.append(
                "IndicXlit (ai4bharat-transliteration) could not be imported "
                f"({type(exc).__name__}: {exc}). Transliteration uses the polish "
                "LLM, which is the Step 3 fallback."
            )
            return
        last_error = None
        for kwargs in (
            {"beam_width": 10, "src_script_type": "en", "rescore": False},
            {"beam_width": 10, "src_script_type": "en"},
            {"beam_width": 10, "src_script_type": "roman", "rescore": False},
            {"beam_width": 10},
        ):
            try:
                self.xlit = XlitEngine("hi", **kwargs)
                self.xlit_backend = "indicxlit"
                self.notes.append(f"IndicXlit loaded with {kwargs}.")
                return
            except TypeError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
                break
        self.xlit = None
        self.xlit_backend = "llm"
        self.notes.append(
            "IndicXlit failed to initialize "
            f"({type(last_error).__name__}: {last_error}). Using the LLM fallback."
        )

    def _xlit_phrase(self, phrase: str) -> str:
        if self.xlit is None:
            raise RuntimeError("IndicXlit is not loaded")
        try:
            return _coerce_xlit(self.xlit.translit_sentence(phrase))
        except Exception:
            parts = []
            for tok in phrase.split():
                try:
                    parts.append(_coerce_xlit(self.xlit.translit_word(tok)))
                except Exception:
                    parts.append(_coerce_xlit(self.xlit.translit_sentence(tok)))
            return " ".join(parts)

    def _load_polish(self) -> None:
        requested = os.environ.get("POLISH_MODEL", DEFAULT_POLISH).strip() or DEFAULT_POLISH
        quant_flag = os.environ.get("POLISH_4BIT", "auto").lower()
        vram = 0.0
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # 4-bit is for the optional 2.9B model. Qwen 1.5B fits this GPU in FP16.
        heavy = requested == OPTIONAL_POLISH
        if quant_flag == "auto":
            use_4bit = heavy and vram > 0 and vram < 12
        else:
            use_4bit = quant_flag in {"1", "true", "yes"}
        if requested == DEFAULT_POLISH:
            self.notes.append(
                f"Polish model is {DEFAULT_POLISH}. {OPTIONAL_POLISH} is optional "
                "via POLISH_MODEL; it was too slow and heavy on an 8 GB GPU."
            )
        elif heavy and vram and vram < 12:
            self.notes.append(
                f"GPU has {vram:.1f} GB. Loading optional {OPTIONAL_POLISH} in 4-bit "
                "when bitsandbytes is available. Unset POLISH_MODEL to use "
                f"{DEFAULT_POLISH}."
            )
        try:
            self._load_causal(requested, use_4bit=use_4bit and self.device == "cuda")
        except Exception as exc:
            if requested == DEFAULT_POLISH:
                raise
            self.notes.append(
                f"Could not load {requested} ({type(exc).__name__}: {exc}). "
                f"Falling back to {DEFAULT_POLISH} in FP16."
            )
            self._release_polish()
            self._load_causal(DEFAULT_POLISH, use_4bit=False)
            return
        if requested != DEFAULT_POLISH and not self._polish_is_usable():
            self.notes.append(
                f"{self.polish_name} loaded, but a polish probe did not follow "
                f"the edit instruction. Switching to {DEFAULT_POLISH}."
            )
            self._release_polish()
            self._load_causal(DEFAULT_POLISH, use_4bit=False)

    def _release_polish(self) -> None:
        self.polish_model = None
        self.polish_tok = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_causal(self, name: str, use_4bit: bool) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        quant_name = "fp16" if self.device == "cuda" else "fp32"
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
                quant_name = "4-bit nf4"
            except Exception as exc:
                self.notes.append(
                    f"4-bit loading is unavailable ({type(exc).__name__}: {exc}). "
                    "Loading the polish model in FP16 instead."
                )
                use_4bit = False
        if not use_4bit:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            kwargs["torch_dtype"] = dtype
            kwargs["device_map"] = "auto" if self.device == "cuda" else None
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if kwargs.get("device_map") is None:
            model = model.to(self.device)
        model.eval()
        self.polish_tok = tok
        self.polish_model = model
        self.polish_name = name
        self.polish_quant = quant_name

    def _polish_is_usable(self) -> bool:
        sample_in = "कल meeting है please send the report"
        try:
            sample_out = self._complete(POLISH_INSTRUCTION, sample_in, fallback="")
        except Exception as exc:
            self.notes.append(f"Polish probe failed: {type(exc).__name__}: {exc}")
            return False
        if not sample_out or len(sample_out) > 240:
            return False
        if not DEVANAGARI_RE.search(sample_out):
            return False
        low = sample_out.lower()
        if any(bad in low for bad in ("as an ai", "i cannot", "sure,", "here is the", "rewritten sentence")):
            return False
        print(f"Polish probe: {sample_out}")
        return True

    def _prompt_ids(self, instruction: str, user_text: str):
        tok = self.polish_tok
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_text},
        ]
        try:
            return tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            merged = [{"role": "user", "content": instruction + "\n\n" + user_text}]
            try:
                return tok.apply_chat_template(
                    merged,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            except Exception:
                return tok(instruction + "\n\n" + user_text, return_tensors="pt")

    def _complete(self, instruction: str, user_text: str, fallback: str) -> str:
        if not user_text or not user_text.strip():
            return ""
        if self.polish_model is None:
            self.load()
        prompt = self._prompt_ids(instruction, user_text.strip())
        if torch.is_tensor(prompt):
            input_ids = prompt.to(self.polish_model.device)
        else:
            input_ids = prompt["input_ids"].to(self.polish_model.device)
        attention = torch.ones_like(input_ids)
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=self.polish_tok.pad_token_id,
            eos_token_id=self.polish_tok.eos_token_id,
        )
        try:
            with torch.inference_mode():
                output = self.polish_model.generate(**gen_kwargs, use_cache=True)
        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.inference_mode():
                output = self.polish_model.generate(**gen_kwargs, use_cache=False)
        new_tokens = output[0][input_ids.shape[-1] :]
        decoded = self.polish_tok.decode(new_tokens, skip_special_tokens=True)
        return _clean_generation(decoded, fallback)

    def _load_translation(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            from IndicTransToolkit import IndicProcessor
        except ImportError:
            try:
                from IndicTransToolkit.processor import IndicProcessor
            except ImportError:
                from IndicTransToolkit.IndicTransToolkit import IndicProcessor
        processor_file = getattr(sys.modules.get("IndicTransToolkit"), "__file__", "")
        if "vendor" in str(processor_file).replace("\\", "/"):
            self.notes.append(
                "IndicTransToolkit's Cython build needs Microsoft C++ Build Tools, "
                "which are not installed. Using the pure-Python port in vendor/."
            )

        _patch_indictrans_tokenizer()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = None
        tok = None
        errors = []
        for name in TRANSLATION_CANDIDATES:
            try:
                tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    name,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                self.translation_name = name
                if name != TRANSLATION_CANDIDATES[0]:
                    self.notes.append(
                        "ai4bharat/indictrans2-indic-en-dist-200M is a gated Hugging Face "
                        "repo and this machine is not logged in. Using the public RoPE "
                        f"Indic-En 200M checkpoint {name}, which is the same model family "
                        "and direction."
                    )
                break
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                tok = None
                model = None
        if model is None:
            raise RuntimeError("No translation checkpoint could be loaded. " + " | ".join(errors))
        try:
            model = model.to(self.device)
            self.trans_device = self.device
        except RuntimeError as exc:
            self.notes.append(
                f"IndicTrans2 did not fit on GPU ({exc}). Running translation on CPU."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.translation_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.trans_device = "cpu"
        model.eval()
        self.trans_model = model
        self.trans_tok = tok
        self.trans_ip = IndicProcessor(inference=True)

    def transliterate(self, text: str, preserve_english: bool = True) -> str:
        """Roman Hinglish to Devanagari.

        preserve_english=True is the improved system. False is the
        transliterate-everything baseline.
        """
        self.load()
        text = "" if text is None else str(text)
        if not text.strip():
            return ""
        if self.xlit is not None:
            return apply_transliteration(text, self._xlit_phrase, preserve_english)
        instruction = TRANSLIT_KEEP_INSTRUCTION if preserve_english else TRANSLIT_ALL_INSTRUCTION
        # Still normalize SMS spellings on the improved path before the LLM sees them.
        from src.lexicon import normalize_hinglish

        source = normalize_hinglish(text) if preserve_english else text
        return self._complete(instruction, source, fallback=source)

    def polish(self, text: str) -> str:
        """Light fluency edit. English words are asked to stay in Latin."""
        self.load()
        text = "" if text is None else str(text).strip()
        if not text:
            return ""
        return self._complete(POLISH_INSTRUCTION, text, fallback=text)

    def translate(self, text: str) -> str:
        """Hindi (possibly code-mixed) to English with IndicTrans2."""
        self.load()
        text = "" if text is None else str(text).strip()
        if not text:
            return ""
        batch = self.trans_ip.preprocess_batch(
            [text], src_lang="hin_Deva", tgt_lang="eng_Latn"
        )
        inputs = self.trans_tok(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.trans_device)
        with torch.inference_mode():
            generated = self.trans_model.generate(
                **inputs,
                num_beams=5,
                max_length=256,
                # Rotary IndicTrans2 expects legacy tuple caches. Transformers
                # 4.57 injects an empty DynamicCache, which this model cannot read.
                use_cache=False,
            )
        decoded = self.trans_tok.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        outputs = self.trans_ip.postprocess_batch(decoded, lang="eng_Latn")
        return (outputs[0] if outputs else "").strip()

    def run(self, text: str) -> dict:
        translit = self.transliterate(text, preserve_english=True)
        polished = self.polish(translit)
        english = self.translate(polished)
        return {
            "transliteration": translit,
            "polished": polished,
            "english": english,
        }


_ASSISTANT: HinglishAssistant | None = None


def get_assistant() -> HinglishAssistant:
    global _ASSISTANT
    if _ASSISTANT is None:
        _ASSISTANT = HinglishAssistant()
    return _ASSISTANT


def transliterate(text: str, preserve_english: bool = True) -> str:
    return get_assistant().transliterate(text, preserve_english=preserve_english)


def polish(text: str) -> str:
    return get_assistant().polish(text)


def translate(text: str) -> str:
    return get_assistant().translate(text)


def run(text: str) -> dict:
    return get_assistant().run(text)
