"""Compare baseline transliteration with the improved pipeline.

Baseline: IndicXlit on the raw sentence (English words included).
Improved transliteration: SMS normalization + English words left in Latin.
Full system: improved transliteration, then LLM polish, then IndicTrans2.

CER and word accuracy use jiwer. chrF and BLEU use sacrebleu.
Scores ignore punctuation and Latin case so a missing danda does not
dominate the transliteration comparison.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jiwer
import pandas as pd
import sacrebleu

from src.pipeline import get_assistant


def normalize_for_metric(text: str) -> str:
    text = unicodedata.normalize("NFC", "" if text is None else str(text))
    kept = []
    for ch in text:
        if ch.isspace():
            kept.append(" ")
        elif unicodedata.category(ch).startswith("P"):
            continue
        else:
            kept.append(ch.lower())
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def _safe_cer(refs: list[str], hyps: list[str]) -> float:
    try:
        return float(jiwer.cer(refs, hyps))
    except Exception:
        scores = []
        for ref, hyp in zip(refs, hyps):
            if not ref:
                continue
            scores.append(float(jiwer.cer(ref, hyp or "")))
        return sum(scores) / max(len(scores), 1)


def _safe_wer(refs: list[str], hyps: list[str]) -> float:
    try:
        return float(jiwer.wer(refs, hyps))
    except Exception:
        scores = []
        for ref, hyp in zip(refs, hyps):
            if not ref:
                continue
            scores.append(float(jiwer.wer(ref, hyp or " ")))
        return sum(scores) / max(len(scores), 1)


def score_devanagari(refs: list[str], hyps: list[str]) -> dict:
    refs_n = [normalize_for_metric(r) for r in refs]
    hyps_n = [normalize_for_metric(h) for h in hyps]
    wer = _safe_wer(refs_n, hyps_n)
    return {
        "cer": round(_safe_cer(refs_n, hyps_n), 4),
        "wer": round(wer, 4),
        "word_accuracy": round(max(0.0, 1.0 - wer), 4),
    }


def score_english(refs: list[str], hyps: list[str]) -> dict:
    refs_n = [normalize_for_metric(r) for r in refs]
    hyps_n = [normalize_for_metric(h) for h in hyps]
    bleu = sacrebleu.corpus_bleu(hyps_n, [refs_n])
    chrf = sacrebleu.corpus_chrf(hyps_n, [refs_n])
    return {
        "chrf": round(float(chrf.score), 2),
        "bleu": round(float(bleu.score), 2),
    }


def main() -> None:
    data_path = ROOT / "data" / "test_cases.csv"
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path, encoding="utf-8")
    assistant = get_assistant()
    assistant.load()

    rows = []
    base_hyps, imp_hyps, pol_hyps = [], [], []
    base_en, full_en = [], []
    refs_dev, refs_en = [], []

    for index, row in df.iterrows():
        hinglish = str(row["hinglish"])
        ref_dev = str(row["reference_devanagari"])
        ref_en = str(row["reference_english"])
        print(f"[{index + 1}/{len(df)}] {hinglish}", flush=True)
        baseline = assistant.transliterate(hinglish, preserve_english=False)
        improved = assistant.transliterate(hinglish, preserve_english=True)
        polished = assistant.polish(improved)
        english = assistant.translate(polished)
        baseline_english = assistant.translate(baseline)
        print(f"  base: {baseline}", flush=True)
        print(f"  imp:  {improved}", flush=True)
        print(f"  pol:  {polished}", flush=True)
        print(f"  en:   {english}", flush=True)
        base_hyps.append(baseline)
        imp_hyps.append(improved)
        pol_hyps.append(polished)
        full_en.append(english)
        base_en.append(baseline_english)
        refs_dev.append(ref_dev)
        refs_en.append(ref_en)
        rows.append(
            {
                "id": index + 1,
                "category": row.get("category", ""),
                "hinglish": hinglish,
                "reference_devanagari": ref_dev,
                "baseline": baseline,
                "improved_transliteration": improved,
                "polished": polished,
                "reference_english": ref_en,
                "baseline_english": baseline_english,
                "english": english,
                "improved_cer": round(
                    _safe_cer(
                        [normalize_for_metric(ref_dev)],
                        [normalize_for_metric(improved)],
                    ),
                    4,
                ),
            }
        )

    metrics = {
        "n": len(df),
        "normalization": "NFC, punctuation removed, Latin lowercased, whitespace collapsed",
        "baseline_transliteration": score_devanagari(refs_dev, base_hyps),
        "improved_transliteration": score_devanagari(refs_dev, imp_hyps),
        "polished": score_devanagari(refs_dev, pol_hyps),
        "baseline_translation": score_english(refs_en, base_en),
        "polished_translation": score_english(refs_en, full_en),
        "models": assistant.info(),
    }
    predictions = pd.DataFrame(rows)
    predictions.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
