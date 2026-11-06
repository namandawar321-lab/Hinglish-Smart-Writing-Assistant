# Hinglish Smart Writing Assistant

Roman Hinglish goes to Devanagari, then a short fluency edit, then English. A Gradio app shows all three stages. Evaluation compares a transliterate-everything baseline with the improved path on 40 hand-written sentences.

## Features

- Improved transliteration keeps listed English words in Latin script. The baseline converts the whole sentence, including those words.
- A small SMS map expands spellings such as `pls`, `kr`, and `bht` before the improved transliteration prompt.
- An instruct model polishes Devanagari without translating it into English.
- Indic-to-English translation runs after the polish step.
- `app.py` is a Gradio UI with one input and three output panels.
- `src/evaluate.py` scores Devanagari with CER and word accuracy (jiwer) and English with chrF and BLEU (sacrebleu).

## Requirements

- Windows, macOS, or Linux. This copy was run on Windows with Python 3.10 and an 8 GB NVIDIA GPU.
- Python 3.10.
- PyTorch with a CUDA build if you have a GPU. CPU works, but it is slow.
- The packages in `requirements.txt`.
- Enough disk space for the Qwen 1.5B weights and the IndicTrans2 checkpoint. They download on first use into `.cache/`, which is not part of this repository.
- For the optional Param-1 polish model, bitsandbytes. On a GPU under 12 GB it loads in 4-bit.

`transformers` must be the 4.57 line. Transformers 5 breaks this IndicTrans2 checkpoint. This environment uses `transformers==4.57.1`.

IndicXlit is not installed here. See [Deviations](#deviations).

## Installation

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install "transformers==4.57.1"
```

On macOS or Linux, activate with `source .venv/bin/activate` and install the PyTorch build that matches your machine from [pytorch.org](https://pytorch.org/get-started/locally/).

The first run downloads `Qwen/Qwen2.5-1.5B-Instruct` and `prajdabre/rotary-indictrans2-indic-en-dist-200M`.

## Usage

Start the app:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:7860. Type Roman Hinglish and press Convert. The panels are transliteration, polished Hindi, and English.

Smoke test:

```powershell
.\.venv\Scripts\python.exe src\smoke_test.py
```

Evaluate the 40 sentences in `data/test_cases.csv`. This writes `results/predictions.csv` and `results/metrics.json`, which are gitignored:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py
```

Optional polish model, instead of the Qwen default:

```powershell
$env:POLISH_MODEL = "bharatgenai/Param-1-2.9B-Instruct"
.\.venv\Scripts\python.exe app.py
```

## Project layout

| Path | Role |
| --- | --- |
| `app.py` | Gradio UI |
| `src/pipeline.py` | Transliteration, polish, and translation |
| `src/lexicon.py` | SMS expansions and the English-word list |
| `src/evaluate.py` | Baseline versus improved scores |
| `src/smoke_test.py` | One translation example and a transliteration check |
| `data/test_cases.csv` | 40 Hinglish sentences with Devanagari and English references |
| `vendor/IndicTransToolkit/` | Pure-Python IndicTrans preprocessor |
| `requirements.txt` | Python dependencies |
| `START_HERE.md` | Original assignment guide |

## Deviations

- **IndicXlit / fairseq.** `ai4bharat-transliteration` depends on fairseq, which does not build on this Windows machine. Transliteration uses the polish LLM instead of IndicXlit. On the improved path, `src/lexicon.py` still expands a few SMS spellings before the prompt. The English-word span rules run only when IndicXlit itself is loaded.
- **Polish model.** BharatGen Param-1-2.9B (`bharatgenai/Param-1-2.9B-Instruct`) is too slow and heavy for this 8 GB GPU, and its weights were still downloading. The default polish model is `Qwen/Qwen2.5-1.5B-Instruct`. Param-1 stays optional: set `POLISH_MODEL=bharatgenai/Param-1-2.9B-Instruct`. On a GPU under 12 GB that optional load uses 4-bit when bitsandbytes is available.
- **IndicTrans2 checkpoint.** `ai4bharat/indictrans2-indic-en-dist-200M` is a gated Hugging Face repo and this machine is not logged in. Translation uses `prajdabre/rotary-indictrans2-indic-en-dist-200M`, the public RoPE release of the same size and Indic-to-English direction. IndicTransToolkit needs Microsoft C++ Build Tools, so preprocessing uses the pure-Python port in `vendor/IndicTransToolkit`.
- **`use_cache=False`.** The RoPE checkpoint expects the old tuple cache. Transformers 4.57 injects an empty `DynamicCache`, so `past_key_values[0][0]` is `None` and generation crashes. Translation calls `generate` with `use_cache=False`.
