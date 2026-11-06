"""Gradio UI: one Hinglish box, three output panels."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from src.pipeline import get_assistant


EXAMPLES = [
    "kal meeting hai please send the report",
    "mujhe ye assignment samajh nahi aaya",
    "pls msg kr dena jab free ho",
    "meeting 15 november ko 4:30 pm hai",
    "mujhe to kal ka plan bata do",
]


def convert(text: str):
    text = (text or "").strip()
    if not text:
        return "", "", ""
    result = get_assistant().run(text)
    return result["transliteration"], result["polished"], result["english"]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Hinglish Smart Writing Assistant") as demo:
        gr.Markdown(
            """
# Hinglish Smart Writing Assistant
Roman Hinglish goes to Devanagari, then a fluency edit, then English.

English words such as **meeting** and **report** stay in Latin script.
The baseline used in evaluation transliterates those words as well.
            """
        )
        source = gr.Textbox(
            label="Hinglish input",
            lines=3,
            placeholder="kal meeting hai please send the report",
        )
        run_button = gr.Button("Convert", variant="primary")
        with gr.Row():
            translit = gr.Textbox(label="1. Transliteration", lines=4)
            polished = gr.Textbox(label="2. Polished Hindi", lines=4)
            english = gr.Textbox(label="3. English translation", lines=4)
        run_button.click(convert, inputs=source, outputs=[translit, polished, english])
        source.submit(convert, inputs=source, outputs=[translit, polished, english])
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=source)
    return demo


def main() -> None:
    get_assistant().load()
    build_demo().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
