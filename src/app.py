"""Streamlit demo: paraphrase detection, bi-encoder vs cross-encoder.

Run from the project root:  streamlit run src/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so "src." imports work under `streamlit run`

from src.model import CrossEncoderParaphraseModel, ParaphraseModel  # noqa: E402

RESULTS_DIR = ROOT / "results"

# Thresholds are the values printed by each model's (small-sample) training
# run. If you replace a checkpoint with its full-data Kaggle version, set the
# threshold that run printed (e.g. 0.60 for the full-data combined bi-encoder).
MODELS = {
    "Bi-encoder (QQP-only)": {
        "dir": "qqp", "type": "bi", "threshold": 0.65,
        "score_note": "cosine similarity of the two embeddings, range [-1, 1]",
    },
    "Bi-encoder (QQP+PAWS combined)": {
        "dir": "qqp-paws", "type": "bi", "threshold": 0.66,
        "score_note": "cosine similarity of the two embeddings, range [-1, 1]",
    },
    "Cross-encoder (QQP+PAWS combined)": {
        "dir": "qqp-paws-cross", "type": "cross", "threshold": -0.43,
        "score_note": "raw logit from the jointly-encoded pair, unbounded",
    },
}

# Example pairs by index into results/paws_errors.json (captured PAWS errors).
EXAMPLES = {
    "Footballer swap (true: not paraphrase)": 7,
    "Irish counties swap (true: not paraphrase)": 6,
    "Darts winner swap (true: not paraphrase)": 13,
    "Harmless reorder (true: paraphrase)": 3,
}


@st.cache_resource
def load_model(name: str):
    cfg = MODELS[name]
    cls = CrossEncoderParaphraseModel if cfg["type"] == "cross" else ParaphraseModel
    return cls.load(str(RESULTS_DIR / cfg["dir"]))


@st.cache_data
def load_examples():
    with open(RESULTS_DIR / "paws_errors.json", encoding="utf-8") as f:
        records = json.load(f)
    return {label: records[i] for label, i in EXAMPLES.items()}


st.title("Paraphrase detection: bi-encoder vs cross-encoder")

choice = st.selectbox("Model", list(MODELS))
cfg = MODELS[choice]
ckpt = RESULTS_DIR / cfg["dir"]
if not (ckpt / "model.safetensors").exists():
    st.error(
        f"Checkpoint missing: `{ckpt}`. This model was only trained on Kaggle. "
        "In the Kaggle session, download `results/" + cfg["dir"] + "` "
        "(zip it in /kaggle/working/paraphrase-detection/results and use the "
        "notebook Output panel), then unzip it to that path locally."
    )
    st.stop()

st.caption(
    f"**What the score means for this model**: {cfg['score_note']}. "
    f"Decision threshold {cfg['threshold']} (tuned on this model's own "
    "validation data). Bi-encoder scores are NOT comparable to cross-encoder "
    "scores — different scales, different meaning."
)

examples = load_examples()
st.write("Load a real captured PAWS case:")
cols = st.columns(len(EXAMPLES))
for col, (label, rec) in zip(cols, examples.items()):
    if col.button(label):
        st.session_state.s1 = rec["sentence1"]
        st.session_state.s2 = rec["sentence2"]

s1 = st.text_area("Sentence 1", key="s1", height=80)
s2 = st.text_area("Sentence 2", key="s2", height=80)

if s1.strip() and s2.strip():
    model = load_model(choice)
    score = float(model.similarity([(s1, s2)])[0])
    is_para = score >= cfg["threshold"]

    st.metric(
        "Prediction",
        "PARAPHRASE" if is_para else "NOT a paraphrase",
        delta=f"score {score:.3f} vs threshold {cfg['threshold']}",
        delta_color="off",
    )

    # If the text matches a loaded example exactly, compare with gold label.
    for label, rec in examples.items():
        if s1 == rec["sentence1"] and s2 == rec["sentence2"]:
            truth = rec["true_label"] == 1
            if is_para == truth:
                st.success(f"Matches the true label ({'paraphrase' if truth else 'not paraphrase'}) — {label}")
            else:
                st.error(f"WRONG — true label is {'paraphrase' if truth else 'not paraphrase'} ({label})")
            st.caption(
                f"When originally captured, the small-sample combined bi-encoder "
                f"scored this pair {rec['similarity']:.3f}."
            )
            break
