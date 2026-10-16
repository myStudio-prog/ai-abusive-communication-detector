
import streamlit as st
import joblib
import re
from pathlib import Path

st.set_page_config(
    page_title="AI Abusive Communication Detector",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "abusive_communication_detector.joblib"
    return joblib.load(path)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("AI-Powered Abusive Communication Detector")

st.write(
    "Analyze communication using a trained NLP model "
    "and identify potentially harmful language."
)

st.caption(
    "Experimental model trained on Jigsaw comments. "
    "Predictions are advisory, not enforcement decisions."
)

try:
    artifact = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Add "
        "abusive_communication_detector.joblib "
        "to the repository."
    )
    st.stop()

model = artifact["model"]
labels = artifact["labels"]
thresholds = artifact["thresholds"]

message = st.text_area(
    "Enter a message to analyze:",
    height=130,
    placeholder="Type a message here..."
)

if st.button("Analyze Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a message.")

    else:
        cleaned = clean_text(message)
        probabilities = model.predict_proba([cleaned])[0]

        scores = {
            label: float(prob)
            for label, prob in zip(labels, probabilities)
        }

        detected = [
            label
            for label in labels
            if scores[label] >= thresholds[label]
        ]

        st.subheader("Analysis Results")

        if detected:
            st.warning(
                "Detected categories: " + ", ".join(detected)
            )
        else:
            st.info(
                "No toxicity category exceeded its threshold."
            )

        st.subheader("Category Scores")

        for label, score in scores.items():
            st.write(
                f"{label.replace('_', ' ').title()}: "
                f"{score:.1%}"
            )
            st.progress(min(max(score, 0.0), 1.0))

        
        st.subheader("Moderation Recommendation")

        # Experimental threshold for possible threats
        threat_review_threshold = 0.70

        if "threat" in detected:
            st.error(
                "Threat detected. Priority human review recommended."
            )

        elif scores["threat"] >= threat_review_threshold:
            st.error(
                "Possible threat detected below classification "
                "threshold. Priority human review recommended."
            )

        elif "identity_hate" in detected:
            st.error(
                "Potential identity-based hate. "
                "Priority human review recommended."
            )

        elif detected:
            st.warning(
                "Potentially harmful communication. "
                "Human moderation review recommended."
            )

        else:
            st.info(
                "No category exceeded its threshold. "
                "This does not guarantee the message is safe."
            )

        st.caption(
            "Scores are uncalibrated model estimates. "
            "The system may miss harmful content or "
            "incorrectly flag harmless messages. "
            "Do not use for automatic enforcement."
        )
