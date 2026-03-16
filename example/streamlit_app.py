"""
Knee Osteoarthritis Severity Grader — Streamlit Demo
Uses a trained YOLO classification model to predict KL grade from knee X-ray images.

Run:
    streamlit run example/streamlit_app.py

    # With a specific model
    KNEE_MODEL_PATH=runs/classify/knee_cls/weights/best.pt streamlit run example/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Allow importing from project root ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Constants ──────────────────────────────────────────────────────────────────
KL_GRADES = {
    0: "Normal",
    1: "Doubtful",
    2: "Minimal",
    3: "Moderate",
    4: "Severe",
}

KL_DESCRIPTIONS = {
    0: "No features of OA. Joint space appears normal.",
    1: "Doubtful narrowing of joint space with possible osteophytic lipping.",
    2: "Definite osteophytes and possible narrowing of the joint space.",
    3: "Multiple osteophytes, definite narrowing, some sclerosis, possible bony deformity.",
    4: "Large osteophytes, marked narrowing, severe sclerosis, and definite bony deformity.",
}

KL_COLORS = {
    0: "#2ecc71",   # green
    1: "#f1c40f",   # yellow
    2: "#e67e22",   # orange
    3: "#e74c3c",   # red
    4: "#8e44ad",   # purple
}

DEFAULT_MODEL = os.environ.get(
    "KNEE_MODEL_PATH",
    str(ROOT / "runs" / "classify" / "knee_cls" / "weights" / "best.pt"),
)


# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model, image: Image.Image, imgsz: int = 224):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    results = model.predict(source=tmp_path, imgsz=imgsz, verbose=False)
    os.unlink(tmp_path)

    result = results[0]
    probs = result.probs

    top5_idx = probs.top5
    top5_conf = probs.top5conf.tolist()

    predictions = [
        {
            "class_id": int(idx),
            "label": KL_GRADES[int(idx)],
            "grade": f"KL-{int(idx)}",
            "confidence": float(conf),
            "description": KL_DESCRIPTIONS[int(idx)],
            "color": KL_COLORS[int(idx)],
        }
        for idx, conf in zip(top5_idx, top5_conf)
    ]
    return predictions


# ── UI ─────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Knee OA Severity Grader",
        page_icon="🦴",
        layout="wide",
    )

    # Header
    st.title("🦴 Knee Osteoarthritis Severity Grader")
    st.markdown(
        "Upload a knee X-ray image to predict the **Kellgren-Lawrence (KL) Grade** "
        "using a YOLO-based deep learning classifier."
    )
    st.divider()

    # Sidebar — model configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_path = st.text_input("Model weights path", value=DEFAULT_MODEL)
        imgsz = st.select_slider("Image size", options=[128, 160, 192, 224, 256, 320, 384, 448], value=224)
        show_all = st.toggle("Show all class probabilities", value=True)

        st.divider()
        st.markdown("**KL Grade Reference**")
        for grade_id, grade_name in KL_GRADES.items():
            color = KL_COLORS[grade_id]
            st.markdown(
                f'<span style="color:{color}; font-weight:bold;">KL-{grade_id} {grade_name}</span>',
                unsafe_allow_html=True,
            )
            st.caption(KL_DESCRIPTIONS[grade_id])

    # Load model
    if not Path(model_path).exists():
        st.warning(
            f"Model weights not found at `{model_path}`.  \n"
            "Please train a model first:  \n"
            "```\npython scripts/train.py\n```"
        )
        st.stop()

    try:
        model = load_model(model_path)
        st.sidebar.success(f"Model loaded successfully")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Upload
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose a knee X-ray image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            label_visibility="collapsed",
        )

        use_sample = st.button("🔍 Use sample image from dataset")

        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        elif use_sample:
            sample_dirs = [
                Path("/home/jayden/data/knee-osteoarthritis-dataset-with-severity/test"),
            ]
            for d in sample_dirs:
                if d.exists():
                    imgs = list(d.rglob("*.png"))[:1]
                    if imgs:
                        image = Image.open(imgs[0]).convert("RGB")
                        st.info(f"Sample: `{imgs[0].name}` (class {imgs[0].parent.name})")
                        break

        if image is not None:
            st.image(image, caption="Input X-ray", use_container_width=True)

    # Prediction
    with col_result:
        st.subheader("Prediction")

        if image is None:
            st.info("Upload an image to see the prediction.")
            return

        with st.spinner("Running inference..."):
            predictions = predict(model, image, imgsz=imgsz)

        best = predictions[0]
        color = best["color"]

        # Main result card
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {color}22, {color}44);
                border: 2px solid {color};
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
            ">
                <h1 style="color:{color}; margin:0; font-size: 3rem;">{best['grade']}</h1>
                <h2 style="color:{color}; margin:4px 0;">{best['label']}</h2>
                <p style="color:#aaa; margin:0;">{best['description']}</p>
                <h3 style="margin-top:12px;">Confidence: {best['confidence']:.1%}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability bars
        if show_all:
            st.markdown("**All Class Probabilities**")
            # Show all 5 classes sorted by class ID
            all_probs = {p["class_id"]: p["confidence"] for p in predictions}
            for cls_id in sorted(KL_GRADES.keys()):
                conf = all_probs.get(cls_id, 0.0)
                bar_color = KL_COLORS[cls_id]
                label = f"KL-{cls_id} {KL_GRADES[cls_id]}"
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; margin:4px 0;">
                        <span style="width:160px; color:{bar_color}; font-weight:bold;">{label}</span>
                        <div style="flex:1; background:#333; border-radius:6px; height:18px; margin:0 10px;">
                            <div style="width:{conf*100:.1f}%; background:{bar_color}; height:100%; border-radius:6px;"></div>
                        </div>
                        <span style="width:50px; text-align:right;">{conf:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Disclaimer
        st.divider()
        st.caption(
            "⚠️ **Disclaimer:** This tool is for research and educational purposes only. "
            "It is not intended for clinical diagnosis. Always consult a qualified medical professional."
        )


if __name__ == "__main__":
    main()
