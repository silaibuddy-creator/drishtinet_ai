import os
import streamlit as st
import numpy as np
from PIL import Image

# Disable MKLDNN
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR
from transformers import pipeline

st.set_page_config(page_title="DrishtiNet AI — Multilingual Document OCR & NER", page_icon="👁️", layout="wide")

st.title("👁️ DrishtiNet AI")
st.subheader("Multilingual Legal Document OCR (Hindi & English) & Entity Extraction")

@st.cache_resource
def load_ocr_models():
    ocr_en = PaddleOCR(lang="en", use_angle_cls=True, enable_mkldnn=False)
    ocr_hi = PaddleOCR(lang="hi", use_angle_cls=True, enable_mkldnn=False)
    return ocr_en, ocr_hi

@st.cache_resource
def load_ner_model():
    try:
        return pipeline("ner", model="triptune/drishtinet-mbert", aggregation_strategy="simple")
    except Exception as e:
        st.warning(f"Defaulting to mBERT base model due to: {e}")
        return pipeline("ner", model="bert-base-multilingual-cased", aggregation_strategy="simple")

col1, col2 = st.columns(2)

with col1:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload FIR or Legal Document Image", type=["png", "jpg", "jpeg", "webp"])
    language = st.radio("Document Language", ["Hindi", "English"], index=0)
    process_btn = st.button("🚀 Extract Text & Entities", type="primary")

with col2:
    st.header("📊 Results")
    if process_btn and uploaded_file is not None:
        with st.spinner("Processing OCR & Extracting Entities..."):
            image = Image.open(uploaded_file)
            img_np = np.array(image)
            
            ocr_en, ocr_hi = load_ocr_models()
            ocr = ocr_hi if language == "Hindi" else ocr_en
            
            try:
                ocr_result = ocr.ocr(img_np, cls=True)
                lines = []
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        lines.append(line[1][0])
                full_text = "\n".join(lines)
            except Exception as e:
                full_text = f"OCR Error: {e}"

            st.text_area("Extracted Text (OCR)", value=full_text, height=200)

            ner_pipe = load_ner_model()
            entities = []
            if ner_pipe and full_text.strip():
                ner_res = ner_pipe(full_text)
                for ent in ner_res:
                    entities.append({
                        "Entity": ent.get("word", ""),
                        "Category": ent.get("entity_group", ent.get("entity", "")),
                        "Confidence Score": round(float(ent.get("score", 0)), 4)
                    })

            st.subheader("Extracted Entities (NER)")
            if entities:
                st.dataframe(entities)
            else:
                st.info("No entities detected.")
