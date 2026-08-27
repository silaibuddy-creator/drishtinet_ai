import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="DrishtiNet AI — Multilingual Legal Document OCR & Entity Extraction",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #1e1b4b 0%, #311b92 50%, #0f172a 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #a5b4fc, #c084fc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.4);
        margin: 0.25rem;
    }
    .entity-tag-per {
        background-color: rgba(59, 130, 246, 0.25);
        border: 1px solid #3b82f6;
        color: #60a5fa;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
    }
    .entity-tag-org {
        background-color: rgba(16, 185, 129, 0.25);
        border: 1px solid #10b981;
        color: #34d399;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
    }
    .entity-tag-loc {
        background-color: rgba(245, 158, 11, 0.25);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Fine-Tuned Models Dictionary
MODELS = {
    "mBERT (Multilingual BERT)": "triptune/drishtinet-mbert",
    "MuRIL (Multilingual Indic BERT)": "triptune/drishtinet-muril"
}

# 1. Load EasyOCR Engine
@st.cache_resource(show_spinner=False)
def load_ocr_engine(lang="hi"):
    try:
        import easyocr
        lang_list = ['hi', 'en'] if lang == "hi" else ['en']
        return easyocr.Reader(lang_list, gpu=False)
    except Exception as e:
        print(f"EasyOCR load error: {e}")
        return None

# 2. Load Fine-Tuned Transformers NER Pipelines
@st.cache_resource(show_spinner=False)
def load_ner_pipeline(model_choice):
    model_repo = MODELS[model_choice]
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        model = AutoModelForTokenClassification.from_pretrained(model_repo)
        return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    except Exception as e:
        from transformers import pipeline
        return pipeline("ner", model="bert-base-multilingual-cased", aggregation_strategy="simple")

# Header UI
st.markdown("""
<div class="main-header">
    <h1>👁️ DrishtiNet AI</h1>
    <p style="color:#94a3b8; font-size:1.1rem;">Multilingual Legal Document OCR (Hindi & English) & Entity Extraction Platform</p>
    <div>
        <span class="badge">100% Free 24/7 Hosting</span>
        <span class="badge">OCR (Hindi 'hi' / English 'en')</span>
        <span class="badge">Fine-Tuned mBERT & MuRIL</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")
ocr_lang = st.sidebar.selectbox("OCR Language Model", ["Hindi (Devanagari)", "English"], index=0)
lang_code = "hi" if "Hindi" in ocr_lang else "en"

model_choice = st.sidebar.selectbox("NER Model Architecture", list(MODELS.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🏷️ Entity Types Legend
- **PER** (Person): Names of individuals
- **LOC** (Location): Cities, States, Countries
- **ORG** (Organization): Companies, Institutions
""")

# Main Workspace Columns
col_input, col_output = st.columns(2)

with col_input:
    st.header("📄 Document & Image Input")
    
    uploaded_file = st.file_uploader("Upload FIR or Document Image", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Preview")
    
    direct_text = st.text_area(
        "Or Enter Document Text Directly",
        value="गवाह श्रीमती सुनिता देवी निवासी नई दिल्ली ने बताया कि घटना के समय वह गूगल ऑफिस के पास उपस्थित थीं।",
        height=130
    )
    
    process_btn = st.button("🚀 Process OCR & Extract Entities", type="primary")

with col_output:
    st.header("📊 Extraction Results")
    
    if process_btn:
        extracted_text = ""
        
        # 1. Run OCR
        if uploaded_file is not None:
            with st.spinner("Running OCR text recognition..."):
                try:
                    ocr = load_ocr_engine(lang_code)
                    if ocr is not None:
                        img_np = np.array(Image.open(uploaded_file))
                        res = ocr.readtext(img_np, detail=0)
                        extracted_text = "\n".join(res)
                    else:
                        extracted_text = "OCR engine initializing..."
                except Exception as e:
                    st.error(f"OCR Processing Error: {e}")

        # Combine with direct text
        if direct_text and direct_text.strip():
            if extracted_text:
                extracted_text += "\n" + direct_text.strip()
            else:
                extracted_text = direct_text.strip()

        if not extracted_text:
            st.warning("No text extracted. Please upload a clear image or enter text above.")
        else:
            st.subheader("📝 Extracted Document Text (OCR)")
            st.text_area("OCR Output", value=extracted_text, height=150, disabled=True)

            # 2. Run NER Extraction
            with st.spinner("Extracting Named Entities (PER, LOC, ORG)..."):
                try:
                    ner_pipe = load_ner_pipeline(model_choice)
                    raw_entities = ner_pipe(extracted_text) if ner_pipe else []
                except Exception as e:
                    st.error(f"NER Error: {e}")
                    raw_entities = []

            st.subheader("🏷️ Highlighted Entities")
            
            highlight_html = extracted_text
            table_rows = []
            
            for ent in raw_entities:
                word = ent.get("word", "")
                group = ent.get("entity_group", ent.get("entity", ""))
                score = round(float(ent.get("score", 0)) * 100, 2)
                
                if word and group != "O":
                    css_class = f"entity-tag-{group.lower()}" if group.lower() in ["per", "org", "loc"] else "entity-tag-per"
                    tag_badge = f'<span class="{css_class}">{word} <sub>{group}</sub></span>'
                    highlight_html = highlight_html.replace(word, tag_badge)
                    
                    table_rows.append({
                        "Entity": word,
                        "Category": group,
                        "Confidence Score": f"{score}%"
                    })

            st.markdown(f'<div style="background:rgba(15, 23, 42, 0.6); padding:1.25rem; border-radius:12px; line-height:2.2;">{highlight_html}</div>', unsafe_allow_html=True)

            st.subheader("📋 Detected Entities Detail Table")
            if table_rows:
                df = pd.DataFrame(table_rows)
                st.dataframe(df)
            else:
                st.info("No PER, LOC, or ORG entities detected in the text.")
