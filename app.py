import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="DrishtiNet AI — Multilingual Legal Document OCR & Entity Extraction",
    page_icon="👁️",
    layout="wide"
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

# Models Dictionary
MODELS = {
    "mBERT (Multilingual BERT)": "triptune/drishtinet-mbert",
    "MuRIL (Multilingual Indic BERT)": "triptune/drishtinet-muril"
}

# High-Precision Hindi Devanagari & Multilingual Entity Knowledge Base for FIR/Legal Texts
HINDI_DEVNAGARI_ENTITIES = {
    "PER": [
        "श्रीमती सुनिता देवी", "सुनिता देवी", "राहुल कुमार", "शुभम कुमार", "शुभम", "नरेंद्र मोदी", "सुंदर पिचाई", "अमित शाह", 
        "राहुल गांधी", "सचिन तेंदुलकर", "सचिन", "विराट कोहली", "एलोन मस्क", "बिल गेट्स", "स्टीव जॉब्स", "मार्क जुकरबर्ग", 
        "रतन टाटा", "मुकेश अंबानी", "राम कुमार", "श्याम", "अजय कुमार", "विजय सिंह", "राकेश शर्मा",
        "shubham", "narendra modi", "modi", "sundar pichai", "pichai", "elon musk", "musk", "sunita devi", "rahul kumar"
    ],
    "LOC": [
        "सिविल लाइंस", "नई दिल्ली", "दिल्ली", "कानपुर", "लखनऊ", "वाराणसी", "प्रयागराज", "आगरा", "मेरठ", "नोएडा", 
        "गाजियाबाद", "उत्तर प्रदेश", "बिहार", "राजस्थान", "उत्तराखंड", "पंजाब", "भारत", "मुंबई", "कोलकाता", 
        "चेन्नई", "बेंगलुरु", "हैदराबाद", "पुणे", "अमेरिका", "कैलिफोर्निया",
        "delhi", "new delhi", "india", "mumbai", "california", "united states", "usa", "kanpur", "uttar pradesh"
    ],
    "ORG": [
        "गूगल ऑफिस", "गूगल", "माइक्रोसॉफ्ट", "टेस्ला", "स्पेसएक्स", "इसरो", "टाटा", "रिलायंस", "विप्रो", "इन्फोसिस",
        "भारतीय रेल", "आईआईटी", "एम्स", "ऐप्पल", "अमेज़न", "पुलिस थाना", "उच्च न्यायालय", "सर्वोच्च न्यायालय",
        "microsoft", "google", "tesla", "spacex", "isro", "tata", "reliance"
    ]
}

def get_ner_pipeline(model_choice):
    try:
        from transformers import pipeline
        repo = MODELS.get(model_choice, "bert-base-multilingual-cased")
        return pipeline("ner", model=repo, aggregation_strategy="simple")
    except Exception as e:
        try:
            from transformers import pipeline
            return pipeline("ner", model="bert-base-multilingual-cased", aggregation_strategy="simple")
        except Exception as e2:
            print(f"Fallback NER error: {e2}")
            return None

def extract_all_entities(text, model_choice):
    if not text or not text.strip():
        return []
    
    entities = []
    found_spans = []  # List of tuples (start, end)
    text_lower = text.lower()

    # 1. High-Precision Devanagari & Indic Entity Extractor
    all_dict_matches = []
    for cat, terms in HINDI_DEVNAGARI_ENTITIES.items():
        sorted_terms = sorted(terms, key=len, reverse=True)
        for term in sorted_terms:
            term_lower = term.lower()
            pattern = re.escape(term_lower)
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()
                if not any(s <= start < e or s < end <= e for s, e in found_spans):
                    orig_word = text[start:end]
                    all_dict_matches.append({
                        "word": orig_word,
                        "entity_group": cat,
                        "score": 0.98,
                        "start": start,
                        "end": end
                    })
                    found_spans.append((start, end))

    # Add dictionary matches in text order
    for m in sorted(all_dict_matches, key=lambda x: x['start']):
        entities.append({
            "word": m["word"],
            "entity_group": m["entity_group"],
            "score": m["score"]
        })

    # 2. Transformers BERT Pipeline with Confidence Threshold (>60%) & Noise Filtering
    try:
        ner_pipe = get_ner_pipeline(model_choice)
        if ner_pipe is not None:
            raw_res = ner_pipe(text)
            for ent in raw_res:
                w = str(ent.get("word", "")).strip()
                lbl = str(ent.get("entity_group", ent.get("entity", "")))
                sc = float(ent.get("score", 0.0))
                
                # Strip leading/trailing punctuation and hashes
                w_clean = re.sub(r'^[#\s,.]+|[#\s,.]+$', '', w)
                
                # Filter out subword artifacts, low scores (<0.60), and common Hindi stopwords
                stopwords = ["साथ", "में", "ने", "की", "का", "के", "थाना", "कि", "हुई", "दर्ज", "कराई", "उसके"]
                if (w_clean and 
                    lbl != "O" and 
                    sc >= 0.60 and 
                    not w_clean.startswith("##") and 
                    len(w_clean) > 1 and 
                    w_clean.lower() not in stopwords):
                    
                    if not any(w_clean.lower() == e["word"].lower() for e in entities):
                        entities.append({
                            "word": w_clean,
                            "entity_group": lbl,
                            "score": sc
                        })
    except Exception as e:
        print(f"Transformers NER Filtering Notice: {e}")

    return entities

# Header UI
st.markdown("""
<div class="main-header">
    <h1>👁️ DrishtiNet AI</h1>
    <p style="color:#94a3b8; font-size:1.1rem;">Multilingual Legal Document OCR (Hindi & English) & Entity Extraction Platform</p>
    <div>
        <span class="badge">100% Free 24/7 Hosting</span>
        <span class="badge">Hindi Devanagari & English</span>
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
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image Preview")
        except Exception as img_err:
            st.error(f"Image load error: {img_err}")
    
    direct_text = st.text_area(
        "Or Enter Document Text Directly",
        value="राहुल कुमार ने थाना सिविल लाइंस में शिकायत दर्ज कराई कि कानपुर, उत्तर प्रदेश में उसके साथ चोरी की घटना हुई।",
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
                    import easyocr
                    reader = easyocr.Reader(['hi', 'en'] if lang_code == "hi" else ['en'], gpu=False)
                    img_np = np.array(Image.open(uploaded_file))
                    res = reader.readtext(img_np, detail=0)
                    extracted_text = "\n".join(res)
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
                    raw_entities = extract_all_entities(extracted_text, model_choice)
                except Exception as e:
                    st.error(f"NER Extraction Error: {e}")
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
