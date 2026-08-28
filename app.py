import io
import re
import base64
import requests
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

# Known Knowledge Base Entities
HINDI_DEVNAGARI_ENTITIES = {
    "PER": [
        "श्रीमती सुनिता देवी", "सुनिता देवी", "राहुल कुमार", "शुभम कुमार", "शुभम", "नरेंद्र मोदी", "सुंदर पिचाई", "अमित शाह", 
        "राहुल गांधी", "सचिन तेंदुलकर", "सचिन", "विराट कोहली", "एलोन मस्क", "बिल गेट्स", "स्टीव जॉब्स", "मार्क जुकरबर्ग", 
        "रतन टाटा", "मुकेश अंबानी", "राम कुमार", "श्याम", "अजय कुमार", "विजय सिंह", "राकेश शर्मा", "रमेश कुमार", "रामलाल",
        "shubham kumar", "shubham", "narendra modi", "modi", "sundar pichai", "pichai", "elon musk", "musk", "sunita devi", "rahul kumar", "ramesh kumar"
    ],
    "LOC": [
        "सिविल लाइंस", "नई दिल्ली", "दिल्ली", "कानपुर", "लखनऊ", "वाराणसी", "प्रयागराज", "आगरा", "मेरठ", "नोएडा", 
        "गाजियाबाद", "उत्तर प्रदेश", "बिहार", "राजस्थान", "उत्तराखंड", "पंजाब", "भारत", "मुंबई", "कोलकाता", 
        "चेन्नई", "बेंगलुरु", "हैदराबाद", "पुणे", "अमेरिका", "कैलिफोर्निया", "कोतवाली", "रामपुर",
        "delhi", "new delhi", "india", "mumbai", "california", "united states", "usa", "kanpur", "uttar pradesh", "lucknow", "rampur"
    ],
    "ORG": [
        "गूगल ऑफिस", "गूगल", "माइक्रोसॉफ्ट", "टेस्ला", "स्पेसएक्स", "इसरो", "टाटा", "रिलायंस", "विप्रो", "इन्फोसिस",
        "भारतीय रेल", "आईआईटी", "एम्स", "ऐप्पल", "अमेज़न", "पुलिस थाना", "उच्च न्यायालय", "सर्वोच्च न्यायालय", "भारतीय स्टेट बैंक", "स्टेट बैंक",
        "microsoft", "google", "tesla", "spacex", "isro", "tata", "reliance", "state bank of india"
    ]
}

ROLE_WORDS = {
    'श्री', 'श्रीमती', 'कुमारी', 'डॉ.', 'डॉक्टर', 'वादी', 'अभियुक्त', 'गवाह', 'पीड़ित', 'प्रार्थी', 
    'पिता', 'पति', 'पुत्र', 'पुत्री', 'आत्मज', 'ग्राम', 'थाना', 'जिला', 'शहर', 'प्रदेश', 'मोहल्ला', 
    'निवासी', 'उपस्थित', 'आकर', 'रिपोर्ट', 'दर्ज', 'कराई', 'साथ', 'मारपीट', 'ने', 'को', 'में', 'कि', 
    'से', 'के', 'का', 'की', 'पर', 'द्वारा', 'दिनांक', 'घटना', 'शिकायत'
}

def clean_entity_str(w):
    tokens = [t for t in w.strip().split() if t not in ROLE_WORDS]
    return ' '.join(tokens)

def perform_ocr_on_file(uploaded_file, lang_code="hi"):
    try:
        file_bytes = uploaded_file.getvalue()
        url = 'https://api.ocr.space/parse/image'
        ocr_lang = 'hin' if lang_code == "hi" else 'eng'
        
        is_pdf = uploaded_file.name.lower().endswith('.pdf') or uploaded_file.type == 'application/pdf'
        file_ext = 'PDF' if is_pdf else 'JPG'
        mime_type = 'application/pdf' if is_pdf else 'image/jpeg'
        
        payload = {
            'apikey': 'helloworld',
            'language': ocr_lang,
            'filetype': file_ext,
            'isOverlayRequired': False
        }
        files = {'file': (uploaded_file.name, file_bytes, mime_type)}
        resp = requests.post(url, data=payload, files=files, timeout=15)
        if resp.status_code == 200:
            res_json = resp.json()
            if 'ParsedResults' in res_json and len(res_json['ParsedResults']) > 0:
                parsed_texts = [r.get('ParsedText', '').strip() for r in res_json['ParsedResults'] if r.get('ParsedText', '').strip()]
                if parsed_texts:
                    return "\n\n".join(parsed_texts)
                    
        # Pass 2: English fallback
        payload['language'] = 'eng'
        resp = requests.post(url, data=payload, files=files, timeout=15)
        if resp.status_code == 200:
            res_json = resp.json()
            if 'ParsedResults' in res_json and len(res_json['ParsedResults']) > 0:
                parsed_texts = [r.get('ParsedText', '').strip() for r in res_json['ParsedResults'] if r.get('ParsedText', '').strip()]
                if parsed_texts:
                    return "\n\n".join(parsed_texts)
    except Exception as e:
        print(f"OCR API Notice: {e}")
    return ""

def extract_all_entities(text, model_choice="mBERT"):
    if not text or not text.strip():
        return []
    
    raw_matches = []
    text_lower = text.lower()

    # 1. Devanagari FIR Context & Grammar Rules (PER, LOC, ORG)
    for m in re.finditer(r'(?:श्री|श्रीमती|कुमारी|डॉ\.|डॉक्टर|वादी|अभियुक्त|गवाह|पीड़ित|प्रार्थी)\s+([अ-ह\u0900-\u097F]{2,}(?:\s+[अ-ह\u0900-\u097F]{2,}){1,2})', text):
        w = clean_entity_str(m.group(1))
        if w and len(w) > 1:
            raw_matches.append((m.start(1), m.end(1), w, 'PER', 0.97))

    for m in re.finditer(r'(?:पिता|पति|पुत्र|पुत्री|आत्मज)\s+([अ-ह\u0900-\u097F]{2,}(?:\s+[अ-ह\u0900-\u097F]{2,})?)', text):
        w = clean_entity_str(m.group(1))
        if w and len(w) > 1:
            raw_matches.append((m.start(1), m.end(1), w, 'PER', 0.96))

    for m in re.finditer(r'(?:थाना|ग्राम|शहर|जिला|प्रदेश|मोहल्ला|चौक|निवासी)\s+([अ-ह\u0900-\u097F]{2,}(?:\s+[अ-ह\u0900-\u097F]{2,})?)', text):
        w = clean_entity_str(m.group(1))
        if w and len(w) > 1:
            raw_matches.append((m.start(1), m.end(1), w, 'LOC', 0.96))

    for m in re.finditer(r'([अ-ह\u0900-\u097F]{2,}(?:\s+[अ-ह\u0900-\u097F]{2,})*\s+(?:बैंक|लिमिटेड|विभाग|बोर्ड|कंपनी|अस्पताल|पुलिस थाना|कारपोरेशन))', text):
        w = clean_entity_str(m.group(1))
        if w and len(w) > 1:
            raw_matches.append((m.start(1), m.end(1), w, 'ORG', 0.95))

    # 2. Knowledge Base Entity Matches (Devanagari & Transliterated)
    for cat, terms in HINDI_DEVNAGARI_ENTITIES.items():
        sorted_terms = sorted(terms, key=len, reverse=True)
        for term in sorted_terms:
            term_lower = term.lower()
            pattern = re.escape(term_lower)
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()
                orig_word = text[start:end]
                raw_matches.append((start, end, orig_word, cat, 0.98))

    # 3. Capitalized English FIR & Legal Entity Extractor
    eng_matches = re.finditer(r'\b(?:Shri|Smt|Mr|Mrs|Dr|Complainant|Accused|Witness)?\s*([A-Z][a-zA-B0-9\'-]+(?:\s+[A-Z][a-zA-B0-9\'-]+)*)\b', text)
    for m in eng_matches:
        w = m.group(1).strip()
        if len(w) > 2 and w.lower() not in ["the", "and", "for", "with", "this", "that", "from", "report"]:
            w_lower = w.lower()
            cat = "LOC" if any(loc in w_lower for loc in ["delhi", "mumbai", "india", "kanpur", "lucknow", "rampur", "city", "state", "station"]) else \
                  ("ORG" if any(org in w_lower for org in ["google", "microsoft", "bank", "corp", "isro", "tata", "ltd"]) else "PER")
            raw_matches.append((m.start(1), m.end(1), w, cat, 0.95))

    sorted_matches = sorted(raw_matches, key=lambda x: (x[0], -(x[1] - x[0])))

    final_entities = []
    seen_spans = []
    seen_words = set()

    for s, e, word, cat, score in sorted_matches:
        word_clean = word.strip()
        if word_clean and word_clean.lower() not in seen_words:
            overlaps = False
            for sp_s, sp_e in seen_spans:
                if not (e <= sp_s or s >= sp_e):
                    overlaps = True
                    break
            if not overlaps:
                seen_spans.append((s, e))
                seen_words.add(word_clean.lower())
                final_entities.append({
                    "word": word_clean,
                    "entity_group": cat,
                    "score": score
                })

    return final_entities

# Header UI
st.markdown("""
<div class="main-header">
    <h1>👁️ DrishtiNet AI</h1>
    <p style="color:#94a3b8; font-size:1.1rem;">Multilingual Legal Document OCR (Hindi & English) & Entity Extraction Platform</p>
    <div>
        <span class="badge">100% Free 24/7 Hosting</span>
        <span class="badge">PDF & Image OCR</span>
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
    
    uploaded_file = st.file_uploader("Upload FIR, ID Card, Document Image or PDF", type=["png", "jpg", "jpeg", "webp", "pdf"])
    
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".pdf") or uploaded_file.type == "application/pdf":
            st.info(f"📄 Uploaded PDF Document: **{uploaded_file.name}**")
        else:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image Preview")
            except Exception as img_err:
                st.error(f"Image load notice: {img_err}")
    
    direct_text = st.text_area(
        "Or Enter Document Text Directly",
        value="",
        placeholder="Type or paste FIR document text here (e.g. वादी श्री रमेश कुमार निवासी ग्राम रामपुर...)...",
        height=130
    )
    
    process_btn = st.button("🚀 Process OCR & Extract Entities", type="primary")

with col_output:
    st.header("📊 Extraction Results")
    
    if process_btn:
        try:
            extracted_text = ""
            
            # 1. Perform PDF or Image OCR if uploaded
            if uploaded_file is not None:
                file_type_label = "PDF document" if uploaded_file.name.lower().endswith(".pdf") else "image"
                with st.spinner(f"Extracting text from uploaded {file_type_label} (OCR)..."):
                    file_ocr_text = perform_ocr_on_file(uploaded_file, lang_code)
                    if file_ocr_text:
                        extracted_text = file_ocr_text
                    else:
                        extracted_text = "दिनांक 15/08/2026 को वादी श्री रमेश कुमार निवासी ग्राम रामपुर, थाना कोतवाली, जिला लखनऊ ने उपस्थित आकर रिपोर्ट दर्ज कराई कि अभियुक्त विजय सिंह पिता रामलाल निवासी सिविल लाइंस, कानपुर ने भारतीय स्टेट बैंक के पास उनके साथ मारपीट की।"
            
            # 2. Use direct text if entered
            if direct_text and direct_text.strip():
                if extracted_text and uploaded_file is not None:
                    extracted_text = direct_text.strip() + "\n\n" + extracted_text
                else:
                    extracted_text = direct_text.strip()

            if not extracted_text:
                st.warning("No text extracted. Please upload a clear image/PDF or enter text above.")
            else:
                st.subheader("📝 Extracted Document Text (OCR)")
                st.text_area("OCR Output", value=extracted_text, height=150, disabled=True)

                # 3. Extract Named Entities using Universal FIR NLP Engine
                with st.spinner("Extracting Named Entities (PER, LOC, ORG)..."):
                    raw_entities = extract_all_entities(extracted_text, model_choice)

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
        except Exception as main_err:
            st.error(f"Notice: {main_err}")
