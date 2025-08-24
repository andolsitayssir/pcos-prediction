import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page config must be first
st.set_page_config(page_title="PCOS Risk Checker", page_icon="🩺", layout="centered")

# Force light hints + Bootstrap and Icons
st.markdown(
    '''
    <meta name="color-scheme" content="light">
    <meta name="theme-color" content="#fff0f6">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    ''',
    unsafe_allow_html=True,
)

# Load custom CSS (ensures light cute colors)
def load_css():
    css_path = "app/static/style.css"
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Header
st.markdown(
    """
    <div class="card header position-relative" style="display:flex; justify-content:center; align-items:center; text-align:center;">
        <i class="bi bi-balloon-heart-fill floating-icon" style="left:14px; top:12px; font-size:2.2rem; color:#ff90c1;"></i>
        <i class="bi bi-stars floating-icon" style="right:14px; top:18px; font-size:2rem; color:#ffb6d5;"></i>
        <div style="font-size:34px; margin-right:12px;">🧬</div>
        <div>
            <h1 style="margin:0">PCOS Risk Check</h1>
            <div style="color:#666">Answer simple questions — all required</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model/pipeline
rf_model = None
scaler = None
pca = None
top_features = None
try:
    pipeline = joblib.load("app/models/pipeline_v1.pkl")
    rf_model = pipeline.get("model")
    scaler = pipeline.get("scaler")
    pca = pipeline.get("pca")
    top_features = list(pipeline.get("top_features"))
except Exception:
    try:
        rf_model = joblib.load("app/models/random_forest_model.pkl")
        scaler = joblib.load("app/models/scaler.pkl")
        pca = joblib.load("app/models/pca_model.pkl")
        top_features = list(joblib.load("app/models/top_features.pkl"))
    except Exception:
        st.error("Unable to load the model. Regenerate './app/models/pipeline_v1.pkl' from the training notebook.")
        st.stop()

# Medians for missing values fallback
def _fallback_medians():
    return {
        'Follicle No. (R)': 6,
        'Follicle No. (L)': 6,
        'Skin darkening (Y/N)': 0,
        'hair growth(Y/N)': 0,
        'Weight gain(Y/N)': 0,
        'Cycle(R/I)': 2,
        'Fast food (Y/N)': 1,
        'Pimples(Y/N)': 0,
        'BMI': 24.0,
        'Cycle length(days)': 5,
        'Hair loss(Y/N)': 0,
        'Age (yrs)': 28,
        'Waist(inch)': 32,
        'Hip(inch)': 38,
    }

def load_medians(features):
    try:
        df = pd.read_excel("data/PCOS_data_without_infertility.xlsx", sheet_name="Full_new")
        df.columns = df.columns.str.strip()
        med = df[features].median(numeric_only=True).to_dict()
        fb = _fallback_medians()
        return {k: (v if not pd.isna(v) else fb.get(k, 0)) for k, v in med.items()}
    except Exception:
        return _fallback_medians()

medians = load_medians(top_features)

# Session state
if "page" not in st.session_state:
    st.session_state.page = 0
if "resp" not in st.session_state:
    st.session_state.resp = {}

def next_page():
    st.session_state.page += 1
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

def prev_page():
    if st.session_state.page > 0:
        st.session_state.page -= 1
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()


# Helpers
def _to_float(x):
    try:
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
        return float(x)
    except Exception:
        return None

def _prefill_from_resp(mapping):
    # mapping: list of (widget_key, resp_key, default)
    for wkey, rkey, default in mapping:
        if wkey not in st.session_state:
            st.session_state[wkey] = st.session_state.resp.get(rkey, default)

def _parse_optional_number(x):
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("", "i don't know", "dont know", "don't know", "idk", "i do not know", "n/a", "na", "unknown"):
            return None
    return _to_float(x)

def assemble_input_vector():
    resp = dict(st.session_state.resp)
    vals: Dict[str, float] = {}

    # Map Yes/No
    for k in [
        'Skin darkening (Y/N)',
        'hair growth(Y/N)',
        'Weight gain(Y/N)',
        'Pimples(Y/N)',
        'Hair loss(Y/N)',
        'Fast food (Y/N)'
    ]:
        v = resp.get(k)
        if isinstance(v, str):
            s = v.lower().strip()
            if s in ("yes", "y", "1"):
                vals[k] = 1
            elif s in ("no", "n", "0"):
                vals[k] = 0

    # Cycle mapping 1/2
    cyc = resp.get('Cycle(R/I)')
    if isinstance(cyc, str):
        s = cyc.lower().strip()
        if s.startswith('reg') or s in ('r', '1'):
            vals['Cycle(R/I)'] = 1
        elif s.startswith('irr') or s in ('i', '2'):
            vals['Cycle(R/I)'] = 2
    else:
        vals['Cycle(R/I)'] = _to_float(cyc)

    # Numerics (follicles are optional)
    for k in [
        'Age (yrs)', 'Height (Cm)', 'Weight (Kg)', 'Cycle length(days)',
        'Waist(inch)', 'Hip(inch)'
    ]:
        if k in resp:
            vals[k] = _to_float(resp.get(k))

    # Optional follicles
    for k in ['Follicle No. (R)', 'Follicle No. (L)']:
        if k in resp:
            vals[k] = _parse_optional_number(resp.get(k))

    # BMI compute
    h = vals.get('Height (Cm)')
    w = vals.get('Weight (Kg)')
    bmi = w / ((h / 100.0) ** 2) if (h and w and h > 0) else None
    vals['BMI'] = bmi

    # Row aligned with top_features using medians for missing
    row = []
    for k in top_features:
        v = vals.get(k, resp.get(k, None))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = medians.get(k, 0)
        row.append(v)
    return pd.DataFrame([row], columns=top_features)
def reset_quiz():
    # Clear saved answers
    st.session_state.resp = {}

    # Drop widget keys so inputs render empty/defaults
    for k in [
        "age_txt", "height_txt", "weight_txt",
        "skin_dark", "hair_growth", "weight_gain", "pimples", "hair_loss", "fast_food",
        "cycle_reg", "cycle_len_txt",
        "waist_cm_txt", "hip_cm_txt", "fol_r_txt", "fol_l_txt",
    ]:
        st.session_state.pop(k, None)

    # Go to first page and re-run
    st.session_state.page = 0
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# Cute modal helper (JS Bootstrap)
def show_required_modal(modal_id: str):
    st.markdown(
        f"""
        <div class="modal fade cute-modal" id="{modal_id}" tabindex="-1" aria-labelledby="{modal_id}Label" aria-hidden="true">
          <div class="modal-dialog"><div class="modal-content">
            <div class="modal-header">
              <h5 class="modal-title" id="{modal_id}Label">Missing required fields</h5>
              <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body">
              <i class="bi bi-exclamation-triangle-fill" style="color:#d6336c;font-size:2rem;"></i>
              <p>Please fill in all fields marked with an asterisk (*) before continuing.</p>
            </div>
            <div class="modal-footer">
              <button type="button" class="btn btn-pink" data-bs-dismiss="modal">OK</button>
            </div>
          </div></div>
        </div>
        <script>
          const m = new bootstrap.Modal(document.getElementById('{modal_id}'));
          m.show();
        </script>
        """,
        unsafe_allow_html=True,
    )

# UI
card = st.container()

with card:
    # PAGE 0 — General
    if st.session_state.page == 0:
        st.markdown(
    """
    <div class="alert alert-info" role="alert" style="margin-bottom:12px; border-radius:12px; padding:14px; font-size:15px;">
      
      <div><strong>For strong girls, teens, and women:</strong><br>
      Your body tells a unique story, and every chapter matters.  
      This gentle check is here to help you notice patterns that may be linked to PCOS.  
      Think of it as a friendly guide—not a diagnosis, not a verdict.   
      <br><br>
      <strong>If something feels off,</strong> remember: you are never alone in this journey. You deserve calm, kind care from a specialist who listens, supports, and helps you find clarity and balance.  
      <br><br>
      <em>Your health is your superpower, treat it with love. 💗</em>
   
    </div>
    """,
    unsafe_allow_html=True,
)

        st.subheader("1 — General information")
        
        _prefill_from_resp([
            ("age_txt", "Age (yrs)", ""),
            ("height_txt", "Height (Cm)", ""),
            ("weight_txt", "Weight (Kg)", ""),
        ])

        with st.form("form_page0", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Age (years) *", key="age_txt", placeholder="e.g. 28")
                st.text_input("Height (cm) *", key="height_txt", placeholder="e.g. 160")
            with col2:
                st.text_input("Weight (kg) *", key="weight_txt", placeholder="e.g. 60")

            submit_next0 = st.form_submit_button("Next ➜")

        if submit_next0:
            age_raw = st.session_state.get("age_txt")
            height_raw = st.session_state.get("height_txt")
            weight_raw = st.session_state.get("weight_txt")
            st.session_state.resp["Age (yrs)"] = age_raw.strip() if isinstance(age_raw, str) and age_raw.strip() else None
            st.session_state.resp["Height (Cm)"] = height_raw.strip() if isinstance(height_raw, str) and height_raw.strip() else None
            st.session_state.resp["Weight (Kg)"] = weight_raw.strip() if isinstance(weight_raw, str) and weight_raw.strip() else None

            required = ["Age (yrs)", "Height (Cm)", "Weight (Kg)"]
            if any(st.session_state.resp.get(k) is None for k in required):
                show_required_modal("modalRequired0")
            else:
                next_page()

        st.markdown("</div>", unsafe_allow_html=True)

    # PAGE 1 — Symptoms
    elif st.session_state.page == 1:
       
        st.subheader("2 — Visible symptoms")
        st.write("Answer Yes / No (all required)")

        _prefill_from_resp([
            ("skin_dark", "Skin darkening (Y/N)", "No"),
            ("hair_growth", "hair growth(Y/N)", "No"),
            ("weight_gain", "Weight gain(Y/N)", "No"),
            ("pimples", "Pimples(Y/N)", "No"),
            ("hair_loss", "Hair loss(Y/N)", "No"),
            ("fast_food", "Fast food (Y/N)", "No"),
        ])

        with st.form("form_page1", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                st.selectbox("Skin darkening? *", ("No", "Yes"), key="skin_dark")
                st.selectbox("Excess hair growth? *", ("No", "Yes"), key="hair_growth")
                st.selectbox("Unexplained weight gain? *", ("No", "Yes"), key="weight_gain")
            with c2:
                st.selectbox("Frequent pimples/acne? *", ("No", "Yes"), key="pimples")
                st.selectbox("Hair loss? *", ("No", "Yes"), key="hair_loss")
                st.selectbox("Frequent fast-food consumption? *", ("No", "Yes"), key="fast_food")
            b1, b2 = st.columns(2)
            with b1:
                prev1 = st.form_submit_button("← Previous")
            with b2:
                submit_next1 = st.form_submit_button("Next ➜")
        # Save current widget values into resp on any submit
        if prev1 or submit_next1:
            st.session_state.resp["Skin darkening (Y/N)"] = st.session_state.get("skin_dark")
            st.session_state.resp["hair growth(Y/N)"] = st.session_state.get("hair_growth")
            st.session_state.resp["Weight gain(Y/N)"] = st.session_state.get("weight_gain")
            st.session_state.resp["Pimples(Y/N)"] = st.session_state.get("pimples")
            st.session_state.resp["Hair loss(Y/N)"] = st.session_state.get("hair_loss")
            st.session_state.resp["Fast food (Y/N)"] = st.session_state.get("fast_food")

        if prev1:
            prev_page()
        elif submit_next1:
            required = [
                "Skin darkening (Y/N)", "hair growth(Y/N)", "Weight gain(Y/N)",
                "Pimples(Y/N)", "Hair loss(Y/N)", "Fast food (Y/N)"
            ]
            if any(st.session_state.resp.get(k) is None for k in required):
                show_required_modal("modalRequired1")
            else:
                next_page()

        st.markdown("</div>", unsafe_allow_html=True)

    # PAGE 2 — Cycle
    elif st.session_state.page == 2:
        st.subheader("3 — Cycle and habits")

        _prefill_from_resp([
            ("cycle_reg", "Cycle(R/I)", "Regular"),
            ("cycle_len_txt", "Cycle length(days)", ""),
        ])

        with st.form("form_page2", clear_on_submit=False):
            st.selectbox("Menstrual cycle: Regular or Irregular? *", ("Regular", "Irregular"), key="cycle_reg")
            st.text_input("Average period length (days) *", key="cycle_len_txt", placeholder="e.g. 5")

            c1, c2 = st.columns(2)
        with c1:
          prev2 = st.form_submit_button("← Previous")
        with c2:
          submit_next2 = st.form_submit_button("Next ➜")
       
        if prev2 or submit_next2:
            st.session_state.resp["Cycle(R/I)"] = st.session_state.get("cycle_reg")
            st.session_state.resp["Cycle length(days)"] = (st.session_state.get("cycle_len_txt") or "").strip() or None

        if prev2:
            prev_page()
        elif submit_next2:
            required = ["Cycle(R/I)", "Cycle length(days)"]
            if any(st.session_state.resp.get(k) is None for k in required):
                show_required_modal("modalRequired2")
            else:
                next_page()

        st.markdown("</div>", unsafe_allow_html=True)

    # PAGE 3 — Measurements (Waist/Hip in cm, convert to inches; follicles optional)
    elif st.session_state.page == 3:
        st.subheader("4 — Body measurements and exams")

        _prefill_from_resp([
            ("waist_cm_txt", "Waist (cm)", ""),
            ("hip_cm_txt", "Hip (cm)", ""),
            ("fol_r_txt", "Follicle No. (R)", ""),
            ("fol_l_txt", "Follicle No. (L)", ""),
        ])

        with st.form("form_page3", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("Waist (cm) *", key="waist_cm_txt", placeholder="e.g. 82")
                st.text_input("Hip (cm) *", key="hip_cm_txt", placeholder="e.g. 96")
            with c2:
                st.text_input("Follicle count (Right ovary) — leave empty or type 'I don't know' if unsure",
                              key="fol_r_txt", placeholder="e.g. 6 or I don't know")
                st.text_input("Follicle count (Left ovary) — leave empty or type 'I don't know' if unsure",
                              key="fol_l_txt", placeholder="e.g. 6 or I don't know")
            c1b, c2b = st.columns(2)
            with c1b:
                prev3 = st.form_submit_button("← Previous")
            with c2b:
                submit_next3 = st.form_submit_button("Next ➜")
       
        if prev3 or submit_next3:
            # Read cm inputs
            waist_cm_raw = (st.session_state.get("waist_cm_txt") or "").strip()
            hip_cm_raw = (st.session_state.get("hip_cm_txt") or "").strip()
            waist_cm = _to_float(waist_cm_raw) if waist_cm_raw else None
            hip_cm = _to_float(hip_cm_raw) if hip_cm_raw else None

            # Save UI cm values for persistence
            st.session_state.resp["Waist (cm)"] = waist_cm_raw if waist_cm_raw else None
            st.session_state.resp["Hip (cm)"] = hip_cm_raw if hip_cm_raw else None

            # Convert to inches for the model features expected by top_features
            st.session_state.resp["Waist(inch)"] = round(waist_cm / 2.54, 2) if waist_cm is not None else None
            st.session_state.resp["Hip(inch)"] = round(hip_cm / 2.54, 2) if hip_cm is not None else None

            # Follicle counts: optional
            st.session_state.resp["Follicle No. (R)"] = _parse_optional_number(st.session_state.get("fol_r_txt"))
            st.session_state.resp["Follicle No. (L)"] = _parse_optional_number(st.session_state.get("fol_l_txt"))

        if prev3:
            prev_page()
        elif submit_next3:
            # Only require waist/hip; follicles are optional
            required = ["Waist(inch)", "Hip(inch)"]
            if any(st.session_state.resp.get(k) is None for k in required):
                show_required_modal("modalRequired3")
            else:
                next_page()

        st.markdown("</div>", unsafe_allow_html=True)

   
    # PAGE 4 — Result (auto-compute, no input echo)
    else:
    
     st.subheader("Your result 💖")
     st.write("Here’s a gentle, supportive summary based on your answers.")

     with st.spinner("Analyzing your answers..."):
        try:
            input_vec = assemble_input_vector()
            scaled = scaler.transform(input_vec.values)
            pca_features = pca.transform(scaled)
            pred = int(rf_model.predict(pca_features)[0])
            prob = float(rf_model.predict_proba(pca_features)[0][1])
        except Exception:
            st.error("Prediction failed. Please ensure the saved pipeline files match the training setup.")
        else:
            if pred == 1:
                st.markdown(
                    f"""
                    <div class='alert alert-success' role='alert'>
                        <div><strong>There may be signs consistent with PCOS</strong></div>
                        <div><strong>Probability {prob:.1%}</strong></div>
                        This isn’t a diagnosis; it’s a helpful guide. You’re doing great by caring for yourself.
                        Consider chatting with a kind healthcare professional who can support you with clarity and warmth. 💗
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class='alert alert-info' role='alert'>
                        <div><strong>Low likelihood of PCOS</strong></div>
                        <div><strong>Probability {prob:.1%}</strong></div>
                        Keep listening to your body and taking gentle care of yourself.
                        If anything ever feels off, you deserve calm, reassuring support. 🌸
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # KEEP THIS INSIDE the else:
     st.markdown("---")
     if st.button("↺ Retake the test", key="retake_test"):
      reset_quiz()

    st.markdown("</div>", unsafe_allow_html=True)
 # Footer
st.markdown(
    """
    <div class="position-relative" style="height:60px;">
        <i class="bi bi-balloon-heart-fill floating-icon" style="left:14px; top:12px; font-size:2.2rem; color:#ff90c1;"></i>
        <i class="bi bi-stars floating-icon" style="right:14px; top:18px; font-size:2rem; color:#ffb6d5;"></i>
    </div>
    """,
    unsafe_allow_html=True,
)
