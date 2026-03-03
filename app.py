import streamlit as st
import joblib
import re
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Light vs Dark Analyzer", layout="wide")

# =========================
# 加载模型
# =========================
bundle_lr  = joblib.load("models/lr.pkl")
bundle_mnb = joblib.load("models/mnb.pkl")
bundle_cnb = joblib.load("models/cnb.pkl")
bundle_bnb = joblib.load("models/bnb_best_model.pkl")

models = {
    "Logistic Regression": bundle_lr,
    "Multinomial NB": bundle_mnb,
    "Complement NB": bundle_cnb,
    "Bernoulli NB (Best)": bundle_bnb
}

# =========================
# 统一预测函数
# =========================
def predict_force_alignment(text, bundle, threshold=0.5):
    proba = bundle.predict_proba(text)
    return {
        "Light": proba[0],
        "Dark": proba[1],
        "prediction": "Light" if proba[0] >= threshold else "Dark"
    }

# =========================
# UI
# =========================
st.title("✨ Light vs Dark Ideology Analyzer")
st.markdown("Enter a speech and reveal its alignment.")

text_input = st.text_area("Enter your speech here:", height=200)

if st.button("⚡ Analyze the Force"):

    if len(text_input.strip()) == 0:
        st.warning("Please enter some text.")
    else:

        sentences = re.split(r'(?<=[.!?]) +', text_input)
        sentences = [s.strip() for s in sentences if s.strip()]

        results = {}
        sentence_scores = {}

        # =========================
        # 预测
        # =========================
        for name, bundle in models.items():
            preds = []

            for s in sentences:
                res = predict_force_alignment(s, bundle)
                preds.append(res["Dark"])

            sentence_scores[name] = preds
            results[name] = np.mean(preds)

        overall_avg = np.mean(list(results.values()))
        verdict = "🌑 DARK SIDE" if overall_avg > 0.5 else "🌕 LIGHT SIDE"

        # =========================
        # 动态背景
        # =========================
        if overall_avg > 0.5:
            st.markdown("""
            <style>
            .stApp { background-color: #0b1120; color: white; }
            textarea { background-color: #111827 !important; color: white !important; }
            button { background-color: #1f2937 !important; color: white !important; }
            [data-testid="metric-container"] {
                background-color: #111827;
                border-radius: 16px;
                padding: 15px;
            }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(135deg, #fef9c3, #fde68a, #fcd34d);
                color: black;
            }
            textarea { background-color: white !important; color: black !important; }
            button { background-color: #facc15 !important; color: black !important; }
            [data-testid="metric-container"] {
                background-color: white;
                border-radius: 16px;
                padding: 15px;
            }
            </style>
            """, unsafe_allow_html=True)

        st.markdown(f"<h1 style='text-align:center'>{verdict}</h1>", unsafe_allow_html=True)

        # =========================
        # 横向展示
        # =========================
        st.subheader("🔮 Model Comparison")

        cols = st.columns(len(results))
        for col, (name, score) in zip(cols, results.items()):
            col.metric(name, f"{score:.2f}")

        # =========================
        # 折线图
        # =========================
        st.subheader("📈 Sentence-level Dark Probability")

        fig = go.Figure()

        for name in sentence_scores:
            fig.add_trace(go.Scatter(
                y=sentence_scores[name],
                mode='lines+markers',
                name=name,
                line=dict(width=4)
            ))

        fig.update_layout(
            yaxis=dict(range=[0,1]),
            xaxis_title="Sentence Index",
            yaxis_title="Dark Probability",
            height=550
        )

        fig.add_hline(y=0.5, line_dash="dash")

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # 极端句子（用 Logistic 作参考）
        # =========================
        darkest_index = np.argmax(sentence_scores["Logistic Regression"])
        lightest_index = np.argmin(sentence_scores["Logistic Regression"])

        col_dark, col_light = st.columns(2)

        col_dark.markdown("### 🌑 Most Dark Sentence")
        col_dark.write(sentences[darkest_index])

        col_light.markdown("### 🌕 Most Light Sentence")
        col_light.write(sentences[lightest_index])