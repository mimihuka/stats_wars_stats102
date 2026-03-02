import streamlit as st
import joblib
import re
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Light vs Dark Analyzer", layout="wide")

# 加载模型
vectorizer = joblib.load("models/vectorizer.pkl")
lr = joblib.load("models/lr.pkl")
svm = joblib.load("models/svm.pkl")
mnb = joblib.load("models/mnb.pkl")
cnb = joblib.load("models/cnb.pkl")

st.title("🌞 Light vs Dark Speech Analyzer")
st.markdown("Analyze whether a speech leans toward Light or Dark ideology.")

text_input = st.text_area("Enter your speech here:", height=200)

if st.button("Analyze"):

    if len(text_input.strip()) == 0:
        st.warning("Please enter some text.")
    else:

        sentences = re.split(r'(?<=[.!?]) +', text_input)
        sentences = [s.strip() for s in sentences if s.strip()]

        X = vectorizer.transform(sentences)

        # 预测
        scores_lr = lr.predict_proba(X)[:,1]
        scores_mnb = mnb.predict_proba(X)[:,1]
        scores_cnb = cnb.predict_proba(X)[:,1]

        scores_svm = svm.decision_function(X)
        scores_svm = (scores_svm - scores_svm.min()) / (scores_svm.max() - scores_svm.min())

        avg_scores = {
            "Logistic Regression": np.mean(scores_lr),
            "Multinomial NB": np.mean(scores_mnb),
            "Complement NB": np.mean(scores_cnb),
            "SVM": np.mean(scores_svm),
        }

        overall_avg = np.mean(list(avg_scores.values()))

        # 🎨 动态背景
        if overall_avg > 0.5:
            bg_color = "#0f172a"   # 深蓝
            text_color = "white"
        else:
            bg_color = "#fff9c4"   # 阳光黄
            text_color = "black"

        st.markdown(f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        [data-testid="metric-container"] {{
            background-color: rgba(255,255,255,0.2);
            border-radius: 12px;
            padding: 15px;
        }}
        </style>
        """, unsafe_allow_html=True)

        # 🔮 横向模型展示
        st.subheader("🔮 Model Predictions")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Logistic Regression", f"{avg_scores['Logistic Regression']:.2f}")
        col2.metric("Multinomial NB", f"{avg_scores['Multinomial NB']:.2f}")
        col3.metric("Complement NB", f"{avg_scores['Complement NB']:.2f}")
        col4.metric("SVM", f"{avg_scores['SVM']:.2f}")

        # 📈 折线图
        st.subheader("📈 Sentence-level Dark Probability")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=scores_lr,
            mode='lines+markers',
            name="Logistic Regression",
            line=dict(width=3)
        ))

        fig.add_trace(go.Scatter(
            y=scores_mnb,
            mode='lines+markers',
            name="Multinomial NB",
            line=dict(width=3)
        ))

        fig.add_trace(go.Scatter(
            y=scores_cnb,
            mode='lines+markers',
            name="Complement NB",
            line=dict(width=3)
        ))

        fig.add_trace(go.Scatter(
            y=scores_svm,
            mode='lines+markers',
            name="SVM",
            line=dict(width=3)
        ))

        fig.update_layout(
            yaxis=dict(range=[0,1]),
            xaxis_title="Sentence Index",
            yaxis_title="Dark Probability",
            template="plotly_white",
            legend=dict(orientation="h", y=-0.2),
            height=500
        )

        fig.add_hline(y=0.5, line_dash="dash")

        st.plotly_chart(fig, use_container_width=True)

        # 🔥 找最Dark和最Light句子（用LR作为代表）
        darkest_index = np.argmax(scores_lr)
        lightest_index = np.argmin(scores_lr)

        darkest_sentence = sentences[darkest_index]
        lightest_sentence = sentences[lightest_index]

        st.subheader("🌓 Extreme Sentences")

        col_dark, col_light = st.columns(2)

        col_dark.markdown("### 🌑 Most Dark Sentence")
        col_dark.write(darkest_sentence)

        col_light.markdown("### 🌕 Most Light Sentence")
        col_light.write(lightest_sentence)