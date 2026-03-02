import streamlit as st
import joblib
import re
import numpy as np
import plotly.graph_objects as go

st.markdown("""
<style>
body {
    background-color: #f7fbff;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Light vs Dark Analyzer", layout="wide")

# 🌞 标题
st.title("🌞 Light vs Dark Speech Analyzer")
st.markdown("Analyze whether a speech leans toward Light or Dark ideology.")

# 加载模型
vectorizer_bi = joblib.load("models/vectorizer_bi.pkl")
lr = joblib.load("models/lr_bi.pkl")
svm = joblib.load("models/svm.pkl")
mnb = joblib.load("models/mnb.pkl")
cnb = joblib.load("models/cnb.pkl")

# 输入框
text_input = st.text_area("Enter your speech here:", height=200)

if st.button("Analyze"):

    if len(text_input.strip()) == 0:
        st.warning("Please enter some text.")
    else:

        # 句子切分
        sentences = re.split(r'(?<=[.!?]) +', text_input)
        sentences = [s.strip() for s in sentences if s.strip()]

        X = vectorizer_bi.transform(sentences)

        # 获取概率
        scores_lr = lr.predict_proba(X)[:,1]
        scores_mnb = mnb.predict_proba(X)[:,1]
        scores_cnb = cnb.predict_proba(X)[:,1]

        scores_svm = svm.decision_function(X)
        scores_svm = (scores_svm - scores_svm.min()) / (scores_svm.max() - scores_svm.min())

        # 整体平均概率
        avg_scores = {
            "Logistic Regression": np.mean(scores_lr),
            "Multinomial NB": np.mean(scores_mnb),
            "Complement NB": np.mean(scores_cnb),
            "SVM": np.mean(scores_svm),
        }

        st.subheader("🔮 Model Predictions")

        for model, score in avg_scores.items():
            st.metric(model, f"{score:.2f} Dark")

        # 折线图
        st.subheader("📈 Sentence-level Dark Probability")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=scores_lr,
            mode='lines+markers',
            name="Logistic"
        ))

        fig.add_trace(go.Scatter(
            y=scores_svm,
            mode='lines+markers',
            name="SVM"
        ))

        fig.update_layout(
            yaxis=dict(range=[0,1]),
            xaxis_title="Sentence Index",
            yaxis_title="Dark Probability",
            template="plotly_white"
        )

        fig.add_hline(y=0.5, line_dash="dash")

        st.plotly_chart(fig, use_container_width=True)