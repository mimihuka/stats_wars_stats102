import streamlit as st
import joblib
import re
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Light vs Dark Analyzer", layout="wide")

# =========================
# 加载模型
# =========================
vectorizer = joblib.load("models/vectorizer.pkl")
lr = joblib.load("models/lr.pkl")
svm = joblib.load("models/svm.pkl")
mnb = joblib.load("models/mnb.pkl")
cnb = joblib.load("models/cnb.pkl")

st.title("✨ Light vs Dark Ideology Analyzer")
st.markdown("Enter a speech and watch the Force reveal its alignment.")

text_input = st.text_area("Enter your speech here:", height=200)

if st.button("⚡ Analyze the Force"):

    if len(text_input.strip()) == 0:
        st.warning("Please enter some text.")
    else:

        sentences = re.split(r'(?<=[.!?]) +', text_input)
        sentences = [s.strip() for s in sentences if s.strip()]

        X = vectorizer.transform(sentences)

        # =========================
        # 预测
        # =========================
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

        # =========================
        # 动态模式 + 渐变背景
        # =========================
        if overall_avg > 0.5:
            mode = "dark"
            gradient = "linear-gradient(135deg, #0f172a, #1e3a8a, #312e81)"
            glow = "0 0 30px rgba(99,102,241,0.7)"
            font_color = "white"
            verdict = "🌑 DARK SIDE"
        else:
            mode = "light"
            gradient = "linear-gradient(135deg, #fef9c3, #fde68a, #fcd34d)"
            glow = "0 0 30px rgba(255,179,0,0.8)"
            font_color = "black"
            verdict = "🌕 LIGHT SIDE"

        st.markdown(f"""
        <style>
        .stApp {{
            background: {gradient};
            color: {font_color};
            transition: background 0.8s ease-in-out;
        }}
        .verdict {{
            font-size: 48px;
            font-weight: 800;
            text-align: center;
            margin: 30px 0;
            text-shadow: {glow};
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        [data-testid="metric-container"] {{
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 15px;
        }}
        </style>
        """, unsafe_allow_html=True)

        st.markdown(f"<div class='verdict'>{verdict}</div>", unsafe_allow_html=True)

        # =========================
        # 横向模型展示
        # =========================
        st.subheader("🔮 Model Comparison")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Logistic", f"{avg_scores['Logistic Regression']:.2f}")
        col2.metric("MNB", f"{avg_scores['Multinomial NB']:.2f}")
        col3.metric("CNB", f"{avg_scores['Complement NB']:.2f}")
        col4.metric("SVM", f"{avg_scores['SVM']:.2f}")

        # =========================
        # 图表颜色根据模式变化
        # =========================
        if mode == "dark":
            colors = ["#60a5fa", "#c084fc", "#f472b6", "#34d399"]
            paper_bg = "#0f172a"
            plot_bg = "#1e293b"
        else:
            colors = ["#f97316", "#eab308", "#22c55e", "#3b82f6"]
            paper_bg = "#fff7ed"
            plot_bg = "#ffffff"

        # =========================
        # 动画折线图
        # =========================
        st.subheader("📈 Sentence-level Dark Probability")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=scores_lr,
            mode='lines+markers',
            name="Logistic",
            line=dict(width=4, color=colors[0])
        ))

        fig.add_trace(go.Scatter(
            y=scores_mnb,
            mode='lines+markers',
            name="MNB",
            line=dict(width=4, color=colors[1])
        ))

        fig.add_trace(go.Scatter(
            y=scores_cnb,
            mode='lines+markers',
            name="CNB",
            line=dict(width=4, color=colors[2])
        ))

        fig.add_trace(go.Scatter(
            y=scores_svm,
            mode='lines+markers',
            name="SVM",
            line=dict(width=4, color=colors[3])
        ))

        fig.update_layout(
            yaxis=dict(range=[0,1]),
            xaxis_title="Sentence Index",
            yaxis_title="Dark Probability",
            legend=dict(orientation="h", y=-0.2),
            height=550,
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            font=dict(color=font_color),
            transition=dict(duration=800)
        )

        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="white" if mode=="dark" else "black"
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # 极端句子
        # =========================
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