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
        # 真·模式切换
        # =========================
        if overall_avg > 0.5:
            mode = "dark"
            verdict = "🌑 DARK SIDE"

            st.markdown("""
            <style>
            .stApp {
                background-color: #0b1120;
                color: white;
            }

            /* 输入框 */
            textarea {
                background-color: #111827 !important;
                color: white !important;
            }

            /* 按钮 */
            button {
                background-color: #1f2937 !important;
                color: white !important;
            }

            /* metric卡片 */
            [data-testid="metric-container"] {
                background-color: #111827;
                border-radius: 16px;
                padding: 15px;
            }

            /* 标题 */
            h1, h2, h3, h4 {
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)

        else:
            mode = "light"
            verdict = "🌕 LIGHT SIDE"

            st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(135deg, #fef9c3, #fde68a, #fcd34d);
                color: black;
            }

            textarea {
                background-color: white !important;
                color: black !important;
            }

            button {
                background-color: #facc15 !important;
                color: black !important;
            }

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

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Logistic", f"{avg_scores['Logistic Regression']:.2f}")
        col2.metric("MNB", f"{avg_scores['Multinomial NB']:.2f}")
        col3.metric("CNB", f"{avg_scores['Complement NB']:.2f}")
        col4.metric("SVM", f"{avg_scores['SVM']:.2f}")

        # =========================
        # 图表
        # =========================
        st.subheader("📈 Sentence-level Dark Probability")

        if mode == "dark":
            paper_bg = "#0b1120"
            plot_bg = "#111827"
            font_color = "white"
        else:
            paper_bg = "#fff7ed"
            plot_bg = "white"
            font_color = "black"

        fig = go.Figure()

        fig.add_trace(go.Scatter(y=scores_lr, mode='lines+markers', name="Logistic", line=dict(width=4)))
        fig.add_trace(go.Scatter(y=scores_mnb, mode='lines+markers', name="MNB", line=dict(width=4)))
        fig.add_trace(go.Scatter(y=scores_cnb, mode='lines+markers', name="CNB", line=dict(width=4)))
        fig.add_trace(go.Scatter(y=scores_svm, mode='lines+markers', name="SVM", line=dict(width=4)))

        fig.update_layout(
            yaxis=dict(range=[0,1]),
            xaxis_title="Sentence Index",
            yaxis_title="Dark Probability",
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            font=dict(color=font_color),
            height=550
        )

        fig.add_hline(y=0.5, line_dash="dash")

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # 极端句子
        # =========================
        darkest_index = np.argmax(scores_lr)
        lightest_index = np.argmin(scores_lr)

        col_dark, col_light = st.columns(2)

        col_dark.markdown("### 🌑 Most Dark Sentence")
        col_dark.write(sentences[darkest_index])

        col_light.markdown("### 🌕 Most Light Sentence")
        col_light.write(sentences[lightest_index])