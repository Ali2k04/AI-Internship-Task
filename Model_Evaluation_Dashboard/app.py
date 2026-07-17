"""
app.py — Model Evaluation Dashboard
------------------------------------
Interactive Streamlit dashboard that compares ML model performance
(Accuracy, Precision, Recall, F1-score) for a text classification task.

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils import (
    load_artifacts,
    train_and_evaluate,
    save_artifacts,
    get_confusion_matrix,
    get_roc_curve,
)

st.set_page_config(
    page_title="Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Evaluation Dashboard")
st.caption("Comparing Accuracy, Precision, Recall and F1-score across ML models")

# ---------------------------------------------------------------------------
# Sidebar — choose data source
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")

data_source = st.sidebar.radio(
    "Dataset source",
    ["Use built-in dataset (pretrained)", "Upload my own CSV"],
)


@st.cache_data(show_spinner=False)
def _load_builtin():
    return load_artifacts()


if data_source == "Upload my own CSV":
    st.sidebar.markdown("Your CSV needs a **text** column and a **label** column "
                         "(e.g. spam/ham, real/fake, positive/negative).")
    uploaded = st.sidebar.file_uploader("Upload dataset (.csv)", type=["csv"])

    if uploaded is None:
        st.info(
            "👈 Upload a CSV with `text` and `label` columns to train the models on "
            "your own data, or switch back to the built-in dataset in the sidebar."
        )
        st.stop()

    df_upload = pd.read_csv(uploaded)
    cols = list(df_upload.columns)
    text_col = st.sidebar.selectbox(
        "Text column", cols, index=cols.index("text") if "text" in cols else 0
    )
    label_col = st.sidebar.selectbox(
        "Label column", cols,
        index=cols.index("label") if "label" in cols else min(1, len(cols) - 1),
    )

    with st.spinner("Training Logistic Regression, SVM, and Naive Bayes on your data..."):
        metrics_df, vectorizer, artifacts, pos_label = train_and_evaluate(
            df_upload, text_col=text_col, label_col=label_col
        )
        save_artifacts(metrics_df, vectorizer, artifacts, pos_label)
    st.sidebar.success("Models trained on your uploaded data ✅")
else:
    metrics_df, vectorizer, artifacts, pos_label = _load_builtin()

model_names = list(artifacts.keys())

# ---------------------------------------------------------------------------
# Section 1 — Model comparison table
# ---------------------------------------------------------------------------
st.subheader("📋 Model Comparison Table")

display_df = metrics_df.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
    display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"

st.dataframe(display_df, use_container_width=True, hide_index=True)

csv_bytes = metrics_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Export metrics as CSV",
    data=csv_bytes,
    file_name="model_metrics.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Section 2 — Bar chart comparing metrics across models
# ---------------------------------------------------------------------------
st.subheader("📈 Metric Comparison")

metric_cols = ["Accuracy", "Precision", "Recall", "F1-score"]
chart_df = metrics_df.melt(
    id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score"
)
chart_df["Score"] = (chart_df["Score"] * 100).round(2)

fig_bar = px.bar(
    chart_df, x="Model", y="Score", color="Metric", barmode="group",
    text="Score", labels={"Score": "Score (%)"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
fig_bar.update_layout(yaxis_range=[0, 110], legend_title_text="")
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3 — Model selection + detailed inspection
# ---------------------------------------------------------------------------
st.subheader("🔍 Detailed Model Inspection")

selected_model = st.selectbox("Select a model", model_names)
row = metrics_df[metrics_df["Model"] == selected_model].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{row['Accuracy'] * 100:.1f}%")
c2.metric("Precision", f"{row['Precision'] * 100:.1f}%")
c3.metric("Recall", f"{row['Recall'] * 100:.1f}%")
c4.metric("F1-score", f"{row['F1-score'] * 100:.1f}%")

art = artifacts[selected_model]
y_test, y_pred, y_proba = art["y_test"], art["y_pred"], art["y_proba"]

col_left, col_right = st.columns(2)

# --- Confusion matrix ---
with col_left:
    st.markdown("**🧩 Confusion Matrix**")
    cm = get_confusion_matrix(y_test, y_pred)
    labels = [f"Not {pos_label}", pos_label]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm, x=labels, y=labels, colorscale="Blues",
            text=cm, texttemplate="%{text}", showscale=False,
        )
    )
    fig_cm.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        yaxis_autorange="reversed", height=350,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# --- ROC curve ---
with col_right:
    st.markdown("**📉 ROC Curve**")
    fpr, tpr, roc_auc = get_roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))
    fig_roc.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                   line=dict(dash="dash", color="gray"), name="Random guess")
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=350, legend=dict(x=0.55, y=0.05),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

st.divider()
st.caption("Built with Streamlit • scikit-learn • Plotly — Model Evaluation Dashboard project")
