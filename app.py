import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import json
import os
from merger_core import LLMMerger
from evaluator import ModelEvaluator, DEFAULT_TEST_CASES
from transformers import AutoTokenizer

st.set_page_config(
    page_title="LLM Model Merger",
    page_icon="⚗️",
    layout="wide"
)

st.markdown("""
<style>
    .strategy-card {
        background: #f8f9fa;
        border-left: 4px solid #0066cc;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 14px;
    }
    .result-win  { color: #1a7f37; font-weight: bold; }
    .result-lose { color: #888; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("⚗️ LLM Merger")
    st.caption("Merge two models into one")

    st.divider()
    st.subheader("1. Model Paths")

    base_path = st.text_input(
        "Base model path",
        value=r"D:\models\phi-2",
        help="The original base model both A and B were fine-tuned from"
    )
    model_a_path = st.text_input(
        "Model A path",
        value=r"D:\models\phi-2",
        help="First model to merge"
    )
    model_b_path = st.text_input(
        "Model B path",
        value=r"D:\models\phi-2",
        help="Second model to merge"
    )

    st.divider()
    st.subheader("2. Merge Strategy")

    strategy = st.selectbox(
        "Strategy",
        options=["slerp", "ties", "dare"],
        format_func=lambda x: {
            "slerp": "SLERP — Spherical blend",
            "ties":  "TIES  — Trim, Elect, Merge",
            "dare":  "DARE  — Drop and Rescale"
        }[x]
    )

    t = st.slider(
        "Blend factor (t)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0.0 = all Model A | 0.5 = equal | 1.0 = all Model B"
    )

    extra_kwargs = {}
    if strategy == "ties":
        trim_ratio = st.slider("Trim ratio", 0.0, 0.5, 0.2, 0.05,
            help="Fraction of smallest weight changes to zero out")
        extra_kwargs["trim_ratio"] = trim_ratio

    elif strategy == "dare":
        drop_rate = st.slider("Drop rate", 0.5, 0.99, 0.9, 0.05,
            help="Fraction of delta weights to randomly zero out")
        extra_kwargs["drop_rate"] = drop_rate

    st.divider()
    st.subheader("3. Output")

    output_path = st.text_input(
        "Save merged model to",
        value=r"D:\models\merged-phi-2"
    )

    run_btn = st.button("⚗️ Merge Models", use_container_width=True, type="primary")
    eval_btn = st.button("📊 Evaluate & Compare", use_container_width=True)


st.title("LLM Model Merger")
st.caption("SLERP · TIES · DARE — three strategies to combine two LLMs into one")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="strategy-card">
    <strong>SLERP</strong><br>
    Travels the spherical arc between two weight vectors.<br>
    Best for: smooth capability blending.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="strategy-card" style="border-color:#cc6600">
    <strong>TIES</strong><br>
    Trims noise → resolves sign conflicts → merges agreements.<br>
    Best for: models fine-tuned on different tasks.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="strategy-card" style="border-color:#006600">
    <strong>DARE</strong><br>
    Drops 90% of delta weights, rescales the rest.<br>
    Best for: reducing interference between models.
    </div>
    """, unsafe_allow_html=True)

st.divider()


if "merger" not in st.session_state:
    st.session_state.merger = None
if "merged_model" not in st.session_state:
    st.session_state.merged_model = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None


if run_btn:
    with st.spinner("Loading models..."):
        merger = LLMMerger(base_path, model_a_path, model_b_path)
        merger.load_models()
        st.session_state.merger = merger

    with st.spinner(f"Running {strategy.upper()} merge (t={t})..."):
        merged_model = merger.merge(strategy=strategy, t=t, **extra_kwargs)
        st.session_state.merged_model = merged_model

    st.success(f"✅ {strategy.upper()} merge complete!")

    total = sum(p.numel() for p in merged_model.parameters())
    st.metric("Merged model parameters", f"{total/1e9:.2f}B")

    if output_path:
        st.info("⚠️ Skipping save to preserve memory for evaluation.")


if eval_btn:
    if st.session_state.merger is None or st.session_state.merged_model is None:
        st.error("Run the merge first!")
        st.stop()

    merger       = st.session_state.merger
    merged_model = st.session_state.merged_model

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    evaluator = ModelEvaluator(tokenizer)

    models_to_compare = {
        "Base model":   merger.base_model,
        "Model A":      merger.model_a,
        "Model B":      merger.model_b,
        f"Merged ({strategy.upper()})": merged_model,
    }

    with st.spinner("Evaluating all models... this takes a few minutes..."):
        comparison = evaluator.compare_models(models_to_compare, DEFAULT_TEST_CASES)
        st.session_state.eval_results = comparison

    st.success(f"✅ Evaluation complete! Winner: **{comparison['winner']}**")


if st.session_state.eval_results:
    comp = st.session_state.eval_results
    results = comp["results"]
    winner  = comp["winner"]

    st.subheader("Evaluation Results")
    st.caption(f"Winner by BERTScore F1: **{winner}** 🏆")

    tab1, tab2, tab3 = st.tabs(["📊 Score Comparison", "📝 Generated Text", "📈 Charts"])

    with tab1:
        rows = []
        for model_name, res in results.items():
            agg = res["aggregate"]
            rows.append({
                "Model":          model_name,
                "ROUGE-1":        agg["avg_rouge1"],
                "ROUGE-2":        agg["avg_rouge2"],
                "ROUGE-L":        agg["avg_rougeL"],
                "BERTScore P":    agg["bert_precision"],
                "BERTScore R":    agg["bert_recall"],
                "BERTScore F1":   agg["bert_f1"],
                "Winner":         "🏆" if model_name == winner else ""
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab2:
        for i, tc in enumerate(DEFAULT_TEST_CASES):
            st.markdown(f"**Prompt {i+1}:** {tc['prompt']}")
            st.markdown(f"*Reference:* {tc['reference']}")

            cols = st.columns(len(results))
            for col, (model_name, res) in zip(cols, results.items()):
                pp = res["per_prompt"][i]
                with col:
                    st.markdown(f"**{model_name}**")
                    st.markdown(pp["generated"])
                    st.caption(f"ROUGE-L: {pp['rougeL']}")
            st.divider()

    with tab3:
        model_names = list(results.keys())

        metrics = ["avg_rouge1", "avg_rouge2", "avg_rougeL", "bert_f1"]
        metric_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"]

        fig = go.Figure()
        for model_name in model_names:
            agg = results[model_name]["aggregate"]
            values = [agg[m] for m in metrics]
            values.append(values[0])

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_labels + [metric_labels[0]],
                fill='toself',
                name=model_name,
                opacity=0.6
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Model Comparison — All Metrics (Radar)",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        bert_vals = [results[m]["aggregate"]["bert_f1"] for m in model_names]
        colors = ["gold" if m == winner else "#4a90d9" for m in model_names]

        fig2 = go.Figure(go.Bar(
            x=model_names,
            y=bert_vals,
            marker_color=colors,
            text=[f"{v:.4f}" for v in bert_vals],
            textposition="outside"
        ))
        fig2.update_layout(
            title="BERTScore F1 — Higher is Better",
            yaxis=dict(range=[0, 1]),
            height=380
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    if not run_btn and not eval_btn:
        st.info("👈 Set your model paths in the sidebar, then click **Merge Models**.")
