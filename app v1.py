from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    CAFFEINE_BUCKET_LABELS,
    DEFAULT_COUNTRIES,
    DEFAULT_OCCUPATIONS,
    FORM_DEFAULTS,
    NAP_OPTIONS,
    SCREEN_TIME_OPTIONS,
)
from src.predict import load_predictor
from src.preprocess import build_single_input

st.set_page_config(
    page_title="Sleep Risk Studio",
    page_icon="💤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #081120 0%, #0d1b2a 100%);}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1280px;}
    h1, h2, h3 {color: #f8fbff;}
    p, label, div, span {color: #d7e3f2;}
    [data-testid="stMetricValue"] {font-size: 2rem;}
    .hero {
        background: linear-gradient(135deg, rgba(76, 201, 240, 0.18), rgba(114, 9, 183, 0.18));
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.3rem 1.4rem;
        border-radius: 22px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.7rem 1rem;
        height: 100%;
    }
    .pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor():
    return load_predictor()


def nice_feature_name(name: str) -> str:
    labels = {
        "sleep_duration_hrs": "Sleep duration",
        "bmi": "BMI",
        "sleep_latency_mins": "Time to fall asleep",
        "stress_score": "Stress score",
        "country": "Country",
        "occupation": "Occupation",
        "wake_episodes_per_night": "Night awakenings",
        "age": "Age",
        "work_hours_that_day": "Work hours today",
        "Depression": "Depression flag",
        "alcohol_units_before_bed": "Alcohol before bed",
        "Anxiety": "Anxiety flag",
        "nap_time": "Nap duration",
        "nb_cafe_before_bed": "Caffeine bucket",
        "time_screen_before_sleep": "Screen time before sleep",
    }
    return labels.get(name, name)


def make_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(1, min(4, score)),
            number={"suffix": " / 4"},
            title={"text": "Estimated sleep risk score"},
            gauge={
                "axis": {"range": [1, 4], "tickvals": [1, 2, 3, 4]},
                "bar": {"color": "#4cc9f0"},
                "steps": [
                    {"range": [1, 1.75], "color": "rgba(76, 201, 240, 0.20)"},
                    {"range": [1.75, 2.5], "color": "rgba(114, 9, 183, 0.20)"},
                    {"range": [2.5, 3.25], "color": "rgba(247, 127, 0, 0.20)"},
                    {"range": [3.25, 4], "color": "rgba(230, 57, 70, 0.23)"},
                ],
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def make_band_chart(probabilities_like: dict) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=list(probabilities_like.keys()),
            y=[100 * v for v in probabilities_like.values()],
            text=[f"{100*v:.1f}%" for v in probabilities_like.values()],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Closeness to each risk band",
        yaxis_title="Relative confidence (%)",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_factor_cards(title: str, factors: list, positive: bool = True) -> None:
    st.markdown(f"### {title}")
    if not factors:
        st.info("No strong factor identified.")
        return
    for factor in factors:
        feat = nice_feature_name(factor["feature"])
        val = factor["value"]
        impact = factor["impact"]
        emoji = "🔺" if positive else "🟢"
        action = "pushes the risk up" if positive else "pulls the risk down"
        st.markdown(
            f"<div class='card'><b>{emoji} {feat}</b><br/>"
            f"Current value: <span class='pill'>{val}</span>"
            f"Estimated effect: <b>{action}</b> ({impact:+.3f})"
            f"</div>",
            unsafe_allow_html=True,
        )


def sidebar_form(options: dict) -> dict:
    st.sidebar.markdown("## Tune the profile")
    st.sidebar.caption("Adjust the inputs and the prediction updates instantly.")

    country_options = options.get("country") or DEFAULT_COUNTRIES
    occupation_options = options.get("occupation") or DEFAULT_OCCUPATIONS

    payload = {
        "age": st.sidebar.slider("Age", 18, 90, int(FORM_DEFAULTS["age"])),
        "sleep_duration_hrs": st.sidebar.slider("Sleep duration (hours)", 2.0, 12.0, float(FORM_DEFAULTS["sleep_duration_hrs"]), 0.1),
        "sleep_latency_mins": st.sidebar.slider("Minutes to fall asleep", 0, 120, int(FORM_DEFAULTS["sleep_latency_mins"])),
        "wake_episodes_per_night": st.sidebar.slider("Night awakenings", 0, 10, int(FORM_DEFAULTS["wake_episodes_per_night"])),
        "stress_score": st.sidebar.slider("Stress score", 0.0, 10.0, float(FORM_DEFAULTS["stress_score"]), 0.1),
        "bmi": st.sidebar.slider("BMI", 15.0, 45.0, float(FORM_DEFAULTS["bmi"]), 0.1),
        "work_hours_that_day": st.sidebar.slider("Work hours today", 0.0, 16.0, float(FORM_DEFAULTS["work_hours_that_day"]), 0.1),
        "alcohol_units_before_bed": st.sidebar.slider("Alcohol units before bed", 0.0, 8.0, float(FORM_DEFAULTS["alcohol_units_before_bed"]), 0.5),
        "nb_cafe_before_bed": st.sidebar.select_slider(
            "Caffeine before bed",
            options=list(CAFFEINE_BUCKET_LABELS.keys()),
            value=FORM_DEFAULTS["nb_cafe_before_bed"],
            format_func=lambda x: CAFFEINE_BUCKET_LABELS[x],
        ),
        "country": st.sidebar.selectbox("Country", country_options, index=country_options.index(FORM_DEFAULTS["country"]) if FORM_DEFAULTS["country"] in country_options else 0),
        "occupation": st.sidebar.selectbox("Occupation", occupation_options, index=occupation_options.index(FORM_DEFAULTS["occupation"]) if FORM_DEFAULTS["occupation"] in occupation_options else 0),
        "nap_time": st.sidebar.selectbox("Nap duration", NAP_OPTIONS, index=NAP_OPTIONS.index(FORM_DEFAULTS["nap_time"])),
        "time_screen_before_sleep": st.sidebar.selectbox("Screen time before sleep", SCREEN_TIME_OPTIONS, index=SCREEN_TIME_OPTIONS.index(FORM_DEFAULTS["time_screen_before_sleep"])),
        "Anxiety": 1 if st.sidebar.toggle("Anxiety symptoms", value=bool(FORM_DEFAULTS["Anxiety"])) else 0,
        "Depression": 1 if st.sidebar.toggle("Depressive symptoms", value=bool(FORM_DEFAULTS["Depression"])) else 0,
    }
    return payload


def main() -> None:
    st.markdown(
        """
        <div class='hero'>
            <h1>💤 Sleep Risk Studio</h1>
            <p>Interactive sleep-risk prediction app based on your notebook’s final LightGBM pipeline.</p>
            <p>Change the profile, watch the score move instantly, and understand <b>what increases</b> or <b>reduces</b> the estimated risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Expected next step: put your exported model in models/lgbm_full.joblib, or run train.py to generate it.")
        return

    metadata = predictor.metadata or {}
    category_options = metadata.get("category_options", {})
    payload = sidebar_form(category_options)
    X = build_single_input(payload)
    result = predictor.predict_dataframe(pd.DataFrame([payload]))

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("## Predicted profile")
        a, b, c = st.columns(3)
        a.metric("Risk band", result.risk_label)
        b.metric("Rounded class", result.risk_class)
        b.caption("1 = healthy • 4 = severe")
        c.metric("Raw model score", f"{result.raw_score:.2f}")
        st.plotly_chart(make_gauge(result.raw_score), use_container_width=True)

    with right:
        st.markdown("## Quick interpretation")
        st.markdown(f"<div class='card'><h3>{result.risk_description}</h3><p>This model outputs a continuous score and the app rounds it to the nearest risk class, exactly like the notebook.</p></div>", unsafe_allow_html=True)
        st.plotly_chart(make_band_chart(result.probabilities_like), use_container_width=True)

    st.markdown("## Why this prediction?")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        render_factor_cards("Main factors increasing risk", result.top_positive_factors, positive=True)
    with exp_col2:
        render_factor_cards("Main factors reducing risk", result.top_negative_factors, positive=False)

    st.markdown("## Current input snapshot")
    st.dataframe(X, use_container_width=True)

    with st.expander("Production notes"):
        st.write(
            "- Model logic reused from the notebook: LightGBM regressor trained on ordinal target 1→4.\n"
            "- Feature engineering reused: nap_time, screen-time bucket, caffeine bucket, Anxiety/Depression flags.\n"
            "- SHAP is used when possible. If no SHAP explainer is available, the app falls back to a heuristic explanation based on training metadata."
        )


if __name__ == "__main__":
    main()
