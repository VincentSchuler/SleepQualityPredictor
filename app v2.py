from __future__ import annotations

import math

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
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #07101d 0%, #0b1524 100%);}
    .block-container {padding-top: 1.1rem; padding-bottom: 1.8rem; max-width: 1380px;}
    h1, h2, h3 {color: #f8fbff; letter-spacing: -0.02em;}
    p, label, div, span {color: #d7e3f2;}
    [data-testid="stMetricValue"] {font-size: 2.25rem;}
    [data-testid="stMetricLabel"] {font-size: 1rem;}
    .hero {
        background: linear-gradient(135deg, rgba(76, 201, 240, 0.18), rgba(114, 9, 183, 0.18));
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.25rem 1.35rem;
        border-radius: 24px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .section-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1rem 1rem 0.7rem 1rem;
        height: 100%;
        backdrop-filter: blur(6px);
    }
    .mini-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.75rem;
    }
    .input-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
        color: #f8fbff;
    }
    .muted {
        color: #9fb2c8;
        font-size: 0.98rem;
        margin-bottom: 0.85rem;
    }
    .factor-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
    }
    .factor-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.35rem;
    }
    .factor-name {
        font-weight: 650;
        font-size: 1rem;
        color: #f8fbff;
    }
    .factor-val {
        color: #b8c9dc;
        font-size: 0.98rem;
    }
    .impact-up {color: #ff7b7b; font-weight: 700; letter-spacing: 0.05em;}
    .impact-down {color: #71e3b1; font-weight: 700; letter-spacing: 0.05em;}
    .band-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.92rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.04);
    }
    .legend-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.1rem;
        margin-bottom: 0.4rem;
    }
    .legend-chip {
        padding: 0.28rem 0.6rem;
        border-radius: 999px;
        font-size: 0.86rem;
        color: white;
        font-weight: 600;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stSlider {
        margin-bottom: 0.25rem;
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
        "Depression": "Depression",
        "alcohol_units_before_bed": "Alcohol before bed",
        "Anxiety": "Anxiety",
        "nap_time": "Nap duration",
        "nb_cafe_before_bed": "Caffeine before bed",
        "time_screen_before_sleep": "Screen time before sleep",
    }
    return labels.get(name, name)


def score_to_quality(raw_score: float) -> float:
    score = max(1.0, min(4.0, raw_score))
    return (4.0 - score) / 3.0 * 100.0


def quality_label(quality: float) -> str:
    if quality >= 75:
        return "Excellent sleep outlook"
    if quality >= 50:
        return "Fairly solid"
    if quality >= 25:
        return "Fragile sleep outlook"
    return "Needs attention"


def make_gauge(raw_score: float) -> go.Figure:
    quality = score_to_quality(raw_score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=quality,
            number={"suffix": "%", "font": {"size": 46}},
            title={"text": "Estimated sleep quality"},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100]},
                "bar": {"color": "#7bdff2", "thickness": 0.28},
                "steps": [
                    {"range": [0, 25], "color": "rgba(230, 57, 70, 0.24)"},
                    {"range": [25, 50], "color": "rgba(247, 127, 0, 0.24)"},
                    {"range": [50, 75], "color": "rgba(114, 9, 183, 0.22)"},
                    {"range": [75, 100], "color": "rgba(76, 201, 240, 0.22)"},
                ],
            },
        )
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=65, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def band_chip(label: str) -> str:
    styles = {
        "Healthy": "background: rgba(76, 201, 240, 0.15); color: #9be7ff;",
        "Mild risk": "background: rgba(140, 82, 255, 0.16); color: #d5c3ff;",
        "Moderate risk": "background: rgba(247, 127, 0, 0.16); color: #ffc38a;",
        "Severe risk": "background: rgba(230, 57, 70, 0.16); color: #ff9aa3;",
    }
    return f"<span class='band-chip' style='{styles.get(label, '')}'>{label}</span>"


def impact_arrows(impact: float, max_abs: float) -> str:
    if max_abs <= 1e-9:
        level = 1
    else:
        ratio = abs(impact) / max_abs
        if ratio >= 0.78:
            level = 4
        elif ratio >= 0.52:
            level = 3
        elif ratio >= 0.26:
            level = 2
        else:
            level = 1
    arrow = "↑" if impact > 0 else "↓"
    css = "impact-up" if impact > 0 else "impact-down"
    return f"<span class='{css}'>{arrow * level}</span>"


def render_factor_cards(title: str, factors: list) -> None:
    st.markdown(f"### {title}")
    if not factors:
        st.info("No strong factor identified.")
        return

    max_abs = max(abs(f["impact"]) for f in factors) if factors else 1.0
    for factor in factors:
        feat = nice_feature_name(factor["feature"])
        val = factor["value"]
        arrows = impact_arrows(float(factor["impact"]), max_abs)
        st.markdown(
            f"""
            <div class='factor-card'>
                <div class='factor-top'>
                    <div class='factor-name'>{feat}</div>
                    <div>{arrows}</div>
                </div>
                <div class='factor-val'>{val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def input_panel(options: dict) -> dict:
    country_options = options.get("country") or DEFAULT_COUNTRIES
    occupation_options = options.get("occupation") or DEFAULT_OCCUPATIONS

    st.markdown(
        """
        <div class='section-card'>
            <div class='input-title'>Tune the profile</div>
            <div class='muted'>Everything updates instantly. Adjust lifestyle, habits and context without scrolling through a long sidebar.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='legend-row'>
            <span class='legend-chip' style='background: rgba(76, 201, 240, 0.75);'>75–100% strong</span>
            <span class='legend-chip' style='background: rgba(114, 9, 183, 0.75);'>50–75% decent</span>
            <span class='legend-chip' style='background: rgba(247, 127, 0, 0.78);'>25–50% fragile</span>
            <span class='legend-chip' style='background: rgba(230, 57, 70, 0.78);'>0–25% poor</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Daily profile")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 90, int(FORM_DEFAULTS["age"]))
        sleep_duration_hrs = st.slider("Sleep duration (hours)", 2.0, 12.0, float(FORM_DEFAULTS["sleep_duration_hrs"]), 0.1)
        sleep_latency_mins = st.slider("Minutes to fall asleep", 0, 120, int(FORM_DEFAULTS["sleep_latency_mins"]))
        wake_episodes_per_night = st.slider("Night awakenings", 0, 10, int(FORM_DEFAULTS["wake_episodes_per_night"]))
    with c2:
        stress_score = st.slider("Stress score", 0.0, 10.0, float(FORM_DEFAULTS["stress_score"]), 0.1)
        bmi = st.slider("BMI", 15.0, 45.0, float(FORM_DEFAULTS["bmi"]), 0.1)
        work_hours_that_day = st.slider("Work hours today", 0.0, 16.0, float(FORM_DEFAULTS["work_hours_that_day"]), 0.1)
        alcohol_units_before_bed = st.slider("Alcohol before bed", 0.0, 8.0, float(FORM_DEFAULTS["alcohol_units_before_bed"]), 0.5)

    st.markdown("### Habits & context")
    c3, c4 = st.columns(2)
    with c3:
        nb_cafe_before_bed = st.select_slider(
            "Caffeine before bed",
            options=list(CAFFEINE_BUCKET_LABELS.keys()),
            value=FORM_DEFAULTS["nb_cafe_before_bed"],
            format_func=lambda x: CAFFEINE_BUCKET_LABELS[x],
        )
        nap_time = st.selectbox("Nap duration", NAP_OPTIONS, index=NAP_OPTIONS.index(FORM_DEFAULTS["nap_time"]))
        time_screen_before_sleep = st.selectbox(
            "Screen time before sleep",
            SCREEN_TIME_OPTIONS,
            index=SCREEN_TIME_OPTIONS.index(FORM_DEFAULTS["time_screen_before_sleep"]),
        )
        country = st.selectbox(
            "Country",
            country_options,
            index=country_options.index(FORM_DEFAULTS["country"]) if FORM_DEFAULTS["country"] in country_options else 0,
        )
    with c4:
        occupation = st.selectbox(
            "Occupation",
            occupation_options,
            index=occupation_options.index(FORM_DEFAULTS["occupation"]) if FORM_DEFAULTS["occupation"] in occupation_options else 0,
        )
        anxiety = 1 if st.toggle("Anxiety symptoms", value=bool(FORM_DEFAULTS["Anxiety"])) else 0
        depression = 1 if st.toggle("Depressive symptoms", value=bool(FORM_DEFAULTS["Depression"])) else 0

    return {
        "age": age,
        "sleep_duration_hrs": sleep_duration_hrs,
        "sleep_latency_mins": sleep_latency_mins,
        "wake_episodes_per_night": wake_episodes_per_night,
        "stress_score": stress_score,
        "bmi": bmi,
        "work_hours_that_day": work_hours_that_day,
        "alcohol_units_before_bed": alcohol_units_before_bed,
        "nb_cafe_before_bed": nb_cafe_before_bed,
        "country": country,
        "occupation": occupation,
        "nap_time": nap_time,
        "time_screen_before_sleep": time_screen_before_sleep,
        "Anxiety": anxiety,
        "Depression": depression,
    }


def main() -> None:
    st.markdown(
        """
        <div class='hero'>
            <h1>💤 Sleep Risk Studio</h1>
            <p>Interactive sleep-quality estimator based on your notebook’s final LightGBM pipeline.</p>
            <p>Move the profile, watch the quality score shift instantly, and see which variables matter most.</p>
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

    left, right = st.columns([1.12, 0.88], gap="large")

    with left:
        payload = input_panel(category_options)

    result = predictor.predict_dataframe(pd.DataFrame([payload]))
    quality = score_to_quality(result.raw_score)
    X = build_single_input(payload)

    with right:
        st.markdown("## Sleep quality")
        a, b, c = st.columns(3)
        a.metric("Quality score", f"{quality:.0f}%")
        b.metric("Risk class", result.risk_class)
        c.markdown("**Risk band**")
        c.markdown(band_chip(result.risk_label), unsafe_allow_html=True)
        st.caption(quality_label(quality))
        st.plotly_chart(make_gauge(result.raw_score), use_container_width=True)

        st.markdown("## Why this prediction?")
        st.caption("More arrows = stronger impact on the predicted score.")
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            render_factor_cards("Factors increasing risk", result.top_positive_factors)
        with exp_col2:
            render_factor_cards("Factors reducing risk", result.top_negative_factors)

    with st.expander("Production notes"):
        st.write(
            "- Model logic reused from the notebook: LightGBM regressor trained on ordinal target 1→4.\n"
            "- The dial is now expressed as sleep quality from 0 to 100%, mapped directly from the notebook’s 1→4 score.\n"
            "- Feature engineering reused: nap_time, screen-time bucket, caffeine bucket, Anxiety/Depression flags.\n"
            "- SHAP is used when possible. If no SHAP explainer is available, the app falls back to a heuristic explanation based on training metadata."
        )


if __name__ == "__main__":
    main()
