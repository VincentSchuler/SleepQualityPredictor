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

st.set_page_config(
    page_title="Sleep Quality",
    page_icon="💤",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #07101d 0%, #0b1524 100%);}
    .block-container {padding-top: 1.0rem; padding-bottom: 1.6rem; max-width: 1420px;}
    h1, h2, h3 {color: #f8fbff; letter-spacing: -0.02em;}
    p, label, div, span {color: #d7e3f2;}
    .hero {
        background: linear-gradient(135deg, rgba(76, 201, 240, 0.16), rgba(114, 9, 183, 0.14));
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.05rem 1.2rem;
        border-radius: 24px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .section-title {
        font-size: 1.65rem;
        font-weight: 760;
        margin: 0.1rem 0 0.65rem 0;
        color: #f8fbff;
    }
    .subtle {
        color: #9fb2c8;
        font-size: 0.96rem;
        margin-top: -0.2rem;
        margin-bottom: 0.7rem;
    }
    .factor-card {
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .factor-risk {
        background: rgba(230, 57, 70, 0.10);
        border-color: rgba(230, 57, 70, 0.22);
    }
    .factor-good {
        background: rgba(46, 204, 113, 0.10);
        border-color: rgba(46, 204, 113, 0.22);
    }
    .factor-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.35rem;
    }
    .factor-name {
        font-weight: 700;
        font-size: 1rem;
        color: #f8fbff;
    }
    .factor-val {
        color: #d6e0ec;
        font-size: 0.98rem;
    }
    .impact-up {color: #ff7b7b; font-weight: 800; letter-spacing: 0.05em; font-size: 1.05rem;}
    .impact-down {color: #71e3b1; font-weight: 800; letter-spacing: 0.05em; font-size: 1.05rem;}
    .lang-wrap {
        display: flex;
        justify-content: flex-end;
        align-items: center;
    }
    .stSlider, div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        margin-bottom: 0.1rem;
    }
    .st-emotion-cache-1r6slb0, .st-emotion-cache-13k62yr {
        border-radius: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


TEXT = {
    "EN": {
        "title": "How good is your sleep quality?",
        "language": "Language",
        "profile_title": "Tune the profile",
        "country": "Country",
        "occupation": "Occupation",
        "age": "Age",
        "sleep_duration": "Sleep duration",
        "fall_asleep": "Minutes to fall asleep",
        "night_awakenings": "Night awakenings",
        "stress": "Stress score",
        "bmi": "BMI",
        "work_hours": "Work hours today",
        "alcohol": "Alcohol before bed",
        "caffeine": "Caffeine before bed",
        "nap": "Nap duration",
        "screen": "Screen time before sleep",
        "anxiety": "Anxiety symptoms",
        "depression": "Depressive symptoms",
        "results": "Results",
        "why": "Why this prediction?",
        "why_caption": "Only the strongest signals are shown.",
        "risk_up": "Raises risk",
        "risk_down": "Improves sleep",
        "no_factor": "No strong factor stands out.",
        "model_missing": "Expected next step: place your exported model in models/lgbm_full.joblib, or run train.py to generate it.",
        "yes": "Yes",
        "no": "No",
    },
    "FR": {
        "title": "Quelle est la qualité de votre sommeil ?",
        "language": "Langue",
        "profile_title": "Ajustez le profil",
        "country": "Pays",
        "occupation": "Profession",
        "age": "Âge",
        "sleep_duration": "Durée de sommeil",
        "fall_asleep": "Minutes pour s'endormir",
        "night_awakenings": "Réveils nocturnes",
        "stress": "Score de stress",
        "bmi": "IMC",
        "work_hours": "Heures de travail aujourd'hui",
        "alcohol": "Alcool avant le coucher",
        "caffeine": "Caféine avant le coucher",
        "nap": "Sieste",
        "screen": "Temps d'écran avant le coucher",
        "anxiety": "Symptômes anxieux",
        "depression": "Symptômes dépressifs",
        "results": "Résultats",
        "why": "Pourquoi cette prédiction ?",
        "why_caption": "Seuls les facteurs les plus marquants sont affichés.",
        "risk_up": "Augmente le risque",
        "risk_down": "Améliore le sommeil",
        "no_factor": "Aucun facteur marquant ne ressort.",
        "model_missing": "Étape attendue : place ton modèle exporté dans models/lgbm_full.joblib, ou lance train.py pour le générer.",
        "yes": "Oui",
        "no": "Non",
    },
}


@st.cache_resource
def get_predictor():
    return load_predictor()


def t(lang: str, key: str) -> str:
    return TEXT[lang][key]


def nice_feature_name(name: str, lang: str) -> str:
    labels = {
        "sleep_duration_hrs": {"EN": "Sleep duration", "FR": "Durée de sommeil"},
        "bmi": {"EN": "BMI", "FR": "IMC"},
        "sleep_latency_mins": {"EN": "Time to fall asleep", "FR": "Temps pour s'endormir"},
        "stress_score": {"EN": "Stress score", "FR": "Score de stress"},
        "country": {"EN": "Country", "FR": "Pays"},
        "occupation": {"EN": "Occupation", "FR": "Profession"},
        "wake_episodes_per_night": {"EN": "Night awakenings", "FR": "Réveils nocturnes"},
        "age": {"EN": "Age", "FR": "Âge"},
        "work_hours_that_day": {"EN": "Work hours today", "FR": "Heures de travail aujourd'hui"},
        "Depression": {"EN": "Depressive symptoms", "FR": "Symptômes dépressifs"},
        "alcohol_units_before_bed": {"EN": "Alcohol before bed", "FR": "Alcool avant le coucher"},
        "Anxiety": {"EN": "Anxiety symptoms", "FR": "Symptômes anxieux"},
        "nap_time": {"EN": "Nap duration", "FR": "Sieste"},
        "nb_cafe_before_bed": {"EN": "Caffeine before bed", "FR": "Caféine avant le coucher"},
        "time_screen_before_sleep": {"EN": "Screen time before sleep", "FR": "Temps d'écran avant le coucher"},
    }
    return labels.get(name, {}).get(lang, name)


def score_to_quality(raw_score: float) -> float:
    score = max(1.0, min(4.0, raw_score))
    return (4.0 - score) / 3.0 * 100.0


def color_mix_hex(color_a: str, color_b: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    a = tuple(int(color_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(color_b[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round((1 - ratio) * x + ratio * y) for x, y in zip(a, b))
    return "#%02x%02x%02x" % mixed


def make_gauge(raw_score: float) -> go.Figure:
    quality = score_to_quality(raw_score)
    bar_color = color_mix_hex("#e63946", "#2ecc71", quality / 100.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=quality,
            number={"suffix": "%", "font": {"size": 52}},
            gauge={
                "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "tickfont": {"size": 14}},
                "bar": {"color": bar_color, "thickness": 0.34},
                "steps": [
                    {"range": [0, 25], "color": "rgba(230, 57, 70, 0.22)"},
                    {"range": [25, 50], "color": "rgba(247, 127, 0, 0.22)"},
                    {"range": [50, 75], "color": "rgba(247, 201, 72, 0.18)"},
                    {"range": [75, 100], "color": "rgba(46, 204, 113, 0.20)"},
                ],
            },
        )
    )
    fig.update_layout(height=355, margin=dict(l=20, r=20, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def impact_level(impact: float, max_abs: float) -> int:
    if max_abs <= 1e-9:
        return 0
    ratio = abs(impact) / max_abs
    if ratio >= 0.78:
        return 4
    if ratio >= 0.52:
        return 3
    if ratio >= 0.26:
        return 2
    return 1


def impact_arrows(impact: float, max_abs: float) -> tuple[str, int]:
    level = impact_level(impact, max_abs)
    arrow = "↑" if impact > 0 else "↓"
    css = "impact-up" if impact > 0 else "impact-down"
    return f"<span class='{css}'>{arrow * max(level, 1)}</span>", level


def format_time_hours(hours: float) -> str:
    total_minutes = int(round(hours * 60))
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h}h{m:02d}"


def format_value(feature: str, value, lang: str) -> str:
    if feature in {"sleep_duration_hrs", "work_hours_that_day"}:
        return format_time_hours(float(value))
    if feature in {"Anxiety", "Depression"}:
        return t(lang, "yes") if int(value) == 1 else t(lang, "no")
    if feature == "alcohol_units_before_bed":
        return f"{float(value):.1f}"
    if feature == "bmi":
        return f"{float(value):.1f}"
    if feature == "stress_score":
        return f"{float(value):.1f}/10"
    if feature == "age":
        return str(int(value))
    return str(value)


def render_factor_cards(title: str, factors: list, lang: str, positive: bool) -> None:
    st.markdown(f"### {title}")
    if not factors:
        st.info(t(lang, "no_factor"))
        return

    max_abs = max(abs(f["impact"]) for f in factors) if factors else 1.0
    displayed = []
    for factor in factors:
        arrows, level = impact_arrows(float(factor["impact"]), max_abs)
        if level < 2:
            continue
        displayed.append((factor, arrows))

    if not displayed:
        st.info(t(lang, "no_factor"))
        return

    card_class = "factor-risk" if positive else "factor-good"
    for factor, arrows in displayed:
        feat = nice_feature_name(factor["feature"], lang)
        val = format_value(factor["feature"], factor["value"], lang)
        st.markdown(
            f"""
            <div class='factor-card {card_class}'>
                <div class='factor-top'>
                    <div class='factor-name'>{feat}</div>
                    <div>{arrows}</div>
                </div>
                <div class='factor-val'>{val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def input_panel(options: dict, lang: str) -> dict:
    country_options = options.get("country") or DEFAULT_COUNTRIES
    occupation_options = options.get("occupation") or DEFAULT_OCCUPATIONS

    st.markdown(f"<div class='section-title'>{t(lang, 'profile_title')}</div>", unsafe_allow_html=True)

    top1, top2 = st.columns(2)
    with top1:
        country = st.selectbox(
            t(lang, "country"),
            country_options,
            index=country_options.index(FORM_DEFAULTS["country"]) if FORM_DEFAULTS["country"] in country_options else 0,
        )
    with top2:
        occupation = st.selectbox(
            t(lang, "occupation"),
            occupation_options,
            index=occupation_options.index(FORM_DEFAULTS["occupation"]) if FORM_DEFAULTS["occupation"] in occupation_options else 0,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider(t(lang, "age"), 18, 90, int(FORM_DEFAULTS["age"]))
        sleep_minutes = st.slider(
            t(lang, "sleep_duration"),
            min_value=120,
            max_value=720,
            value=int(round(float(FORM_DEFAULTS["sleep_duration_hrs"]) * 60)),
            step=5,
        )
        st.caption(format_time_hours(sleep_minutes / 60))
        stress_score = st.slider(t(lang, "stress"), 0.0, 10.0, float(FORM_DEFAULTS["stress_score"]), 0.1)

    with c2:
        sleep_latency_mins = st.slider(t(lang, "fall_asleep"), 0, 120, int(FORM_DEFAULTS["sleep_latency_mins"]))
        wake_episodes_per_night = st.slider(t(lang, "night_awakenings"), 0, 10, int(FORM_DEFAULTS["wake_episodes_per_night"]))
        bmi = st.slider(t(lang, "bmi"), 15.0, 45.0, float(FORM_DEFAULTS["bmi"]), 0.1)

    with c3:
        work_minutes = st.slider(
            t(lang, "work_hours"),
            min_value=0,
            max_value=960,
            value=int(round(float(FORM_DEFAULTS["work_hours_that_day"]) * 60)),
            step=5,
        )
        st.caption(format_time_hours(work_minutes / 60))
        alcohol_units_before_bed = st.slider(t(lang, "alcohol"), 0.0, 8.0, float(FORM_DEFAULTS["alcohol_units_before_bed"]), 0.5)

    b1, b2, b3 = st.columns(3)
    with b1:
        nb_cafe_before_bed = st.select_slider(
            t(lang, "caffeine"),
            options=list(CAFFEINE_BUCKET_LABELS.keys()),
            value=FORM_DEFAULTS["nb_cafe_before_bed"],
            format_func=lambda x: CAFFEINE_BUCKET_LABELS[x],
        )
    with b2:
        nap_time = st.selectbox(t(lang, "nap"), NAP_OPTIONS, index=NAP_OPTIONS.index(FORM_DEFAULTS["nap_time"]))
    with b3:
        time_screen_before_sleep = st.selectbox(
            t(lang, "screen"),
            SCREEN_TIME_OPTIONS,
            index=SCREEN_TIME_OPTIONS.index(FORM_DEFAULTS["time_screen_before_sleep"]),
        )

    d1, d2 = st.columns(2)
    with d1:
        anxiety = 1 if st.toggle(t(lang, "anxiety"), value=bool(FORM_DEFAULTS["Anxiety"])) else 0
    with d2:
        depression = 1 if st.toggle(t(lang, "depression"), value=bool(FORM_DEFAULTS["Depression"])) else 0

    return {
        "age": age,
        "sleep_duration_hrs": sleep_minutes / 60,
        "sleep_latency_mins": sleep_latency_mins,
        "wake_episodes_per_night": wake_episodes_per_night,
        "stress_score": stress_score,
        "bmi": bmi,
        "work_hours_that_day": work_minutes / 60,
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
    lang = st.radio(TEXT["EN"]["language"], ["EN", "FR"], horizontal=True)
    st.markdown(f"<div class='hero'><h1>💤 {t(lang, 'title')}</h1></div>", unsafe_allow_html=True)

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(t(lang, "model_missing"))
        return

    metadata = predictor.metadata or {}
    category_options = metadata.get("category_options", {})

    left, right = st.columns([1.08, 0.92], gap="large")

    with left:
        payload = input_panel(category_options, lang)

    result = predictor.predict_dataframe(pd.DataFrame([payload]))

    with right:
        st.markdown(f"<div class='section-title'>{t(lang, 'results')}</div>", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(result.raw_score), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div class='section-title' style='margin-top:0.6rem'>{t(lang, 'why')}</div>", unsafe_allow_html=True)
        st.caption(t(lang, "why_caption"))
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            render_factor_cards(t(lang, "risk_up"), result.top_positive_factors, lang, positive=True)
        with exp_col2:
            render_factor_cards(t(lang, "risk_down"), result.top_negative_factors, lang, positive=False)


if __name__ == "__main__":
    main()
