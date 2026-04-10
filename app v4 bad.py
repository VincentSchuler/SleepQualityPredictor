from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
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
    .block-container {padding-top: 1rem; padding-bottom: 1.6rem; max-width: 1420px;}
    h1, h2, h3 {color: #f8fbff; letter-spacing: -0.02em;}
    p, label, div, span {color: #d7e3f2;}

    .topbar {
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:1rem;
        margin-bottom: 0.6rem;
    }
    .page-title {
        font-size: 2.7rem;
        font-weight: 820;
        color: #f8fbff;
        margin: 0;
        line-height: 1.05;
        letter-spacing: -0.03em;
    }
    .section-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 22px;
        padding: 1rem 1rem 0.6rem 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 760;
        color: #f8fbff;
        margin: 0 0 0.75rem 0;
    }
    .factor-card {
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(255,255,255,0.07);
    }
    .factor-risk {
        background: rgba(230, 57, 70, 0.12);
        border-color: rgba(230, 57, 70, 0.28);
    }
    .factor-good {
        background: rgba(46, 204, 113, 0.12);
        border-color: rgba(46, 204, 113, 0.28);
    }
    .factor-top {
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:0.75rem;
        margin-bottom:0.2rem;
    }
    .factor-name {font-weight:700; font-size:1.02rem; color:#f8fbff;}
    .factor-val {color:#d6e0ec; font-size:0.97rem;}
    .impact-up {color:#ff7b7b; font-weight:800; letter-spacing:0.06em; font-size:1.05rem;}
    .impact-down {color:#71e3b1; font-weight:800; letter-spacing:0.06em; font-size:1.05rem;}
    .legend-row {display:flex; flex-wrap:wrap; gap:0.5rem; margin-top: -0.1rem; margin-bottom:0.3rem;}
    .legend-chip {
        padding: 0.34rem 0.7rem;
        border-radius: 999px;
        font-size: 0.86rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.045);
    }
    .small-note {color:#9fb2c8; font-size:0.94rem; margin-top: 0.1rem;}
    .stSlider, div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {margin-bottom: 0.15rem;}
    .stToggle label p, .stRadio label p {font-size: 0.98rem !important;}
    div[role="radiogroup"] {
        display:flex;
        gap:0.45rem;
        justify-content:flex-end;
    }
    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 999px;
        padding: 0.1rem 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


TEXT = {
    "FR": {
        "title": "Quelle est la qualité de votre sommeil ?",
        "lang_label": "Langue",
        "panel_title": "Tune the profile",
        "country": "Pays",
        "occupation": "Profession",
        "age": "Âge",
        "sleep_duration": "Durée de sommeil (heures)",
        "fall_asleep": "Temps d'endormissement (min)",
        "night_awakenings": "Réveils nocturnes",
        "stress": "Stress (/10)",
        "bmi": "IMC",
        "work_hours": "Heures de travail (heures)",
        "alcohol": "Verres d'alcool avant le coucher",
        "coffee": "Cafés avant le coucher",
        "nap": "Sieste",
        "screen": "Temps d'écran avant le coucher",
        "anxiety": "Symptômes anxieux",
        "depression": "Symptômes dépressifs",
        "results": "Résultat",
        "why": "Pourquoi cette prédiction ?",
        "why_caption": "Seuls les facteurs les plus marquants sont affichés.",
        "aggravating": "Facteur aggravant",
        "beneficial": "Facteur bénéfique",
        "no_factor": "Aucun facteur majeur ne ressort.",
        "yes": "Oui",
        "no": "Non",
        "quality_chips": [
            "0–25% : qualité très dégradée",
            "25–50% : qualité fragile",
            "50–75% : qualité moyenne",
            "75–100% : qualité saine",
        ],
    },
    "EN": {
        "title": "How good is your sleep quality?",
        "lang_label": "Language",
        "panel_title": "Tune the profile",
        "country": "Country",
        "occupation": "Occupation",
        "age": "Age",
        "sleep_duration": "Sleep duration (hours)",
        "fall_asleep": "Time to fall asleep (min)",
        "night_awakenings": "Night awakenings",
        "stress": "Stress (/10)",
        "bmi": "BMI",
        "work_hours": "Work hours (hours)",
        "alcohol": "Alcoholic drinks before bed",
        "coffee": "Coffees before bed",
        "nap": "Nap",
        "screen": "Screen time before sleep",
        "anxiety": "Anxiety symptoms",
        "depression": "Depressive symptoms",
        "results": "Result",
        "why": "Why this prediction?",
        "why_caption": "Only the strongest signals are shown.",
        "aggravating": "Aggravating factor",
        "beneficial": "Beneficial factor",
        "no_factor": "No major factor stands out.",
        "yes": "Yes",
        "no": "No",
        "quality_chips": [
            "0–25%: very poor quality",
            "25–50%: fragile quality",
            "50–75%: average quality",
            "75–100%: healthy quality",
        ],
    },
}

COFFEE_OPTIONS = [0, 1, 2, 3]
COFFEE_LABELS = {
    "FR": {0: "0 café", 1: "1 café", 2: "2 cafés", 3: "3 cafés ou +"},
    "EN": {0: "0 coffee", 1: "1 coffee", 2: "2 coffees", 3: "3+ coffees"},
}


@st.cache_resource
def get_predictor():
    return load_predictor()


def tr(lang: str, key: str):
    return TEXT[lang][key]


def nice_feature_name(name: str, lang: str) -> str:
    labels = {
        "sleep_duration_hrs": {"FR": "Durée de sommeil", "EN": "Sleep duration"},
        "bmi": {"FR": "IMC", "EN": "BMI"},
        "sleep_latency_mins": {"FR": "Temps d'endormissement", "EN": "Time to fall asleep"},
        "stress_score": {"FR": "Stress", "EN": "Stress"},
        "country": {"FR": "Pays", "EN": "Country"},
        "occupation": {"FR": "Profession", "EN": "Occupation"},
        "wake_episodes_per_night": {"FR": "Réveils nocturnes", "EN": "Night awakenings"},
        "age": {"FR": "Âge", "EN": "Age"},
        "work_hours_that_day": {"FR": "Heures de travail", "EN": "Work hours"},
        "Depression": {"FR": "Symptômes dépressifs", "EN": "Depressive symptoms"},
        "alcohol_units_before_bed": {"FR": "Verres d'alcool", "EN": "Alcoholic drinks"},
        "Anxiety": {"FR": "Symptômes anxieux", "EN": "Anxiety symptoms"},
        "nap_time": {"FR": "Sieste", "EN": "Nap"},
        "nb_cafe_before_bed": {"FR": "Cafés avant le coucher", "EN": "Coffees before bed"},
        "time_screen_before_sleep": {"FR": "Temps d'écran avant le coucher", "EN": "Screen time before sleep"},
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
            number={"suffix": "%", "font": {"size": 56}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "tickfont": {"size": 14},
                },
                "bar": {"color": bar_color, "thickness": 0.34},
                "steps": [
                    {"range": [0, 25], "color": "rgba(230, 57, 70, 0.28)"},
                    {"range": [25, 50], "color": "rgba(247, 127, 0, 0.22)"},
                    {"range": [50, 75], "color": "rgba(247, 201, 72, 0.18)"},
                    {"range": [75, 100], "color": "rgba(46, 204, 113, 0.22)"},
                ],
            },
        )
    )
    fig.update_layout(height=355, margin=dict(l=20, r=20, t=10, b=5), paper_bgcolor="rgba(0,0,0,0)")
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


def format_quarter_hours(value: float) -> str:
    return f"{value:.2f} h"


def format_value(feature: str, value, lang: str) -> str:
    if feature in {"sleep_duration_hrs", "work_hours_that_day"}:
        return format_quarter_hours(float(value))
    if feature == "stress_score":
        return f"{float(value):.1f}/10"
    if feature == "sleep_latency_mins":
        return f"{int(value)} min"
    if feature == "wake_episodes_per_night":
        return str(int(value))
    if feature == "age":
        return str(int(value))
    if feature == "bmi":
        return f"{float(value):.1f}"
    if feature == "alcohol_units_before_bed":
        unit = "verre" if lang == "FR" else "drink"
        unit_pl = "verres" if lang == "FR" else "drinks"
        amount = float(value)
        unit_out = unit if amount <= 1 else unit_pl
        return f"{amount:.1f} {unit_out}"
    if feature == "nb_cafe_before_bed":
        return COFFEE_LABELS[lang].get(int(value), str(value))
    if feature in {"Anxiety", "Depression"}:
        return tr(lang, "yes") if int(value) == 1 else tr(lang, "no")
    return str(value)


def build_single_input(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([payload])


def render_factor_cards(title: str, factors: list, lang: str, positive: bool) -> None:
    st.markdown(f"### {title}")
    if not factors:
        st.info(tr(lang, "no_factor"))
        return

    max_abs = max(abs(f["impact"]) for f in factors) if factors else 1.0
    shown = []
    for factor in factors:
        arrows, level = impact_arrows(float(factor["impact"]), max_abs)
        if level >= 2:
            shown.append((factor, arrows))

    if not shown:
        st.info(tr(lang, "no_factor"))
        return

    card_class = "factor-risk" if positive else "factor-good"
    for factor, arrows in shown:
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

    with st.container(border=False):
        st.markdown(f"<div class='section-title'>{tr(lang, 'panel_title')}</div>", unsafe_allow_html=True)

        top1, top2, top3 = st.columns(3)
        with top1:
            country = st.selectbox(
                tr(lang, "country"),
                country_options,
                index=country_options.index(FORM_DEFAULTS["country"]) if FORM_DEFAULTS["country"] in country_options else 0,
            )
        with top2:
            occupation = st.selectbox(
                tr(lang, "occupation"),
                occupation_options,
                index=occupation_options.index(FORM_DEFAULTS["occupation"]) if FORM_DEFAULTS["occupation"] in occupation_options else 0,
            )
        with top3:
            age = st.slider(tr(lang, "age"), 18, 90, int(FORM_DEFAULTS["age"]))

        row1 = st.columns(3)
        with row1[0]:
            sleep_duration_hrs = st.slider(
                tr(lang, "sleep_duration"), 2.0, 12.0, float(FORM_DEFAULTS["sleep_duration_hrs"]), 0.25, format="%.2f h"
            )
        with row1[1]:
            sleep_latency_mins = st.slider(tr(lang, "fall_asleep"), 0, 120, int(FORM_DEFAULTS["sleep_latency_mins"]), 5)
        with row1[2]:
            wake_episodes_per_night = st.slider(
                tr(lang, "night_awakenings"), 0, 10, int(FORM_DEFAULTS["wake_episodes_per_night"])
            )

        row2 = st.columns(3)
        with row2[0]:
            stress_score = st.slider(tr(lang, "stress"), 0.0, 10.0, float(FORM_DEFAULTS["stress_score"]), 0.5)
        with row2[1]:
            bmi = st.slider(tr(lang, "bmi"), 15.0, 45.0, float(FORM_DEFAULTS["bmi"]), 0.5)
        with row2[2]:
            work_hours_that_day = st.slider(
                tr(lang, "work_hours"), 0.0, 16.0, float(FORM_DEFAULTS["work_hours_that_day"]), 0.25, format="%.2f h"
            )

        row3 = st.columns(3)
        with row3[0]:
            alcohol_units_before_bed = st.slider(
                tr(lang, "alcohol"), 0.0, 8.0, float(FORM_DEFAULTS["alcohol_units_before_bed"]), 0.5
            )
        with row3[1]:
            nb_cafe_before_bed = st.select_slider(
                tr(lang, "coffee"),
                options=COFFEE_OPTIONS,
                value=int(FORM_DEFAULTS["nb_cafe_before_bed"]),
                format_func=lambda x: COFFEE_LABELS[lang][x],
            )
        with row3[2]:
            nap_time = st.selectbox(tr(lang, "nap"), NAP_OPTIONS, index=NAP_OPTIONS.index(FORM_DEFAULTS["nap_time"]))

        row4 = st.columns(3)
        with row4[0]:
            time_screen_before_sleep = st.selectbox(
                tr(lang, "screen"),
                SCREEN_TIME_OPTIONS,
                index=SCREEN_TIME_OPTIONS.index(FORM_DEFAULTS["time_screen_before_sleep"]),
            )
        with row4[1]:
            anxiety = 1 if st.toggle(tr(lang, "anxiety"), value=bool(FORM_DEFAULTS["Anxiety"])) else 0
        with row4[2]:
            depression = 1 if st.toggle(tr(lang, "depression"), value=bool(FORM_DEFAULTS["Depression"])) else 0

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
    top_left, top_right = st.columns([0.82, 0.18])
    with top_left:
        st.markdown("<div class='page-title'>Quelle est la qualité de votre sommeil ?</div>", unsafe_allow_html=True)
    with top_right:
        lang = st.radio(
            "Langue / Language",
            options=["FR", "EN"],
            horizontal=True,
            label_visibility="collapsed",
            index=0,
        )

    if lang == "EN":
        st.markdown("<script>document.title = 'Sleep Quality';</script>", unsafe_allow_html=True)

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Place your exported model in models/lgbm_full.joblib, or run train.py to generate it.")
        return

    metadata = predictor.metadata or {}
    category_options = metadata.get("category_options", {})

    left, right = st.columns([1.28, 0.72], gap="large")

    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        payload = input_panel(category_options, lang)
        st.markdown("</div>", unsafe_allow_html=True)

    result = predictor.predict_dataframe(pd.DataFrame([payload]))
    quality = score_to_quality(result.raw_score)
    _ = build_single_input(payload)

    with right:
        st.markdown(f"<div class='section-card'><div class='section-title'>{tr(lang, 'results')}</div>", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(result.raw_score), use_container_width=True)
        st.markdown("<div class='legend-row'>", unsafe_allow_html=True)
        chip_colors = [
            "rgba(230, 57, 70, 0.14)",
            "rgba(247, 127, 0, 0.14)",
            "rgba(247, 201, 72, 0.12)",
            "rgba(46, 204, 113, 0.14)",
        ]
        chips = []
        for text, color in zip(tr(lang, "quality_chips"), chip_colors):
            chips.append(f"<span class='legend-chip' style='background:{color}'>{text}</span>")
        st.markdown("".join(chips), unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-card'><div class='section-title'>{tr(lang, 'why')}</div><div class='small-note'>{tr(lang, 'why_caption')}</div>", unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        render_factor_cards(tr(lang, "aggravating"), result.top_positive_factors, lang, positive=True)
    with exp_col2:
        render_factor_cards(tr(lang, "beneficial"), result.top_negative_factors, lang, positive=False)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
