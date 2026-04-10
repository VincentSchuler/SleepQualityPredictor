from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import DEFAULT_COUNTRIES, DEFAULT_OCCUPATIONS, FORM_DEFAULTS, NAP_OPTIONS, SCREEN_TIME_OPTIONS
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
    .block-container {padding-top: 0.9rem; padding-bottom: 1.4rem; max-width: 1450px;}
    h1, h2, h3 {color: #f8fbff; letter-spacing: -0.02em;}
    p, label, div, span {color: #d7e3f2;}

    .hero {
        background: linear-gradient(135deg, rgba(76, 201, 240, 0.16), rgba(114, 9, 183, 0.14));
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.05rem 1.2rem;
        border-radius: 24px;
        margin-bottom: 0.95rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 1rem;
        align-items: center;
    }
    .page-title {
        font-size: clamp(2.15rem, 3.4vw, 4rem);
        font-weight: 840;
        color: #f8fbff;
        margin: 0;
        line-height: 1.06;
        letter-spacing: -0.045em;
    }

    .header-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.018));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        margin-bottom: 0.75rem;
    }
    .header-grid {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 0.75rem;
    }
    .header-title {
        text-align: center;
        font-size: 1.55rem;
        font-weight: 780;
        color: #f8fbff;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .title-badge {
        justify-self: end;
        display: inline-flex;
        align-items: center;
        gap: 0.48rem;
        padding: 0.5rem 0.92rem;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 730;
        color: #f8fbff;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 22px rgba(0,0,0,0.16);
        white-space: nowrap;
    }
    .badge-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.95);
        box-shadow: 0 0 16px rgba(255,255,255,0.28);
        flex: 0 0 10px;
    }

    .lang-switch-wrap {min-width: 168px;}

    .section-title-large {
        font-size: 2.1rem;
        font-weight: 820;
        color: #f8fbff;
        margin: 0 0 0.7rem 0;
        letter-spacing: -0.035em;
    }
    .small-note {
        color:#9fb2c8;
        font-size:1.02rem;
        margin: 0 0 0.35rem 0;
    }
    .factor-column-title {
        font-size: 1.32rem;
        font-weight: 760;
        color: #f8fbff;
        margin: 1rem 0 0.9rem 0;
    }

    .factor-card {
        border-radius: 18px;
        padding: 1rem 1.05rem;
        margin-bottom: 0.86rem;
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
        min-height: 148px;
    }
    .factor-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 30px rgba(0,0,0,0.22);
    }
    .factor-risk {
        background: rgba(230, 57, 70, 0.14);
        border-color: rgba(230, 57, 70, 0.30);
    }
    .factor-risk:hover {border-color: rgba(255, 122, 134, 0.6);}
    .factor-good {
        background: rgba(46, 204, 113, 0.14);
        border-color: rgba(46, 204, 113, 0.30);
    }
    .factor-good:hover {border-color: rgba(113, 227, 177, 0.55);}
    .factor-top {
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:0.75rem;
        margin-bottom:0.35rem;
    }
    .factor-name {font-weight:740; font-size:1.06rem; color:#f8fbff;}
    .factor-val {color:#dbe6f3; font-size:1rem; margin-top:0.2rem;}
    .factor-meta {
        display:flex;
        justify-content:center;
        flex-wrap:wrap;
        gap:0.45rem;
        margin-top:1rem;
        min-height: 34px;
        align-items:center;
    }
    .hint-chip {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        padding:0.28rem 0.72rem;
        border-radius:999px;
        font-size:0.92rem;
        font-weight:690;
        color:#f8fbff;
        background:rgba(255,255,255,0.10);
        border:1px solid rgba(255,255,255,0.10);
    }
    .impact-up {color:#ff9c9c; font-weight:800; letter-spacing:0.05em; font-size:1.08rem;}
    .impact-down {color:#90efc0; font-weight:800; letter-spacing:0.05em; font-size:1.08rem;}

    div[data-testid="stHorizontalBlock"] > div:has(> div > div > div[data-testid="stPlotlyChart"]) {align-self: start;}

    div[data-testid="stRadio"] > label {display:none;}
    div[role="radiogroup"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.22rem;
        border-radius: 999px;
        width: fit-content;
        gap: 0.35rem;
    }
    div[role="radiogroup"] > label {
        background: transparent;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 999px;
        padding: 0.18rem 0.8rem 0.18rem 0.55rem;
        min-width: 78px;
        justify-content: center;
        transition: all 0.18s ease;
    }
    div[role="radiogroup"] > label:has(input:checked) {
        background: rgba(255,255,255,0.10);
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    }
    div[role="radiogroup"] > label > div:first-child {
        width: 12px !important;
        height: 12px !important;
        border-width: 2px !important;
    }
    div[role="radiogroup"] > label:has(input:checked) > div:first-child {
        background: #ef6b5f !important;
        border-color: #ef6b5f !important;
    }

    .stSelectbox label, .stSlider label, .stToggle label {
        font-size: 0.97rem !important;
        font-weight: 600 !important;
        color: #e9f1fb !important;
    }
    .stSlider {padding-right: 1rem;}
    .stSlider [data-baseweb="slider"] {max-width: 290px;}
    .stSelectbox > div[data-baseweb="select"] {max-width: 330px;}
    .stToggle {padding-top: 0.2rem;}
    .element-container {margin-bottom: 0.35rem;}
    .stPlotlyChart {margin-top: -0.35rem; margin-bottom: -0.95rem;}

    @media (max-width: 1100px) {
        .hero-grid {grid-template-columns: 1fr;}
        .lang-switch-wrap {justify-self: start;}
        .header-grid {grid-template-columns: 1fr;}
        .title-badge {justify-self: center;}
        .header-title {text-align: center;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


TEXT = {
    "FR": {
        "title": "Quelle est la qualité de votre sommeil ?",
        "lang_label": "Langue / Language",
        "panel_title": "Ajustez votre profil",
        "country": "Pays",
        "occupation": "Profession",
        "body_type": "Gabarit",
        "age": "Âge",
        "sleep_duration": "Durée de sommeil (heures)",
        "fall_asleep": "Temps d'endormissement (min)",
        "night_awakenings": "Réveils nocturnes",
        "stress": "Stress (/10)",
        "work_hours": "Heures de travail (heures)",
        "alcohol": "Verres d'alcool avant le coucher",
        "coffee": "Cafés avant le coucher",
        "nap": "Sieste",
        "screen": "Temps d'écran avant le coucher",
        "anxiety": "Symptômes anxieux",
        "depression": "Symptômes dépressifs",
        "results": "Résultats",
        "why": "Pourquoi cette prédiction ?",
        "why_caption": "Seuls les facteurs les plus marquants sont affichés.",
        "aggravating": "Facteurs aggravants",
        "beneficial": "Facteurs bénéfiques",
        "no_factor": "Aucun facteur majeur ne ressort.",
        "yes": "Oui",
        "no": "Non",
        "quality_prefix": "Qualité de sommeil",
        "quality_labels": [
            (0, 25, "qualité très dégradée"),
            (25, 50, "qualité fragile"),
            (50, 75, "qualité moyenne"),
            (75, 101, "qualité saine"),
        ],
        "high_value": "Valeur élevée",
        "low_value": "Valeur faible",
    },
    "EN": {
        "title": "How good is your sleep quality?",
        "lang_label": "Language",
        "panel_title": "Tune the profile",
        "country": "Country",
        "occupation": "Occupation",
        "body_type": "Body type",
        "age": "Age",
        "sleep_duration": "Sleep duration (hours)",
        "fall_asleep": "Time to fall asleep (min)",
        "night_awakenings": "Night awakenings",
        "stress": "Stress (/10)",
        "work_hours": "Work hours (hours)",
        "alcohol": "Alcoholic drinks before bed",
        "coffee": "Coffees before bed",
        "nap": "Nap",
        "screen": "Screen time before sleep",
        "anxiety": "Anxiety symptoms",
        "depression": "Depressive symptoms",
        "results": "Results",
        "why": "Why this prediction?",
        "why_caption": "Only the strongest signals are shown.",
        "aggravating": "Aggravating factors",
        "beneficial": "Beneficial factors",
        "no_factor": "No major factor stands out.",
        "yes": "Yes",
        "no": "No",
        "quality_prefix": "Sleep quality",
        "quality_labels": [
            (0, 25, "very poor"),
            (25, 50, "fragile"),
            (50, 75, "average"),
            (75, 101, "healthy"),
        ],
        "high_value": "High value",
        "low_value": "Low value",
    },
}

BODY_TYPE_TO_BMI = {
    "FR": {
        "Très maigre": 16.5,
        "Mince": 19.0,
        "Corpulence normale": 22.5,
        "Surpoids léger": 27.0,
        "Obésité": 32.5,
        "Obésité morbide": 40.0,
    },
    "EN": {
        "Very thin": 16.5,
        "Slim": 19.0,
        "Average build": 22.5,
        "Slightly overweight": 27.0,
        "Obesity": 32.5,
        "Morbid obesity": 40.0,
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


def score_to_quality(raw_score: float) -> float:
    score = max(1.0, min(4.0, raw_score))
    return (4.0 - score) / 3.0 * 100.0


def color_mix_hex(color_a: str, color_b: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    a = tuple(int(color_a[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(color_b[i : i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round((1 - ratio) * x + ratio * y) for x, y in zip(a, b))
    return "#%02x%02x%02x" % mixed


def quality_meta(quality: float, lang: str) -> tuple[str, str]:
    for low, high, label in TEXT[lang]["quality_labels"]:
        if low <= quality < high:
            ratio = quality / 100.0
            bg = f"linear-gradient(135deg, {color_mix_hex('#5a1d27', '#193d2b', ratio)}cc, {color_mix_hex('#7a3021', '#225538', ratio)}cc)"
            return f"{TEXT[lang]['quality_prefix']} : {label}", bg
    return f"{TEXT[lang]['quality_prefix']} : {TEXT[lang]['quality_labels'][-1][2]}", "rgba(255,255,255,0.08)"


def make_gauge(raw_score: float) -> go.Figure:
    quality = score_to_quality(raw_score)
    bar_color = color_mix_hex("#e63946", "#2ecc71", quality / 100.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=quality,
            number={"suffix": "%", "font": {"size": 66}},
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "tickfont": {"size": 14},
                },
                "bar": {"color": bar_color, "thickness": 0.40},
                "steps": [
                    {"range": [0, 25], "color": "rgba(230, 57, 70, 0.26)"},
                    {"range": [25, 50], "color": "rgba(247, 127, 0, 0.20)"},
                    {"range": [50, 75], "color": "rgba(247, 201, 72, 0.16)"},
                    {"range": [75, 100], "color": "rgba(46, 204, 113, 0.22)"},
                ],
            },
        )
    )
    fig.update_layout(height=405, margin=dict(l=8, r=8, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
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


def nice_feature_name(name: str, lang: str) -> str:
    labels = {
        "sleep_duration_hrs": {"FR": "Durée de sommeil", "EN": "Sleep duration"},
        "bmi": {"FR": "Gabarit", "EN": "Body type"},
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


def bmi_to_body_type_label(bmi: float, lang: str) -> str:
    mapping = BODY_TYPE_TO_BMI[lang]
    return min(mapping.keys(), key=lambda label: abs(mapping[label] - float(bmi)))


def format_value(feature: str, value, lang: str) -> str:
    if feature in {"sleep_duration_hrs", "work_hours_that_day"}:
        return f"{float(value):.2f} h"
    if feature == "stress_score":
        return f"{float(value):.1f}/10"
    if feature == "sleep_latency_mins":
        return f"{int(value)} min"
    if feature == "wake_episodes_per_night":
        return str(int(value))
    if feature == "age":
        return str(int(value))
    if feature == "bmi":
        return bmi_to_body_type_label(float(value), lang)
    if feature == "alcohol_units_before_bed":
        amount = float(value)
        if lang == "FR":
            unit = "verre" if amount <= 1 else "verres"
        else:
            unit = "drink" if amount <= 1 else "drinks"
        if amount.is_integer():
            return f"{int(amount)} {unit}"
        return f"{amount:.1f} {unit}"
    if feature == "nb_cafe_before_bed":
        return COFFEE_LABELS[lang].get(int(value), str(value))
    if feature in {"Anxiety", "Depression"}:
        return tr(lang, "yes") if int(value) == 1 else tr(lang, "no")
    return str(value)


def factor_hint(feature: str, value, lang: str) -> str | None:
    if feature in {"country", "occupation", "nap_time", "time_screen_before_sleep", "Anxiety", "Depression"}:
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if feature == "sleep_duration_hrs":
        return tr(lang, "low_value") if val < 7 else tr(lang, "high_value")
    if feature == "stress_score":
        return tr(lang, "high_value") if val >= 5 else tr(lang, "low_value")
    if feature == "sleep_latency_mins":
        return tr(lang, "high_value") if val >= 20 else tr(lang, "low_value")
    if feature == "wake_episodes_per_night":
        return tr(lang, "high_value") if val >= 2 else tr(lang, "low_value")
    if feature == "age":
        return tr(lang, "high_value") if val >= 50 else tr(lang, "low_value")
    if feature == "work_hours_that_day":
        return tr(lang, "high_value") if val >= 8 else tr(lang, "low_value")
    if feature == "alcohol_units_before_bed":
        return tr(lang, "high_value") if val >= 1 else tr(lang, "low_value")
    if feature == "nb_cafe_before_bed":
        return tr(lang, "high_value") if val >= 1 else tr(lang, "low_value")
    if feature == "bmi":
        return tr(lang, "high_value") if val >= 27 or val < 18.5 else tr(lang, "low_value")
    return None


def render_factor_cards(title: str, factors: list, lang: str, positive: bool) -> None:
    st.markdown(f"<div class='factor-column-title'>{html.escape(title)}</div>", unsafe_allow_html=True)
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
        feat = html.escape(nice_feature_name(factor["feature"], lang))
        val = html.escape(format_value(factor["feature"], factor["value"], lang))
        chip = factor_hint(factor["feature"], factor["value"], lang)
        chip_html = f"<span class='hint-chip'>{html.escape(chip)}</span>" if chip else ""
        st.markdown(
            f"""
            <div class='factor-card {card_class}'>
                <div class='factor-top'>
                    <div class='factor-name'>{feat}</div>
                    <div>{arrows}</div>
                </div>
                <div class='factor-val'>{val}</div>
                <div class='factor-meta'>{chip_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_header(title: str, badge_html: str = "") -> None:
    st.markdown(
        f"""
        <div class="header-card">
            <div class="header-grid">
                <div></div>
                <div class="header-title">{html.escape(title)}</div>
                <div>{badge_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def input_panel(options: dict, lang: str) -> dict:
    country_options = options.get("country") or DEFAULT_COUNTRIES
    occupation_options = options.get("occupation") or DEFAULT_OCCUPATIONS
    body_labels = list(BODY_TYPE_TO_BMI[lang].keys())
    default_body_label = bmi_to_body_type_label(float(FORM_DEFAULTS["bmi"]), lang)

    render_header(tr(lang, "panel_title"))

    top = st.columns(3)
    with top[0]:
        country = st.selectbox(
            tr(lang, "country"),
            country_options,
            index=country_options.index(FORM_DEFAULTS["country"]) if FORM_DEFAULTS["country"] in country_options else 0,
        )
    with top[1]:
        occupation = st.selectbox(
            tr(lang, "occupation"),
            occupation_options,
            index=occupation_options.index(FORM_DEFAULTS["occupation"]) if FORM_DEFAULTS["occupation"] in occupation_options else 0,
        )
    with top[2]:
        body_type = st.selectbox(
            tr(lang, "body_type"),
            body_labels,
            index=body_labels.index(default_body_label),
        )

    row1 = st.columns(3)
    with row1[0]:
        age = st.slider(tr(lang, "age"), 18, 90, int(FORM_DEFAULTS["age"]))
    with row1[1]:
        sleep_duration_hrs = st.slider(
            tr(lang, "sleep_duration"), 2.0, 12.0, float(FORM_DEFAULTS["sleep_duration_hrs"]), 0.25, format="%.2f h"
        )
    with row1[2]:
        sleep_latency_mins = st.slider(tr(lang, "fall_asleep"), 0, 120, int(FORM_DEFAULTS["sleep_latency_mins"]), 5)

    row2 = st.columns(3)
    with row2[0]:
        wake_episodes_per_night = st.slider(tr(lang, "night_awakenings"), 0, 10, int(FORM_DEFAULTS["wake_episodes_per_night"]))
    with row2[1]:
        stress_score = st.slider(tr(lang, "stress"), 0.0, 10.0, float(FORM_DEFAULTS["stress_score"]), 0.5)
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
        time_screen_before_sleep = st.selectbox(
            tr(lang, "screen"),
            SCREEN_TIME_OPTIONS,
            index=SCREEN_TIME_OPTIONS.index(FORM_DEFAULTS["time_screen_before_sleep"]),
        )

    row4 = st.columns(3)
    with row4[0]:
        nap_time = st.selectbox(tr(lang, "nap"), NAP_OPTIONS, index=NAP_OPTIONS.index(FORM_DEFAULTS["nap_time"]))
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
        "bmi": BODY_TYPE_TO_BMI[lang][body_type],
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
    if "ui_lang" not in st.session_state:
        st.session_state.ui_lang = "FR"

    hero_left, hero_right = st.columns([0.84, 0.16], gap="medium")
    with hero_right:
        st.markdown("<div class='lang-switch-wrap'></div>", unsafe_allow_html=True)
        lang = st.radio(
            tr("FR", "lang_label"),
            options=["FR", "EN"],
            horizontal=True,
            label_visibility="collapsed",
            index=0 if st.session_state.ui_lang == "FR" else 1,
            key="ui_lang",
        )
    with hero_left:
        st.markdown(f"<div class='hero'><div class='hero-grid'><div class='page-title'>{html.escape(tr(lang, 'title'))}</div></div></div>", unsafe_allow_html=True)

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Place your exported model in models/lgbm_full.joblib, or run train.py to generate it.")
        return

    metadata = predictor.metadata or {}
    category_options = metadata.get("category_options", {})

    left, right = st.columns([1.03, 0.97], gap="large")

    with left:
        payload = input_panel(category_options, lang)

    result = predictor.predict_dataframe(pd.DataFrame([payload]))
    quality = score_to_quality(result.raw_score)
    quality_label, quality_bg = quality_meta(quality, lang)
    badge_html = (
        f"<span class='title-badge' style=\"background:{quality_bg};\">"
        f"<span class='badge-dot'></span>{html.escape(quality_label)}</span>"
    )

    with right:
        render_header(tr(lang, "results"), badge_html)
        st.plotly_chart(make_gauge(result.raw_score), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='header-card' style='margin-bottom:0.15rem;'>
            <div class='section-title-large'>{html.escape(tr(lang, 'why'))}</div>
            <div class='small-note'>{html.escape(tr(lang, 'why_caption'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    exp_col1, exp_col2 = st.columns(2, gap="large")
    with exp_col1:
        render_factor_cards(tr(lang, "aggravating"), result.top_positive_factors, lang, positive=True)
    with exp_col2:
        render_factor_cards(tr(lang, "beneficial"), result.top_negative_factors, lang, positive=False)


if __name__ == "__main__":
    main()
