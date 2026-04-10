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
    .block-container {padding-top: 0.8rem; padding-bottom: 1.4rem; max-width: 1450px;}
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
    .page-title {
        font-size: clamp(2.25rem, 3.6vw, 4.2rem);
        font-weight: 840;
        color: #f8fbff;
        margin: 0;
        line-height: 1.05;
        letter-spacing: -0.045em;
    }

    .lang-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 28px;
        padding: 0.35rem;
        display: flex;
        gap: 0.35rem;
        justify-content: center;
        align-items: center;
        width: 100%;
        min-height: 74px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .lang-card .stButton > button {
        width: 100%;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        min-height: 42px;
        font-weight: 700;
    }

    .header-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.018));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        margin-bottom: 0.9rem;
    }
    .header-grid {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 0.75rem;
    }
    .header-title {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 790;
        color: #f8fbff;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .title-badge {
        justify-self: end;
        display: inline-flex;
        align-items: center;
        gap: 0.48rem;
        padding: 0.48rem 0.9rem;
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

    .section-title-large {
        font-size: 2.15rem;
        font-weight: 820;
        color: #f8fbff;
        margin: 0 0 0.55rem 0;
        letter-spacing: -0.035em;
    }
    .small-note {
        color:#9fb2c8;
        font-size:1.02rem;
        margin: 0 0 0.35rem 0;
    }
    .subsection-title {
        font-size: 1.52rem;
        font-weight: 780;
        color: #f8fbff;
        margin: 1.35rem 0 0.95rem 0;
        letter-spacing: -0.02em;
    }

    .factor-card {
        border-radius: 18px;
        padding: 1rem 1.05rem;
        margin-bottom: 0.86rem;
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
        min-height: 124px;
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
        align-items:center;
        gap:0.75rem;
        margin-bottom:0.38rem;
    }
    .factor-left {
        display:flex;
        align-items:center;
        gap:0.6rem;
        min-width:0;
        flex-wrap:wrap;
    }
    .factor-name {font-weight:760; font-size:1.06rem; color:#f8fbff;}
    .factor-val {color:#dbe6f3; font-size:1rem; margin-top:0.15rem;}
    .hint-chip {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        padding:0.24rem 0.68rem;
        border-radius:999px;
        font-size:0.89rem;
        font-weight:690;
        color:#f8fbff;
        background:rgba(255,255,255,0.10);
        border:1px solid rgba(255,255,255,0.10);
        line-height:1;
    }
    .impact-up {color:#ff9c9c; font-weight:800; letter-spacing:0.05em; font-size:1.08rem; white-space:nowrap;}
    .impact-down {color:#90efc0; font-weight:800; letter-spacing:0.05em; font-size:1.08rem; white-space:nowrap;}

    .stSelectbox label, .stSlider label, .stToggle label {
        font-size: 0.97rem !important;
        font-weight: 600 !important;
        color: #e9f1fb !important;
    }
    .element-container {margin-bottom: 0.35rem;}
    .stSlider [data-baseweb="slider"] {max-width: 240px;}
    .stSelectbox > div[data-baseweb="select"] {max-width: 340px;}
    .stPlotlyChart {margin-top: -0.65rem; margin-bottom: -1.35rem;}
    .plot-wrap {margin-top: -0.5rem;}

    @media (max-width: 1100px) {
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
        "panel_title": "Ajustez votre profil",
        "country": "Pays",
        "occupation": "Profession",
        "body_type": "Gabarit",
        "age": "Âge",
        "sleep_duration": "Durée de sommeil",
        "fall_asleep": "Temps d'endormissement",
        "night_awakenings": "Réveils nocturnes",
        "stress": "Stress",
        "work_hours": "Heures de travail",
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
        "quality_band": [
            (0, 25, "très dégradée"),
            (25, 50, "fragile"),
            (50, 75, "moyenne"),
            (75, 101, "saine"),
        ],
        "high_value": "Valeur élevée",
        "low_value": "Valeur faible",
        "present": "Présent",
        "absent": "Absent",
        "model_missing": "Étape attendue : place ton modèle exporté dans models/lgbm_full.joblib, ou lance train.py pour le générer.",
    },
    "EN": {
        "title": "How good is your sleep quality?",
        "panel_title": "Tune the profile",
        "country": "Country",
        "occupation": "Occupation",
        "body_type": "Body type",
        "age": "Age",
        "sleep_duration": "Sleep duration",
        "fall_asleep": "Time to fall asleep",
        "night_awakenings": "Night awakenings",
        "stress": "Stress",
        "work_hours": "Work hours",
        "alcohol": "Alcoholic drinks before bed",
        "coffee": "Coffees before bed",
        "nap": "Nap",
        "screen": "Screen time before bed",
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
        "quality_band": [
            (0, 25, "very degraded"),
            (25, 50, "fragile"),
            (50, 75, "average"),
            (75, 101, "healthy"),
        ],
        "high_value": "High value",
        "low_value": "Low value",
        "present": "Present",
        "absent": "Absent",
        "model_missing": "Expected next step: place your exported model in models/lgbm_full.joblib, or run train.py to generate it.",
    },
}

BODY_TYPE_OPTIONS = {
    "FR": [
        ("Très maigre", 16.5),
        ("Mince", 19.0),
        ("Corpulence normale", 22.0),
        ("Corpulence forte", 26.5),
        ("Obésité", 32.5),
        ("Obésité morbide", 41.0),
    ],
    "EN": [
        ("Very thin", 16.5),
        ("Lean", 19.0),
        ("Average build", 22.0),
        ("Broad build", 26.5),
        ("Obesity", 32.5),
        ("Morbid obesity", 41.0),
    ],
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
        "time_screen_before_sleep": {"FR": "Temps d'écran avant le coucher", "EN": "Screen time before bed"},
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


def quality_band(quality: float, lang: str) -> tuple[str, str]:
    palette = [
        "rgba(230, 57, 70, 0.42)",
        "rgba(247, 127, 0, 0.38)",
        "rgba(212, 170, 55, 0.35)",
        "rgba(46, 204, 113, 0.34)",
    ]
    for idx, (lo, hi, label) in enumerate(tr(lang, "quality_band")):
        if lo <= quality < hi:
            return label, palette[idx]
    return tr(lang, "quality_band")[-1][2], palette[-1]


def make_gauge(raw_score: float) -> go.Figure:
    quality = score_to_quality(raw_score)
    bar_color = color_mix_hex("#e63946", "#2ecc71", quality / 100.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=quality,
            number={"suffix": "%", "font": {"size": 66}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "tickfont": {"size": 14},
                },
                "bar": {"color": bar_color, "thickness": 0.42},
                "steps": [
                    {"range": [0, 25], "color": "rgba(230, 57, 70, 0.26)"},
                    {"range": [25, 50], "color": "rgba(247, 127, 0, 0.22)"},
                    {"range": [50, 75], "color": "rgba(212, 170, 55, 0.20)"},
                    {"range": [75, 100], "color": "rgba(46, 204, 113, 0.22)"},
                ],
            },
        )
    )
    fig.update_layout(height=440, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
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


def body_label_from_bmi(value: float, lang: str) -> str:
    options = BODY_TYPE_OPTIONS[lang]
    best_label, _ = min(options, key=lambda item: abs(item[1] - float(value)))
    return best_label


def format_hours(value: float) -> str:
    return f"{value:.2f} h"


def format_value(feature: str, value, lang: str) -> str:
    if feature in {"sleep_duration_hrs", "work_hours_that_day"}:
        return format_hours(float(value))
    if feature == "stress_score":
        return f"{float(value):.1f}/10"
    if feature == "sleep_latency_mins":
        return f"{int(value)} min"
    if feature == "wake_episodes_per_night":
        return str(int(value))
    if feature == "age":
        return str(int(value))
    if feature == "bmi":
        return body_label_from_bmi(float(value), lang)
    if feature == "alcohol_units_before_bed":
        unit = "verre" if lang == "FR" else "drink"
        unit_pl = "verres" if lang == "FR" else "drinks"
        amount = float(value)
        amount_txt = f"{int(amount)}" if float(amount).is_integer() else f"{amount:.1f}"
        unit_out = unit if amount <= 1 else unit_pl
        return f"{amount_txt} {unit_out}"
    if feature == "nb_cafe_before_bed":
        return COFFEE_LABELS[lang].get(int(value), str(value))
    if feature in {"Anxiety", "Depression"}:
        return tr(lang, "yes") if int(value) == 1 else tr(lang, "no")
    return str(value)


def explain_hint(feature: str, value, lang: str) -> str | None:
    high_label = tr(lang, "high_value")
    low_label = tr(lang, "low_value")

    if feature == "sleep_duration_hrs":
        return low_label if float(value) < 6.5 else None
    if feature == "sleep_latency_mins":
        return high_label if int(value) >= 30 else low_label
    if feature == "wake_episodes_per_night":
        return high_label if int(value) >= 2 else low_label
    if feature == "stress_score":
        return high_label if float(value) >= 6 else low_label
    if feature == "bmi":
        if float(value) >= 27:
            return high_label
        if float(value) <= 18.5:
            return low_label
        return None
    if feature == "work_hours_that_day":
        return high_label if float(value) >= 9 else low_label
    if feature == "alcohol_units_before_bed":
        return high_label if float(value) > 0 else low_label
    if feature == "nb_cafe_before_bed":
        return high_label if int(value) >= 2 else low_label
    if feature == "time_screen_before_sleep":
        value_str = str(value).lower()
        return high_label if ("60" in value_str or "2h" in value_str) else low_label
    if feature == "age":
        return high_label if int(value) >= 60 else None
    return None


def render_factor_cards(title: str, factors: list, lang: str, positive: bool) -> None:
    st.markdown(f"<div class='subsection-title'>{html.escape(title)}</div>", unsafe_allow_html=True)
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
        hint = explain_hint(factor["feature"], factor["value"], lang)
        hint_html = f"<span class='hint-chip'>{html.escape(hint)}</span>" if hint else ""
        st.markdown(
            f"""
            <div class='factor-card {card_class}'>
                <div class='factor-top'>
                    <div class='factor-left'>
                        <div class='factor-name'>{feat}</div>
                        {hint_html}
                    </div>
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
    body_options = BODY_TYPE_OPTIONS[lang]
    body_labels = [label for label, _ in body_options]
    default_body_idx = min(range(len(body_options)), key=lambda i: abs(body_options[i][1] - float(FORM_DEFAULTS["bmi"])))

    st.markdown(
        f"<div class='header-card'><div class='header-grid'><div></div><div class='header-title'>{html.escape(tr(lang, 'panel_title'))}</div><div></div></div></div>",
        unsafe_allow_html=True,
    )

    row0 = st.columns(3)
    with row0[0]:
        country = st.selectbox(
            tr(lang, "country"),
            country_options,
            index=country_options.index(FORM_DEFAULTS["country"]) if FORM_DEFAULTS["country"] in country_options else 0,
        )
    with row0[1]:
        occupation = st.selectbox(
            tr(lang, "occupation"),
            occupation_options,
            index=occupation_options.index(FORM_DEFAULTS["occupation"]) if FORM_DEFAULTS["occupation"] in occupation_options else 0,
        )
    with row0[2]:
        body_label = st.selectbox(tr(lang, "body_type"), body_labels, index=default_body_idx)

    row1 = st.columns(3)
    with row1[0]:
        age = st.slider(tr(lang, "age"), 18, 90, int(FORM_DEFAULTS["age"]))
    with row1[1]:
        sleep_duration_hrs = st.slider(
            tr(lang, "sleep_duration"),
            min_value=3.0,
            max_value=12.0,
            value=float(FORM_DEFAULTS["sleep_duration_hrs"]),
            step=0.25,
            format="%.2f h",
        )
    with row1[2]:
        sleep_latency_mins = st.slider(
            tr(lang, "fall_asleep"),
            min_value=0,
            max_value=120,
            value=int(FORM_DEFAULTS["sleep_latency_mins"]),
            step=5,
            format="%d min",
        )

    row2 = st.columns(3)
    with row2[0]:
        wake_episodes_per_night = st.slider(
            tr(lang, "night_awakenings"),
            min_value=0,
            max_value=10,
            value=int(FORM_DEFAULTS["wake_episodes_per_night"]),
        )
    with row2[1]:
        stress_score = st.slider(
            tr(lang, "stress"),
            min_value=0.0,
            max_value=10.0,
            value=float(FORM_DEFAULTS["stress_score"]),
            step=0.5,
            format="%.1f",
        )
    with row2[2]:
        work_hours_that_day = st.slider(
            tr(lang, "work_hours"),
            min_value=0.0,
            max_value=16.0,
            value=float(FORM_DEFAULTS["work_hours_that_day"]),
            step=0.25,
            format="%.2f h",
        )

    row3 = st.columns(3)
    with row3[0]:
        alcohol_units_before_bed = st.slider(
            tr(lang, "alcohol"),
            min_value=0.0,
            max_value=8.0,
            value=float(FORM_DEFAULTS["alcohol_units_before_bed"]),
            step=0.5,
            format="%.1f",
        )
    with row3[1]:
        nb_cafe_before_bed = st.slider(
            tr(lang, "coffee"),
            min_value=0,
            max_value=3,
            value=int(FORM_DEFAULTS["nb_cafe_before_bed"]),
            step=1,
            format="%d",
        )
    with row3[2]:
        st.empty()

    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

    row4 = st.columns(3)
    with row4[0]:
        nap_time = st.selectbox(tr(lang, "nap"), NAP_OPTIONS, index=NAP_OPTIONS.index(FORM_DEFAULTS["nap_time"]))
    with row4[1]:
        time_screen_before_sleep = st.selectbox(
            tr(lang, "screen"),
            SCREEN_TIME_OPTIONS,
            index=SCREEN_TIME_OPTIONS.index(FORM_DEFAULTS["time_screen_before_sleep"]),
        )
    with row4[2]:
        st.empty()

    st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)

    row5 = st.columns(3)
    with row5[0]:
        anxiety = 1 if st.toggle(tr(lang, "anxiety"), value=bool(FORM_DEFAULTS["Anxiety"])) else 0
    with row5[1]:
        depression = 1 if st.toggle(tr(lang, "depression"), value=bool(FORM_DEFAULTS["Depression"])) else 0
    with row5[2]:
        st.empty()

    bmi = dict(body_options)[body_label]
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


def render_language_switch() -> str:
    if "lang" not in st.session_state:
        st.session_state.lang = "FR"

    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button("FR", key="lang_fr", type="primary" if st.session_state.lang == "FR" else "secondary", use_container_width=True):
            st.session_state.lang = "FR"
    with col2:
        if st.button("EN", key="lang_en", type="primary" if st.session_state.lang == "EN" else "secondary", use_container_width=True):
            st.session_state.lang = "EN"
    return st.session_state.lang


def main() -> None:
    top_left, top_right = st.columns([0.84, 0.16], gap="medium")
    with top_left:
        st.markdown(f"<div class='hero'><h1 class='page-title'>{html.escape(TEXT[st.session_state.get('lang', 'FR')]['title'])}</h1></div>", unsafe_allow_html=True)
    with top_right:
        st.markdown("<div class='lang-card'>", unsafe_allow_html=True)
        lang = render_language_switch()
        st.markdown("</div>", unsafe_allow_html=True)

    # re-render title after language button interactions on the same run
    if lang != st.session_state.get("title_lang_rendered"):
        st.session_state.title_lang_rendered = lang
        st.rerun()

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(tr(lang, "model_missing"))
        return

    metadata = predictor.metadata or {}
    category_options = metadata.get("category_options", {})

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        payload = input_panel(category_options, lang)

    result = predictor.predict_dataframe(pd.DataFrame([payload]))
    quality = score_to_quality(result.raw_score)
    band_label, band_color = quality_band(quality, lang)
    badge_text = f"{tr(lang, 'quality_prefix')} : {band_label}"

    with right:
        st.markdown(
            f"<div class='header-card'><div class='header-grid'><div></div><div class='header-title'>{html.escape(tr(lang, 'results'))}</div><div class='title-badge' style='background:{band_color}'><span class='badge-dot'></span>{html.escape(badge_text)}</div></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='plot-wrap'>", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(result.raw_score), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='header-card'><div class='section-title-large'>{html.escape(tr(lang, 'why'))}</div><div class='small-note'>{html.escape(tr(lang, 'why_caption'))}</div></div>", unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns(2, gap="large")
    with exp_col1:
        render_factor_cards(tr(lang, "aggravating"), result.top_positive_factors, lang, positive=True)
    with exp_col2:
        render_factor_cards(tr(lang, "beneficial"), result.top_negative_factors, lang, positive=False)


if __name__ == "__main__":
    main()
