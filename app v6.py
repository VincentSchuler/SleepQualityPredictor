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
    .block-container {padding-top: 1.1rem; padding-bottom: 1.8rem; max-width: 1440px;}
    h1, h2, h3 {color: #f8fbff; letter-spacing: -0.02em;}
    p, label, div, span {color: #d7e3f2;}

    .hero-card {
        background: linear-gradient(135deg, rgba(76, 201, 240, 0.16), rgba(114, 9, 183, 0.14));
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.05rem 1.2rem;
        border-radius: 24px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .page-title {
        font-size: clamp(1.95rem, 3vw, 3.1rem);
        font-weight: 840;
        color: #f8fbff;
        margin: 0;
        line-height: 1.08;
        letter-spacing: -0.04em;
    }
    .section-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 22px;
        padding: 1.05rem 1rem 0.95rem 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .section-title {
        font-size: 1.42rem;
        font-weight: 790;
        color: #f8fbff;
        margin: 0 0 0.8rem 0;
    }
    .subsection-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #edf4fc;
        margin: 0 0 0.7rem 0;
        opacity: 0.95;
    }
    .quality-badge-wrap {
        display:flex;
        justify-content:center;
        margin-top:0.15rem;
    }
    .quality-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.45rem;
        padding: 0.56rem 1rem;
        border-radius: 999px;
        font-size: 0.98rem;
        font-weight: 720;
        color: #f8fbff;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 22px rgba(0,0,0,0.16);
    }
    .quality-badge-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.95);
        box-shadow: 0 0 16px rgba(255,255,255,0.25);
    }
    .small-note {color:#9fb2c8; font-size:0.95rem; margin-top: 0.08rem; margin-bottom: 0.85rem;}

    .factor-card {
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.82rem;
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        box-shadow: 0 8px 22px rgba(0,0,0,0.12);
        min-height: 146px;
    }
    .factor-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 30px rgba(0,0,0,0.22);
    }
    .factor-risk {
        background: rgba(230, 57, 70, 0.14);
        border-color: rgba(230, 57, 70, 0.30);
    }
    .factor-risk:hover {
        border-color: rgba(255, 122, 134, 0.6);
    }
    .factor-good {
        background: rgba(46, 204, 113, 0.14);
        border-color: rgba(46, 204, 113, 0.30);
    }
    .factor-good:hover {
        border-color: rgba(113, 227, 177, 0.55);
    }
    .factor-top {
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:0.75rem;
        margin-bottom:0.35rem;
    }
    .factor-name {font-weight:730; font-size:1rem; color:#f8fbff;}
    .factor-val {color:#dbe6f3; font-size:0.98rem;}
    .factor-meta {display:flex; justify-content:center; flex-wrap:wrap; gap:0.45rem; margin-top:1rem;}
    .hint-chip {
        display:inline-flex;
        align-items:center;
        padding:0.26rem 0.7rem;
        border-radius:999px;
        font-size:0.82rem;
        font-weight:650;
        border:1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.10);
        color:#eef5ff;
    }
    .impact-up {color:#ff8f8f; font-weight:800; letter-spacing:0.06em; font-size:1.05rem;}
    .impact-down {color:#9cf1c8; font-weight:800; letter-spacing:0.06em; font-size:1.05rem;}

    .stSlider, div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {margin-bottom: 0.1rem;}
    .stSlider {max-width: 90%;}
    .stSlider [data-testid="stThumbValue"] {display:none;}
    .stToggle label p, .stRadio label p {font-size: 0.98rem !important;}

    div[role="radiogroup"] {
        display:flex;
        gap:0.42rem;
        justify-content:flex-end;
        flex-wrap:wrap;
        margin-top:0.2rem;
    }
    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 999px;
        padding: 0.18rem 0.8rem;
        min-height: auto;
    }
    div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(180deg, rgba(255,255,255,0.14), rgba(255,255,255,0.08));
        border-color: rgba(255,255,255,0.22);
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


TEXT = {
    "FR": {
        "title": "Quelle est la qualité de votre sommeil ?",
        "language_ui": "FR / EN",
        "panel_title": "Tune the profile",
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
        "quality_band": [
            (0, 25, "Qualité de sommeil très dégradée"),
            (25, 50, "Qualité de sommeil fragile"),
            (50, 75, "Qualité de sommeil moyenne"),
            (75, 101, "Qualité de sommeil saine"),
        ],
        "high_value": "Valeur élevée",
        "low_value": "Valeur faible",
        "present": "Présent",
        "absent": "Absent",
    },
    "EN": {
        "title": "How good is your sleep quality?",
        "language_ui": "FR / EN",
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
        "quality_band": [
            (0, 25, "Very poor sleep quality"),
            (25, 50, "Fragile sleep quality"),
            (50, 75, "Average sleep quality"),
            (75, 101, "Healthy sleep quality"),
        ],
        "high_value": "High value",
        "low_value": "Low value",
        "present": "Present",
        "absent": "Absent",
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
            number={"suffix": "%", "font": {"size": 64}},
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
    fig.update_layout(height=470, margin=dict(l=6, r=6, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
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


def explain_hint(feature: str, value, impact: float, lang: str) -> str | None:
    high_label = tr(lang, "high_value")
    low_label = tr(lang, "low_value")
    present = tr(lang, "present")
    absent = tr(lang, "absent")

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
    if feature in {"Anxiety", "Depression"}:
        return present if int(value) == 1 else absent
    if feature == "time_screen_before_sleep":
        value_str = str(value).lower()
        return high_label if ("60" in value_str or "2h" in value_str) else low_label
    if feature == "age":
        return high_label if int(value) >= 60 else None
    return None


def render_factor_cards(title: str, factors: list, lang: str, positive: bool) -> None:
    st.markdown(f"<div class='subsection-title'>{title}</div>", unsafe_allow_html=True)
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
        hint = explain_hint(factor["feature"], factor["value"], float(factor["impact"]), lang)
        hint_html = f"<div class='factor-meta'><span class='hint-chip'>{hint}</span></div>" if hint else ""
        st.markdown(
            f"""
            <div class='factor-card {card_class}'>
                <div class='factor-top'>
                    <div class='factor-name'>{feat}</div>
                    <div>{arrows}</div>
                </div>
                <div class='factor-val'>{val}</div>
                {hint_html}
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

    with st.container(border=False):
        st.markdown(f"<div class='section-title'>{tr(lang, 'panel_title')}</div>", unsafe_allow_html=True)

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
            bmi = dict(body_options)[body_label]

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
            wake_episodes_per_night = st.slider(
                tr(lang, "night_awakenings"), 0, 10, int(FORM_DEFAULTS["wake_episodes_per_night"])
            )
        with row2[1]:
            stress_score = st.slider(tr(lang, "stress"), 0.0, 10.0, float(FORM_DEFAULTS["stress_score"]), 0.5)
        with row2[2]:
            work_hours_that_day = st.slider(
                tr(lang, "work_hours"), 0.0, 16.0, float(FORM_DEFAULTS["work_hours_that_day"]), 0.25, format="%.2f h"
            )

        row3 = st.columns(3)
        with row3[0]:
            alcohol_units_before_bed = st.slider(
                tr(lang, "alcohol"), 0, 8, int(round(float(FORM_DEFAULTS["alcohol_units_before_bed"]))), 1
            )
        with row3[1]:
            nb_cafe_before_bed = st.slider(
                tr(lang, "coffee"), 0, 3, int(FORM_DEFAULTS["nb_cafe_before_bed"]), 1, format="%d"
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
        "bmi": bmi,
        "work_hours_that_day": work_hours_that_day,
        "alcohol_units_before_bed": float(alcohol_units_before_bed),
        "nb_cafe_before_bed": nb_cafe_before_bed,
        "country": country,
        "occupation": occupation,
        "nap_time": nap_time,
        "time_screen_before_sleep": time_screen_before_sleep,
        "Anxiety": anxiety,
        "Depression": depression,
    }


def main() -> None:
    st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
    title_col, lang_col = st.columns([0.82, 0.18])
    with title_col:
        title_placeholder = st.empty()
    with lang_col:
        lang = st.radio(
            tr("FR", "language_ui"),
            options=["FR", "EN"],
            horizontal=True,
            label_visibility="collapsed",
            index=0,
        )
    title_placeholder.markdown(f"<div class='page-title'>{tr(lang, 'title')}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Place your exported model in models/lgbm_full.joblib, or run train.py to generate it.")
        return

    metadata = predictor.metadata or {}
    category_options = metadata.get("category_options", {})

    left, right = st.columns([1.18, 0.82], gap="large")

    with left:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        payload = input_panel(category_options, lang)
        st.markdown("</div>", unsafe_allow_html=True)

    result = predictor.predict_dataframe(pd.DataFrame([payload]))
    quality = score_to_quality(result.raw_score)
    badge_text, badge_color = quality_band(quality, lang)

    with right:
        st.markdown(f"<div class='section-card'><div class='section-title'>{tr(lang, 'results')}</div>", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(result.raw_score), use_container_width=True)
        st.markdown(
            f"<div class='quality-badge-wrap'><div class='quality-badge' style='background:{badge_color}'><span class='quality-badge-dot'></span>{badge_text}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-card'><div class='section-title' style='font-size:2.05rem'>{tr(lang, 'why')}</div><div class='small-note'>{tr(lang, 'why_caption')}</div>",
        unsafe_allow_html=True,
    )
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        render_factor_cards(tr(lang, "aggravating"), result.top_positive_factors, lang, positive=True)
    with exp_col2:
        render_factor_cards(tr(lang, "beneficial"), result.top_negative_factors, lang, positive=False)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
