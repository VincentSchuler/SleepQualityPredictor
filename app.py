from __future__ import annotations

import html
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

st.set_page_config(page_title="Sleep Quality", page_icon="💤", layout="wide", initial_sidebar_state="collapsed")

TEXT = {
    "FR": {
        "title": "Quelle est la qualité de votre sommeil ?",
        "profile": "Ajustez votre profil",
        "results": "Résultats",
        "healthy": "Sommeil sain",
        "fair": "Sommeil correct",
        "fragile": "Sommeil fragile",
        "poor": "Sommeil dégradé",
        "why": "Pourquoi cette prédiction ?",
        "why_caption": "Seuls les facteurs les plus marquants sont affichés.",
        "aggravating": "Facteurs aggravants",
        "beneficial": "Facteurs bénéfiques",
        "no_factor": "Aucun facteur marquant ne ressort.",
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
        "screen": "Temps d'écran avant le coucher",
        "nap": "Sieste",
        "anxiety": "Symptômes anxieux",
        "depression": "Symptômes dépressifs",
        "yes": "Oui",
        "no": "Non",
        "low_value": "Valeur faible",
        "high_value": "Valeur élevée",
        "present": "Présent",
    },
    "EN": {
        "title": "How good is your sleep quality?",
        "profile": "Tune the profile",
        "results": "Results",
        "healthy": "Healthy sleep",
        "fair": "Fair sleep",
        "fragile": "Fragile sleep",
        "poor": "Poor sleep",
        "why": "Why this prediction?",
        "why_caption": "Only the strongest factors are shown.",
        "aggravating": "Risk-increasing factors",
        "beneficial": "Helpful factors",
        "no_factor": "No strong factor stands out.",
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
        "screen": "Screen time before bed",
        "nap": "Nap",
        "anxiety": "Anxiety symptoms",
        "depression": "Depressive symptoms",
        "yes": "Yes",
        "no": "No",
        "low_value": "Low value",
        "high_value": "High value",
        "present": "Present",
    },
}

BODY_TYPE_OPTIONS = {
    "FR": [
        ("Très maigre", 16.0),
        ("Mince", 19.0),
        ("Corpulence normale", 23.0),
        ("Surpoids léger", 28.0),
        ("Surpoids marqué", 32.0),
        ("Obésité", 36.0),
        ("Obésité morbide", 42.0),
    ],
    "EN": [
        ("Very thin", 16.0),
        ("Lean", 19.0),
        ("Normal build", 23.0),
        ("Slightly overweight", 28.0),
        ("Overweight", 32.0),
        ("Obesity", 36.0),
        ("Severe obesity", 42.0),
    ],
}
COFFEE_LABELS = {
    "FR": {0: "0 café", 1: "1 café", 2: "2 cafés", 3: "3 cafés", 4: "4+ cafés"},
    "EN": {0: "0 coffee", 1: "1 coffee", 2: "2 coffees", 3: "3 coffees", 4: "4+ coffees"},
}


OCCUPATION_LABELS_FR = {
    "Software Engineer": "Ingénieur logiciel",
    "Doctor": "Médecin",
    "Teacher": "Enseignant",
    "Nurse": "Infirmier",
    "Sales Representative": "Commercial",
    "Salesperson": "Vendeur",
    "Accountant": "Comptable",
    "Scientist": "Scientifique",
    "Lawyer": "Avocat",
    "Engineer": "Ingénieur",
    "Manager": "Manager",
    "Student": "Étudiant",
    "Retired": "Retraité",
    "Unemployed": "Sans emploi",
    "Business Analyst": "Analyste métier",
    "Researcher": "Chercheur",
    "Consultant": "Consultant",
    "Entrepreneur": "Entrepreneur",
    "Marketing Manager": "Responsable marketing",
    "Marketing Specialist": "Spécialiste marketing",
    "HR Manager": "Responsable RH",
    "HR Specialist": "Spécialiste RH",
    "Product Manager": "Chef de produit",
    "Project Manager": "Chef de projet",
    "Data Scientist": "Data scientist",
    "Data Analyst": "Analyste de données",
    "Financial Analyst": "Analyste financier",
    "Architect": "Architecte",
    "Chef": "Chef cuisinier",
    "Driver": "Chauffeur",
    "Mechanic": "Mécanicien",
    "Pharmacist": "Pharmacien",
    "Psychologist": "Psychologue",
    "Dentist": "Dentiste",
    "Journalist": "Journaliste",
    "Designer": "Designer",
    "Artist": "Artiste",
    "Police Officer": "Policier",
    "Firefighter": "Pompier",
}


COUNTRY_LABELS_FR = {
    "France": "France",
    "United States": "États-Unis",
    "United Kingdom": "Royaume-Uni",
    "Canada": "Canada",
    "Germany": "Allemagne",
    "Spain": "Espagne",
    "Italy": "Italie",
    "Australia": "Australie",
    "India": "Inde",
    "Japan": "Japon",
    "China": "Chine",
    "Brazil": "Brésil",
    "Mexico": "Mexique",
    "Netherlands": "Pays-Bas",
    "Belgium": "Belgique",
    "Switzerland": "Suisse",
    "Sweden": "Suède",
    "Norway": "Norvège",
    "Denmark": "Danemark",
    "Ireland": "Irlande",
    "Portugal": "Portugal",
    "Poland": "Pologne",
    "Austria": "Autriche",
    "Greece": "Grèce",
    "Turkey": "Turquie",
    "South Africa": "Afrique du Sud",
    "Argentina": "Argentine",
    "Chile": "Chili",
    "Colombia": "Colombie",
    "South Korea": "Corée du Sud",
    "Singapore": "Singapour",
    "New Zealand": "Nouvelle-Zélande",
}


def country_display_label(value: str, lang: str) -> str:
    if lang == "FR":
        return COUNTRY_LABELS_FR.get(value, value)
    return value

NAP_LABELS = {
    "EN": {
        "No nap": "No nap",
        "Short nap": "Short nap",
        "Long nap": "Long nap",
    },
    "FR": {
        "No nap": "Pas de sieste",
        "Short nap": "Sieste courte",
        "Long nap": "Sieste longue",
    },
}


def occupation_display_label(value: str, lang: str) -> str:
    if lang == "FR":
        return OCCUPATION_LABELS_FR.get(value, value)
    return value


def nap_display_label(value: str, lang: str) -> str:
    return NAP_LABELS.get(lang, {}).get(value, value)


def make_display_options(raw_options: list[str], lang: str, kind: str) -> tuple[list[str], dict[str, str]]:
    labels = []
    reverse = {}
    counts = {}
    for raw in raw_options:
        if kind == "occupation":
            label = occupation_display_label(raw, lang)
        elif kind == "nap":
            label = nap_display_label(raw, lang)
        elif kind == "country":
            label = country_display_label(raw, lang)
        else:
            label = raw
        if label in reverse and reverse[label] != raw:
            counts[label] = counts.get(label, 1) + 1
            label = f"{label} ({raw})"
        reverse[label] = raw
        labels.append(label)
    return labels, reverse

st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #07101d 0%, #0b1524 100%);}
    .block-container {padding-top: 3rem; padding-bottom: 1.35rem; max-width: 1450px;}
    h1,h2,h3 {color:#f8fbff; letter-spacing:-0.02em;}
    p,label,div,span {color:#d7e3f2;}
    .hero {
        background: linear-gradient(135deg, rgba(76, 201, 240, 0.16), rgba(114, 9, 183, 0.14));
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.05rem 1.2rem;
        border-radius: 24px;
        margin-bottom: 0.8rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        min-height: 88px;
        display:flex; align-items:center;
    }
    .page-title {font-size: 3.35rem; font-weight: 800; line-height: 1.0; margin:0; color:#f8fbff;}
    .lang-wrap {display:flex; justify-content:flex-end; align-items:flex-start; padding-top:0.05rem;}
    .lang-wrap div[role="radiogroup"] {
        background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); padding:0.22rem;
        border-radius:999px; width:fit-content; gap:0.35rem;
    }
    .lang-wrap div[role="radiogroup"] > label {background:transparent; border:1px solid rgba(255,255,255,0.10); border-radius:999px; padding:0.18rem 0.8rem; min-width:78px; justify-content:center;}
    .lang-wrap div[role="radiogroup"] > label:has(input:checked) {background:rgba(255,255,255,0.09); border-color:rgba(255,255,255,0.18);}
    .header-card {
        background: rgba(255,255,255,0.035); border:1px solid rgba(255,255,255,0.08); border-radius:22px; padding:0.8rem 1rem; backdrop-filter: blur(6px);
    }
    .header-grid {display:grid; grid-template-columns: 1fr; align-items:center; min-height:54px;}
    .header-title {font-size: 1.75rem; font-weight: 780; color:#f8fbff; text-align:center;}
    .result-badge-wrap {display:flex; justify-content:center; margin:0.45rem 0 0.15rem 0;}
    .title-badge {justify-self:end; display:inline-flex; align-items:center; gap:0.55rem; padding:0.45rem 0.85rem; border-radius:999px; font-size:0.98rem; font-weight:720; color:#f8fbff; border:1px solid rgba(255,255,255,0.10);}
    .badge-dot {width:10px; height:10px; border-radius:999px; background:rgba(255,255,255,0.9); display:inline-block;}
    .section-gap {margin-top: 1.15rem;}
    .factor-section-gap {margin-top: 1.4rem;}
    .factor-column-title {font-size:2rem; font-weight:780; color:#f8fbff; margin: 0.55rem 0 0.95rem 0;}
    .section-title-large {font-size:3rem; font-weight:800; color:#f8fbff; margin:0 0 0.2rem 0;}
    .small-note {color:#a9bdd3; font-size:1.02rem; margin-top:0.15rem;}
    .factor-card {border-radius:20px; padding:1rem 1.1rem; margin-bottom:0.9rem; border:1px solid rgba(255,255,255,0.08); transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; box-shadow:0 8px 22px rgba(0,0,0,0.12); min-height:142px;}
    .factor-card:hover {transform: translateY(-2px); box-shadow:0 16px 30px rgba(0,0,0,0.22);}
    .factor-risk {background: rgba(230, 57, 70, 0.14); border-color: rgba(230, 57, 70, 0.30);}
    .factor-good {background: rgba(46, 204, 113, 0.14); border-color: rgba(46, 204, 113, 0.30);}
    .factor-top {display:flex; justify-content:space-between; align-items:center; gap:0.75rem; margin-bottom:0.2rem;}
    .factor-name {font-weight:760; font-size:1.1rem; color:#f8fbff;}
    .factor-val {color:#dbe6f3; font-size:1rem; margin-top:0.15rem;}
    .factor-meta {display:flex; justify-content:center; flex-wrap:wrap; gap:0.45rem; margin-top:0.75rem; min-height:34px; align-items:center;}
    .hint-chip {display:inline-flex; align-items:center; justify-content:center; padding:0.28rem 0.72rem; border-radius:999px; font-size:0.92rem; font-weight:690; color:#f8fbff; background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.10);}
    .impact-up {color:#ff9c9c; font-weight:800; letter-spacing:0.05em; font-size:1.08rem;}
    .impact-down {color:#90efc0; font-weight:800; letter-spacing:0.05em; font-size:1.08rem;}
    .stSlider {margin-bottom: 0.9rem;}
    div[data-baseweb="select"] > div {margin-bottom:0.65rem;}
    div[data-testid="stHorizontalBlock"] > div:has(> div > div > div[data-testid="stPlotlyChart"]) {align-self: start;}
    </style>
    """,
    unsafe_allow_html=True,
)


def tr(lang: str, key: str) -> str:
    return TEXT[lang][key]


def score_to_quality(raw_score: float) -> float:
    score = max(1.0, min(4.0, raw_score))
    return (4.0 - score) / 3.0 * 100.0


def quality_band(quality: float, lang: str) -> tuple[str, str]:
    if quality >= 80:
        return tr(lang, "healthy"), "rgba(65, 130, 86, 0.78)"
    if quality >= 60:
        return tr(lang, "fair"), "rgba(168, 128, 58, 0.82)"
    if quality >= 35:
        return tr(lang, "fragile"), "rgba(178, 107, 38, 0.82)"
    return tr(lang, "poor"), "rgba(158, 62, 72, 0.82)"


def color_mix_hex(color_a: str, color_b: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    a = tuple(int(color_a[i:i+2], 16) for i in (1, 3, 5))
    b = tuple(int(color_b[i:i+2], 16) for i in (1, 3, 5))
    mixed = tuple(round((1 - ratio) * x + ratio * y) for x, y in zip(a, b))
    return "#%02x%02x%02x" % mixed


def make_gauge(raw_score: float) -> go.Figure:
    quality = score_to_quality(raw_score)
    bar_color = color_mix_hex("#e63946", "#2ecc71", quality / 100.0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=quality,
        number={"suffix": "%", "font": {"size": 54}},
        gauge={
            "shape": "angular",
            "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "tickfont": {"size": 13}},
            "bar": {"color": bar_color, "thickness": 0.36},
            "steps": [
                {"range": [0, 25], "color": "rgba(230, 57, 70, 0.26)"},
                {"range": [25, 50], "color": "rgba(247, 127, 0, 0.20)"},
                {"range": [50, 75], "color": "rgba(247, 201, 72, 0.16)"},
                {"range": [75, 100], "color": "rgba(46, 204, 113, 0.22)"},
            ],
        },
    ))
    fig.update_layout(height=360, margin=dict(l=45, r=45, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)")
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
    arrow = "↓" if impact > 0 else "↑"
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
        "time_screen_before_sleep": {"FR": "Temps d'écran avant le coucher", "EN": "Screen time before bed"},
    }
    return labels.get(name, {}).get(lang, name)


def body_label_from_bmi(value: float, lang: str) -> str:
    options = BODY_TYPE_OPTIONS[lang]
    return min(options, key=lambda item: abs(item[1] - float(value)))[0]


def format_value(feature: str, value, lang: str) -> str:
    if feature in {"sleep_duration_hrs", "work_hours_that_day"}:
        return f"{float(value):.2f} h"
    if feature == "sleep_latency_mins":
        return f"{int(value)} min"
    if feature == "wake_episodes_per_night":
        return str(int(value))
    if feature == "stress_score":
        return f"{float(value):.1f}/10"
    if feature == "alcohol_units_before_bed":
        unit = "verre" if lang == "FR" else "drink"
        unit_pl = "verres" if lang == "FR" else "drinks"
        v = int(value)
        return f"{v} {unit if v == 1 else unit_pl}"
    if feature == "nb_cafe_before_bed":
        return COFFEE_LABELS[lang].get(int(value), str(value))
    if feature == "bmi":
        return body_label_from_bmi(float(value), lang)
    if feature in {"Anxiety", "Depression"}:
        return tr(lang, "yes") if int(value) == 1 else tr(lang, "no")
    if feature == "occupation":
        return occupation_display_label(str(value), lang)
    if feature == "country":
        return country_display_label(str(value), lang)
    if feature == "nap_time":
        return nap_display_label(str(value), lang)
    return str(value)


def factor_hint(feature: str, value, lang: str) -> str | None:
    val = float(value) if isinstance(value, (int, float)) else value
    if feature in {"sleep_duration_hrs"}:
        return tr(lang, "high_value") if float(val) >= 7 else tr(lang, "low_value")
    if feature in {"sleep_latency_mins", "wake_episodes_per_night", "work_hours_that_day"}:
        return tr(lang, "high_value") if float(val) >= (30 if feature == "sleep_latency_mins" else 3 if feature == "wake_episodes_per_night" else 9) else tr(lang, "low_value")
    if feature in {"stress_score", "alcohol_units_before_bed", "nb_cafe_before_bed"}:
        return tr(lang, "high_value") if float(val) >= (6 if feature == "stress_score" else 1) else tr(lang, "low_value")
    if feature in {"Anxiety", "Depression"} and int(val) == 1:
        return tr(lang, "present")
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


@st.cache_resource
def get_predictor():
    return load_predictor()


def input_panel(options: dict, lang: str) -> dict:
    country_options = options.get("country") or DEFAULT_COUNTRIES
    occupation_options = options.get("occupation") or DEFAULT_OCCUPATIONS
    nap_options = NAP_OPTIONS
    country_labels, country_reverse = make_display_options(country_options, lang, "country")
    occupation_labels, occupation_reverse = make_display_options(occupation_options, lang, "occupation")
    nap_labels, nap_reverse = make_display_options(nap_options, lang, "nap")
    body_options = BODY_TYPE_OPTIONS[lang]
    body_labels = [x[0] for x in body_options]
    default_body = body_label_from_bmi(float(FORM_DEFAULTS["bmi"]), lang)
    default_body_idx = body_labels.index(default_body) if default_body in body_labels else 0

    st.markdown(f"<div class='header-card'><div class='header-grid'><div></div><div class='header-title'>{html.escape(tr(lang, 'profile'))}</div><div></div></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)

    row0 = st.columns(3)
    with row0[0]:
        default_country_label = country_display_label(FORM_DEFAULTS["country"], lang)
        country_label = st.selectbox(tr(lang, "country"), country_labels, index=country_labels.index(default_country_label) if default_country_label in country_labels else 0)
        country = country_reverse[country_label]
    with row0[1]:
        age = st.slider(tr(lang, "age"), 18, 90, int(FORM_DEFAULTS["age"]))
    with row0[2]:
        wake_episodes_per_night = st.slider(tr(lang, "night_awakenings"), 0, 10, int(FORM_DEFAULTS["wake_episodes_per_night"]))
        
    row1 = st.columns(3)
    with row1[0]:
        occupation_label = st.selectbox(tr(lang, "occupation"), occupation_labels, index=occupation_labels.index(occupation_display_label(FORM_DEFAULTS["occupation"], lang)) if occupation_display_label(FORM_DEFAULTS["occupation"], lang) in occupation_labels else 0)
        occupation = occupation_reverse[occupation_label]
    with row1[1]:
        sleep_duration_hrs = st.slider(tr(lang, "sleep_duration"), 2.0, 12.0, float(FORM_DEFAULTS["sleep_duration_hrs"]), 0.25, format="%.2f h")
    with row1[2]:
        sleep_latency_mins = st.slider(tr(lang, "fall_asleep"), 0, 120, int(FORM_DEFAULTS["sleep_latency_mins"]), 5, format="%d min")

    row2 = st.columns(3)
    with row2[0]:
        body_label = st.selectbox(tr(lang, "body_type"), body_labels, index=default_body_idx)
    with row2[1]:
        stress_score = st.slider(tr(lang, "stress"), 0.0, 10.0, float(FORM_DEFAULTS["stress_score"]), 0.5, format="%.1f/10")
    with row2[2]:
        work_hours_that_day = st.slider(tr(lang, "work_hours"), 0.0, 16.0, float(FORM_DEFAULTS["work_hours_that_day"]), 0.25, format="%.2f h")

    row3 = st.columns(3)
    with row3[0]:
        default_nap_label = nap_display_label(FORM_DEFAULTS["nap_time"], lang)
        nap_label = st.selectbox(tr(lang, "nap"), nap_labels, index=nap_labels.index(default_nap_label) if default_nap_label in nap_labels else 0)
        nap_time = nap_reverse[nap_label]
    with row3[1]:
        nb_cafe_before_bed = st.slider(tr(lang, "coffee"), 0, 4, int(FORM_DEFAULTS["nb_cafe_before_bed"]), 1)
    with row3[2]:
        alcohol_units_before_bed = st.slider(tr(lang, "alcohol"), 0, 8, int(round(float(FORM_DEFAULTS["alcohol_units_before_bed"]))), 1)
    
    row4 = st.columns(3)
    with row4[0]:
        time_screen_before_sleep = st.selectbox(tr(lang, "screen"), SCREEN_TIME_OPTIONS, index=SCREEN_TIME_OPTIONS.index(FORM_DEFAULTS["time_screen_before_sleep"]))
    with row4[1]:
        anxiety = 1 if st.toggle(tr(lang, "anxiety"), value=bool(FORM_DEFAULTS["Anxiety"])) else 0
    with row4[2]:
        depression = 1 if st.toggle(tr(lang, "depression"), value=bool(FORM_DEFAULTS["Depression"])) else 0


    bmi = dict(body_options)[body_label]
    return {
        "country": country,
        "occupation": occupation,
        "bmi": bmi,
        "age": age,
        "sleep_duration_hrs": sleep_duration_hrs,
        "sleep_latency_mins": sleep_latency_mins,
        "wake_episodes_per_night": wake_episodes_per_night,
        "stress_score": stress_score,
        "work_hours_that_day": work_hours_that_day,
        "alcohol_units_before_bed": alcohol_units_before_bed,
        "nb_cafe_before_bed": nb_cafe_before_bed,
        "time_screen_before_sleep": time_screen_before_sleep,
        "nap_time": nap_time,
        "Anxiety": anxiety,
        "Depression": depression,
    }


def main() -> None:
    top_left, top_right = st.columns([0.84, 0.16], gap="small")
    with top_left:
        if "lang" not in st.session_state:
            st.session_state.lang = "FR"
        st.markdown(f"<div class='hero'><h1 class='page-title'>💤 {html.escape(TEXT[st.session_state.lang]['title'])}</h1></div>", unsafe_allow_html=True)
    with top_right:
        st.markdown("<div class='lang-wrap'>", unsafe_allow_html=True)
        lang = st.radio("Language", ["FR", "EN"], horizontal=False, label_visibility="collapsed", key="lang")
        st.markdown("</div>", unsafe_allow_html=True)

    predictor = get_predictor()
    metadata = predictor.metadata or {}
    category_options = metadata.get("category_options", {})

    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        payload = input_panel(category_options, lang)

    result = predictor.predict_dataframe(pd.DataFrame([payload]))
    quality = score_to_quality(result.raw_score)
    band_label, band_color = quality_band(quality, lang)
    badge_text = band_label

    with right:
        st.markdown(f"<div class='header-card'><div class='header-grid'><div class='header-title'>{html.escape(tr(lang, 'results'))}</div></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-badge-wrap'><div class='title-badge' style='background:{band_color}'><span class='badge-dot'></span>{html.escape(badge_text)}</div></div>", unsafe_allow_html=True)
        st.plotly_chart(make_gauge(result.raw_score), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div class='factor-section-gap'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='header-card'><div class='section-title-large'>{html.escape(tr(lang, 'why'))}</div><div class='small-note'>{html.escape(tr(lang, 'why_caption'))}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='factor-section-gap'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_factor_cards(tr(lang, "aggravating"), result.top_positive_factors, lang, positive=True)
    with c2:
        render_factor_cards(tr(lang, "beneficial"), result.top_negative_factors, lang, positive=False)


if __name__ == "__main__":
    main()
