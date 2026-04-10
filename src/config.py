from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODELS_DIR / "lgbm_full.joblib"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"
SHAP_VALUES_PATH = MODELS_DIR / "shap_values.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"
TRAINING_DATA_PATH = DATA_DIR / "sleep_health_dataset.csv"

TARGET_COL = "y"
DISPLAY_TARGET_COL = "sleep_disorder_risk"

FEATURES = [
    "sleep_duration_hrs",
    "bmi",
    "sleep_latency_mins",
    "stress_score",
    "country",
    "occupation",
    "wake_episodes_per_night",
    "age",
    "work_hours_that_day",
    "Depression",
    "alcohol_units_before_bed",
    "Anxiety",
    "nap_time",
    "nb_cafe_before_bed",
    "time_screen_before_sleep",
]

NUMERIC_FEATURES = [
    "sleep_duration_hrs",
    "bmi",
    "sleep_latency_mins",
    "stress_score",
    "wake_episodes_per_night",
    "age",
    "work_hours_that_day",
    "Depression",
    "alcohol_units_before_bed",
    "Anxiety",
    "nb_cafe_before_bed",
]

CATEGORICAL_FEATURES = [
    "country",
    "occupation",
    "nap_time",
    "time_screen_before_sleep",
]

RAW_TO_MODEL_DOC = {
    "mental_health_condition": "Used to derive Depression and Anxiety flags.",
    "nap_duration_mins": "Bucketized into nap_time.",
    "screen_time_before_bed_mins": "Bucketized into time_screen_before_sleep.",
    "caffeine_mg_before_bed": "Bucketized into nb_cafe_before_bed.",
}

# Notebook-consistent score mapping.
RISK_LABELS = {
    1: "Healthy",
    2: "Mild risk",
    3: "Moderate risk",
    4: "Severe risk",
}

# Note: the notebook displayed Mild as '2. Moderate' and Moderate as '3. High'.
RISK_DESCRIPTIONS = {
    1: "Low estimated risk of sleep disorder",
    2: "Some concerning signals are present",
    3: "Several factors suggest elevated sleep risk",
    4: "Strong signal of high sleep-related risk",
}

DEFAULT_COUNTRIES = [
    "USA", "France", "Japan", "India", "Spain", "Germany", "UK", "Canada",
    "Italy", "Australia", "Brazil", "Mexico", "Netherlands", "Sweden", "South Korea",
]

DEFAULT_OCCUPATIONS = [
    "Software Engineer", "Driver", "Nurse", "Student", "Lawyer", "Teacher",
    "Doctor", "Accountant", "Salesperson", "Manager", "Scientist", "Artist",
]

NAP_OPTIONS = [
    "No nap", "15 minutes", "30 minutes", "45 minutes", "60 minutes", "More than 1 hour",
]

SCREEN_TIME_OPTIONS = [
    "Less than 30 minutes", "30-60 minutes", "1-2 hours", "2-3 hours", "More than 3 hours",
]

CAFFEINE_BUCKET_LABELS = {
    0: "0-40 mg",
    1: "41-100 mg",
    2: "101-200 mg",
    3: "> 200 mg",
}

FORM_DEFAULTS = {
    "sleep_duration_hrs": 6.0,
    "bmi": 24.0,
    "sleep_latency_mins": 40,
    "stress_score": 7.0,
    "country": "France",
    "occupation": "Manager",
    "wake_episodes_per_night": 1,
    "age": 40,
    "work_hours_that_day": 8.0,
    "Depression": 0,
    "alcohol_units_before_bed": 2.0,
    "Anxiety": 1,
    "nap_time": "No nap",
    "nb_cafe_before_bed": 1.0,
    "time_screen_before_sleep": "1-2 hours",
}

FEATURE_DIRECTIONS = {
    "sleep_duration_hrs": -1,
    "bmi": 1,
    "sleep_latency_mins": 1,
    "stress_score": 1,
    "country": 0,
    "occupation": 0,
    "wake_episodes_per_night": 1,
    "age": 1,
    "work_hours_that_day": 1,
    "Depression": 1,
    "alcohol_units_before_bed": 1,
    "Anxiety": 1,
    "nap_time": 1,
    "nb_cafe_before_bed": 1,
    "time_screen_before_sleep": 1,
}
