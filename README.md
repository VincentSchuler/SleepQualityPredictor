# Sleep Risk Studio — Streamlit app from your notebook

This project turns your notebook into a deployable Streamlit application for **interactive sleep-risk prediction**.

## What was extracted from the notebook

### Final model logic
- **Model**: `lightgbm.LGBMRegressor`
- **Target**: `sleep_disorder_risk` mapped to ordinal values:
  - `Healthy -> 1`
  - `Mild -> 2`
  - `Moderate -> 3`
  - `Severe -> 4`
- **Inference rule**: predict a continuous score, then **round and clip to `[1, 4]`**

### Final features used in the notebook
- `sleep_duration_hrs`
- `bmi`
- `sleep_latency_mins`
- `stress_score`
- `country`
- `occupation`
- `wake_episodes_per_night`
- `age`
- `work_hours_that_day`
- `Depression`
- `alcohol_units_before_bed`
- `Anxiety`
- `nap_time`
- `nb_cafe_before_bed`
- `time_screen_before_sleep`

### Reused feature engineering
- `Anxiety` and `Depression` derived from `mental_health_condition`
- `nap_time` derived from `nap_duration_mins`
- `time_screen_before_sleep` derived from `screen_time_before_bed_mins`
- `nb_cafe_before_bed` derived from `caffeine_mg_before_bed`

## Important notebook inconsistency noticed
The notebook renames labels in a confusing way:
- `Mild` displayed as **`2. Moderate`**
- `Moderate` displayed as **`3. High`**

The app keeps the underlying prediction logic, but uses a clearer user-facing display:
- `1 = Healthy`
- `2 = Mild risk`
- `3 = Moderate risk`
- `4 = Severe risk`

## Project structure

```bash
sleep_streamlit_app/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── models/
└── src/
    ├── __init__.py
    ├── config.py
    ├── preprocess.py
    └── predict.py
```

## How the code is split

### `train.py`
For training / retraining from the raw CSV.
- reuses notebook feature engineering
- reuses notebook LightGBM hyperparameters
- runs stratified CV
- trains final full model
- saves:
  - `models/lgbm_full.joblib`
  - `models/metadata.joblib`
  - `models/shap_explainer.joblib` when possible

### `src/preprocess.py`
Contains notebook-derived feature engineering and inference preprocessing.

### `src/predict.py`
Loads artifacts, predicts a score, maps it to a risk class, and generates explanations.

### `app.py`
Streamlit UX layer:
- polished visual layout
- sliders / selectboxes / toggles
- gauge chart
- explanation cards
- input snapshot

## Local setup

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:
```bash
.venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

## Option A — reuse an already saved model
If you already exported your model from the notebook, place it here:

```bash
models/lgbm_full.joblib
```

Optional:
```bash
models/shap_explainer.joblib
models/metadata.joblib
```

Then launch:
```bash
streamlit run app.py
```

## Option B — retrain once from the raw CSV
If you have the raw training CSV locally:

```bash
python train.py --data path/to/sleep_health_dataset.csv --export-dir models
```

Then run:
```bash
streamlit run app.py
```

## Deployment

### Streamlit Community Cloud
1. Push the folder to GitHub.
2. In Streamlit Community Cloud, create a new app.
3. Select the repository.
4. Set the main file to:
```bash
app.py
```
5. Make sure the repository includes:
- `requirements.txt`
- the `models/` folder with your `.joblib` files if you do not want to retrain online

### Alternative deployment platforms
- Hugging Face Spaces
- Render
- Railway

## Recommended deployment workflow for portfolio use
For a portfolio demo, the cleanest setup is:
- train locally once
- save `lgbm_full.joblib`
- optionally save `shap_explainer.joblib`
- deploy only the lightweight inference app

This avoids rerunning the long notebook online.

## Commands summary

### Train
```bash
python train.py --data data/sleep_health_dataset.csv --export-dir models
```

### Run locally
```bash
streamlit run app.py
```

## Next practical step for you
1. Copy your real `.joblib` model into `models/`
2. Launch the app locally
3. Verify that the categorical values expected by the model match the form options
4. If needed, refine the UI labels and text for recruiter-facing presentation
