# Bank Fraud Detection — Streamlit App

## Files
- `app.py` — the Streamlit application
- `model1.pkl` — your trained Gradient Boosting pipeline
- `requirements.txt` — pinned dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a new GitHub repo and push these 3 files (`app.py`, `model1.pkl`, `requirements.txt`) to it.
2. Go to https://share.streamlit.io → **New app**.
3. Select your repo/branch and set **Main file path** to `app.py`.
4. Click **Deploy**. Your app will be live at a `*.streamlit.app` URL in a minute or two.

## Notes
- `model1.pkl` was trained with scikit-learn `1.6.1`. `requirements.txt` pins that
  exact version so the pipeline unpickles cleanly. `app.py` also includes a small
  compatibility shim as a safety net in case the deployment environment ends up
  on a slightly different sklearn version.
- The form auto-computes `High_Value_Flag` (True when transaction amount > 5 Million),
  exactly like the notebook's feature engineering step — you don't need to enter it manually.
- If you retrain the model with different features, just update the `TRANSACTION_TYPES`,
  `MERCHANT_CATEGORIES`, `LOCATIONS`, etc. lists and the `input_df` dictionary in `app.py`
  to match.
