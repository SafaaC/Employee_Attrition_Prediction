# Employee_Attrition_Prediction

End-to-end ML project predicting employee turnover on the IBM HR dataset
(1,470 employees, 16% attrition).

## What's inside
- EDA and preprocessing
- Baseline models and PyCaret AutoML comparison
- SHAP/LIME explainability
- Flask web app for real-time predictions

## Key results
| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| (fill in from your holdout results) | | | | |

Top attrition drivers (from SHAP): 1. ... 2. ... 3. ...

## How to run
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000

## Project structure
- `attrition.ipynb`: analysis and modelling
- `app.py`, `templates/`: Flask app
- `attrition_model.pkl`: trained model

## Screenshots
(add images of your SHAP plots and the web app)