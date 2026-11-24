# BLOCK 1: tools
from flask import Flask, render_template, request
import joblib
import pandas as pd

# BLOCK 2: start the app and load the model
app = Flask(__name__)
model = joblib.load("attrition_model.joblib")

# BLOCK 3: lists of form box names
TEXT_FIELDS   = ["OverTime", "JobRole", "MaritalStatus"]
NUMBER_FIELDS = ["JobLevel", "TotalWorkingYears", "YearsWithCurrManager",
                 "YearsSinceLastPromotion", "NumCompaniesWorked",
                 "EnvironmentSatisfaction", "JobSatisfaction"]

# BLOCK 4: what happens when someone visits the page
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        row = {}
        for name in TEXT_FIELDS:
            row[name] = request.form[name]
        for name in NUMBER_FIELDS:
            row[name] = float(request.form[name])

        chance = model.predict_proba(pd.DataFrame([row]))[0][1]

        if chance >= 0.5:
            result = f"HIGH RISK: {chance:.0%} chance of leaving"
        else:
            result = f"LOW RISK: {chance:.0%} chance of leaving"

    return render_template("index.html", result=result)

# BLOCK 5: start the server
if __name__ == "__main__":
    app.run(debug=True)