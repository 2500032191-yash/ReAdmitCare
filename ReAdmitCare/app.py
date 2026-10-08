from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "dataset/datasetpreprocessed.csv"
CLASSIFICATION_MODEL_PATH = "models/improved_xgboost.pkl"
REGRESSION_MODEL_PATH = "models/hospital_stay_regressor.pkl"


# ============================================================
# START APPLICATION
# ============================================================

print("=" * 70)
print("Starting ReAdmitCare")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found: " + DATA_PATH
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# LOAD CLASSIFICATION MODEL
# ============================================================

if not os.path.exists(CLASSIFICATION_MODEL_PATH):
    raise FileNotFoundError(
        "Classification model not found: "
        + CLASSIFICATION_MODEL_PATH
    )

classification_model = joblib.load(
    CLASSIFICATION_MODEL_PATH
)

print("Classification model loaded.")


# ============================================================
# LOAD REGRESSION MODEL
# ============================================================

if not os.path.exists(REGRESSION_MODEL_PATH):
    raise FileNotFoundError(
        "Regression model not found: "
        + REGRESSION_MODEL_PATH
    )

regression_model = joblib.load(
    REGRESSION_MODEL_PATH
)

print("Regression model loaded.")


# ============================================================
# MODEL INPUT COLUMNS
# ============================================================

MODEL_COLUMNS = [
    column
    for column in df.columns
    if column != "readmitted"
]

print(
    "Number of model input columns:",
    len(MODEL_COLUMNS)
)


# ============================================================
# CREATE PATIENT DATA
# ============================================================

def create_patient_data(form):

    patient = {}

    # Fill every required column with
    # dataset median or mode
    for column in MODEL_COLUMNS:

        if pd.api.types.is_numeric_dtype(df[column]):

            patient[column] = df[column].median()

        else:

            mode_values = df[column].mode()

            if len(mode_values) > 0:
                patient[column] = mode_values.iloc[0]
            else:
                patient[column] = "Unknown"


    # --------------------------------------------------------
    # NUMERICAL INPUT FIELDS
    # --------------------------------------------------------

    numerical_fields = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses"
    ]


    # --------------------------------------------------------
    # CATEGORICAL INPUT FIELDS
    # --------------------------------------------------------

    categorical_fields = [
        "age",
        "gender",
        "race",
        "A1Cresult",
        "max_glu_serum",
        "diabetesMed",
        "change",
        "insulin"
    ]


    # --------------------------------------------------------
    # READ NUMERICAL FORM VALUES
    # --------------------------------------------------------

    for field in numerical_fields:

        value = form.get(field)

        if (
            value is not None
            and value != ""
            and field in patient
        ):

            try:

                patient[field] = float(value)

            except ValueError:

                pass


    # --------------------------------------------------------
    # READ CATEGORICAL FORM VALUES
    # --------------------------------------------------------

    for field in categorical_fields:

        value = form.get(field)

        if (
            value is not None
            and value != ""
            and field in patient
        ):

            patient[field] = value


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    patient_df = pd.DataFrame(
        [patient],
        columns=MODEL_COLUMNS
    )

    return patient_df


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    total_patients = len(df)

    readmitted = int(
        df["readmitted"].sum()
    )

    not_readmitted = (
        total_patients - readmitted
    )

    return render_template(
        "index.html",
        total_patients=total_patients,
        readmitted=readmitted,
        not_readmitted=not_readmitted
    )


# ============================================================
# EDA PAGE
# ============================================================

@app.route("/eda")
def eda():

    return render_template(
        "eda.html"
    )


# ============================================================
# PREPROCESSING PAGE
# ============================================================

@app.route("/preprocessing")
def preprocessing():

    return render_template(
        "preprocessing.html"
    )


# ============================================================
# REGRESSION PAGE
# ============================================================

@app.route("/regression")
def regression():

    return render_template(
        "regression.html"
    )


# ============================================================
# CLASSIFICATION PAGE
# ============================================================

@app.route("/classification")
def classification():

    return render_template(
        "classification.html"
    )


# ============================================================
# TREE MODELS PAGE
# ============================================================

@app.route("/tree-models")
def tree_models():

    return render_template(
        "tree_models.html"
    )


# ============================================================
# BOOSTING PAGE
# ============================================================

@app.route("/boosting")
def boosting():

    return render_template(
        "boosting.html"
    )


# ============================================================
# ADVANCED BOOSTING PAGE
# ============================================================

@app.route("/advanced-boosting")
def advanced_boosting():

    return render_template(
        "advanced_boosting.html"
    )


# ============================================================
# MODEL COMPARISON PAGE
# ============================================================

@app.route("/model-comparison")
def model_comparison():

    return render_template(
        "model_comparison.html"
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = None

    stay_prediction = None

    probability_percent = None

    risk_level = None


    if request.method == "POST":

        try:

            print("\n" + "=" * 70)
            print("NEW PATIENT PREDICTION")
            print("=" * 70)


            # Create patient data
            patient_df = create_patient_data(
                request.form
            )

            print(
                "Patient input shape:",
                patient_df.shape
            )


            # ------------------------------------------------
            # READMISSION PREDICTION
            # ------------------------------------------------

            probability = (
                classification_model
                .predict_proba(patient_df)[0][1]
            )

            probability_percent = round(
                probability * 100,
                2
            )


            # Determine risk
            if probability >= 0.5:

                risk_level = "HIGH"

                result = (
                    "High Risk of 30-Day Readmission"
                )

            else:

                risk_level = "LOW"

                result = (
                    "Low Risk of 30-Day Readmission"
                )


            print(
                "Readmission probability:",
                probability_percent,
                "%"
            )


            # ------------------------------------------------
            # HOSPITAL STAY PREDICTION
            # ------------------------------------------------

            stay_prediction = (
                regression_model
                .predict(patient_df)[0]
            )

            stay_prediction = round(
                float(stay_prediction),
                1
            )


            print(
                "Predicted hospital stay:",
                stay_prediction,
                "days"
            )

            print(
                "Prediction completed successfully."
            )


        except Exception as e:

            print("\nPrediction error:")

            print(e)

            result = (
                "Prediction Error: "
                + str(e)
            )

            stay_prediction = None


    return render_template(
        "predict.html",
        result=result,
        stay_prediction=stay_prediction,
        probability_percent=probability_percent,
        risk_level=risk_level
    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001
    )