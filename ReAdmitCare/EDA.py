import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "dataset/datasetpreprocessed.csv"
IMAGE_PATH = "static/images"

# Create image folder if it doesn't exist
os.makedirs(IMAGE_PATH, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("ReAdmitCare - Exploratory Data Analysis")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nReadmission distribution:")
print(df["readmitted"].value_counts())


# ---------------------------------------------------------
# 1. Readmission Distribution
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="readmitted"
)

plt.title("Hospital Readmission Distribution")
plt.xlabel("Readmitted")
plt.ylabel("Number of Patients")

plt.savefig(
    f"{IMAGE_PATH}/readmission_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 2. Age Distribution
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="age",
    order=df["age"].value_counts().index
)

plt.title("Patient Age Distribution")
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")

plt.xticks(rotation=45)

plt.savefig(
    f"{IMAGE_PATH}/age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 3. Time in Hospital
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="time_in_hospital",
    bins=14,
    kde=True
)

plt.title("Time Spent in Hospital")
plt.xlabel("Days in Hospital")
plt.ylabel("Number of Patients")

plt.savefig(
    f"{IMAGE_PATH}/time_in_hospital.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 4. Number of Medications
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="num_medications",
    bins=30,
    kde=True
)

plt.title("Number of Medications per Patient")
plt.xlabel("Number of Medications")
plt.ylabel("Number of Patients")

plt.savefig(
    f"{IMAGE_PATH}/medications_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 5. Diabetes Medication
# ---------------------------------------------------------
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="diabetesMed"
)

plt.title("Diabetes Medication Status")
plt.xlabel("Diabetes Medication")
plt.ylabel("Number of Patients")

plt.savefig(
    f"{IMAGE_PATH}/diabetes_medication.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 6. A1C Result
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="A1Cresult"
)

plt.title("A1C Test Results")
plt.xlabel("A1C Result")
plt.ylabel("Number of Patients")

plt.savefig(
    f"{IMAGE_PATH}/a1c_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 7. Readmission vs Time in Hospital
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="readmitted",
    y="time_in_hospital"
)

plt.title("Readmission vs Time in Hospital")
plt.xlabel("Readmitted")
plt.ylabel("Days in Hospital")

plt.savefig(
    f"{IMAGE_PATH}/readmission_vs_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nEDA completed successfully!")

print("\nGenerated images:")

for file in os.listdir(IMAGE_PATH):
    print("-", file)