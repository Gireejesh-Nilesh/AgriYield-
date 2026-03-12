import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_SEASON = BASE_DIR / "data" / "raw" / "season_based_crop.csv"
DATA_SOIL = BASE_DIR / "data" / "raw" / "Soil_type_based_Crop.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


print("Loading data...")
try:
    df_season = pd.read_csv(DATA_SEASON)
    df_soil = pd.read_csv(DATA_SOIL) 
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()


print(f"Soil File Columns: {df_soil.columns.tolist()}")



column_map = {
    
    "Crop_Name": "Crop", "Crops": "Crop", "crop": "Crop",
    
    "Soil Type": "Soil_Type", "soil_type": "Soil_Type", "SOIL_TYPE": "Soil_Type"
}
df_soil = df_soil.rename(columns=column_map)


if "Crop" not in df_soil.columns:
    
    found = False
    for col in df_soil.columns:
        if "crop" in col.lower():
            df_soil = df_soil.rename(columns={col: "Crop"})
            found = True
            break
    if not found:
        print("CRITICAL ERROR: Could not find 'Crop' column in Soil CSV.")
        print("Please check the column names printed above.")
        exit()

if "Soil_Type" not in df_soil.columns:
     for col in df_soil.columns:
        if "soil" in col.lower():
            df_soil = df_soil.rename(columns={col: "Soil_Type"})
            break

print(f"Renamed Soil Columns: {df_soil.columns.tolist()}")


def clean(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    return df

df_season = df_season.rename(columns={"State_Name": "State", "District_Name": "District", "Crop_Year": "Year"})
df_season = clean(df_season, ["State", "District", "Season", "Crop"])
df_soil = clean(df_soil, ["Soil_Type", "Crop"])


print("Merging Soil data with Season data...")
merged = df_season.merge(df_soil, on="Crop", how="inner")
print(f"Merged Data Shape: {merged.shape}")

if merged.empty:
    print("WARNING: Merge resulted in empty dataset! Check if crop names match (e.g. 'Rice' vs 'RICE').")
    print(f"Season Crops: {df_season['Crop'].unique()[:5]}")
    print(f"Soil Crops: {df_soil['Crop'].unique()[:5]}")
    exit()


if len(merged) > 100000:
    merged = merged.sample(n=100000, random_state=42)


features = ["State", "District", "Season", "Soil_Type"]
target = "Crop"

X = merged[features]
y = merged[target]

print(f"Training Recommender on {X.shape} samples...")

categorical_features = ["State", "District", "Season", "Soil_Type"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Recommendation Accuracy: {acc:.4f}")


joblib.dump(model, MODEL_DIR / "crop_recommender.pkl")
soil_types = sorted(merged["Soil_Type"].unique())
joblib.dump(soil_types, MODEL_DIR / "soil_types_list.pkl")

print("Recommender saved successfully.")
