import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_SEASON = BASE_DIR / "data" / "raw" / "season_based_crop.csv"
DATA_SOIL = BASE_DIR / "data" / "raw" / "Soil_type_based_Crop.csv"
DATA_RAIN = BASE_DIR / "data" / "raw" / "rain_based_crop.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MAX_SEASON_ROWS = 50000
MAX_TRAIN_ROWS = 100000


print("Loading data...")
try:
    df_season = pd.read_csv(DATA_SEASON)
    df_soil = pd.read_csv(DATA_SOIL)
    df_rain = pd.read_csv(DATA_RAIN)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()


def clean_text(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    return df

df_season = df_season.rename(columns={"State_Name": "State", "District_Name": "District", "Crop_Year": "Year"})
df_season = clean_text(df_season, ["State", "District", "Season", "Crop"])
df_season = df_season[df_season["Area"] > 0]
df_season["Yield"] = df_season["Production"] / df_season["Area"]


Q1 = df_season["Yield"].quantile(0.01)
Q3 = df_season["Yield"].quantile(0.99)
df_season = df_season[(df_season["Yield"] >= Q1) & (df_season["Yield"] <= Q3)]

if len(df_season) > MAX_SEASON_ROWS:
    df_season = df_season.sample(n=MAX_SEASON_ROWS, random_state=42)
    print(f"Sampled season data to {len(df_season)} rows before merge.")


soil_map = {
    "Crop Type": "Crop", "Crop_Name": "Crop", 
    "Soil Type": "Soil_Type", "soil_type": "Soil_Type",
    "Temparature": "Temperature", "temperature": "Temperature" 
}
df_soil = df_soil.rename(columns=soil_map)
df_soil = clean_text(df_soil, ["Crop", "Soil_Type"])



if "Temperature" in df_soil.columns:
    df_soil = df_soil[["Crop", "Soil_Type", "Temperature"]].drop_duplicates()
else:
    df_soil = df_soil[["Crop", "Soil_Type"]].drop_duplicates()
    df_soil["Temperature"] = 25.0 


rain_map = {"label": "Crop", "rainfall": "Rainfall"}
df_rain = df_rain.rename(columns=rain_map)
df_rain = clean_text(df_rain, ["Crop"])

df_rain_agg = df_rain.groupby("Crop")["Rainfall"].mean().reset_index()


print("Merging datasets...")


merged = df_season.merge(df_soil, on="Crop", how="inner")


merged = merged.merge(df_rain_agg, on="Crop", how="left")


merged["Rainfall"] = merged["Rainfall"].fillna(merged["Rainfall"].mean())

print(f"Final Dataset Shape: {merged.shape}")
print(f"Features: State, District, Season, Crop, Area, Soil_Type, Temperature, Rainfall")


if len(merged) > 100000:
    merged = merged.sample(n=MAX_TRAIN_ROWS, random_state=42)
    print(f"Sampled merged data to {len(merged)} rows for training.")


features = ["State", "District", "Season", "Crop", "Soil_Type", "Area", "Temperature", "Rainfall"]
target = "Yield"

X = merged[features]
y = merged[target]

print("Training model...")


cat_features = ["State", "District", "Season", "Crop", "Soil_Type"]
num_features = ["Area", "Temperature", "Rainfall"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")


joblib.dump(model, MODEL_DIR / "yield_model.pkl")


lists = {
    "soils": sorted(merged["Soil_Type"].unique()),
    "crops": sorted(merged["Crop"].unique()),
    "seasons": sorted(merged["Season"].unique())
}
joblib.dump(lists, MODEL_DIR / "app_lists.pkl")

print("Yield Model (with Rain/Soil/Temp) Saved!")
