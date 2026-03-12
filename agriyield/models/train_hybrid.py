import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import r2_score, mean_absolute_error


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_SEASON = BASE_DIR / "data" / "raw" / "season_based_crop.csv"
DATA_SOIL = BASE_DIR / "data" / "raw" / "Soil_type_based_Crop.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MAX_SEASON_ROWS = 25000
MAX_TRAIN_ROWS = 50000

print("1. Loading Data...")
df_season = pd.read_csv(DATA_SEASON)
df_soil = pd.read_csv(DATA_SOIL)

df_season = df_season.rename(columns={
    "State_Name": "State", "District_Name": "District", "Crop_Year": "Year", "Crop_Name": "Crop"
})
df_soil = df_soil.rename(columns={
    "Crop_Name": "Crop", "Crops": "Crop", "Crop Type": "Crop",
    "Soil Type": "Soil_Type", "soil_type": "Soil_Type"
})


for df in [df_season, df_soil]:
    for col in ["Crop", "Soil_Type", "State", "District", "Season"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

if len(df_season) > MAX_SEASON_ROWS:
    df_season = df_season.sample(MAX_SEASON_ROWS, random_state=42)
    print(f"Sampled season data to {len(df_season)} rows before merge.")


print("2. Merging Datasets...")
merged = df_season.merge(df_soil, on="Crop", how="inner")


merged = merged[merged["Area"] > 0]
merged["target_yield"] = merged["Production"] / merged["Area"]


merged = merged[merged["target_yield"] < merged["target_yield"].quantile(0.99)]


if "Rainfall" not in merged.columns:
    merged["rainfall"] = np.random.uniform(100, 300, size=len(merged)) 
if "Temperature" not in merged.columns:
    merged["temperature"] = np.random.uniform(20, 35, size=len(merged))
if "ndvi" not in merged.columns:
    merged["ndvi"] = np.random.uniform(0.1, 0.5, size=len(merged))


FEATURES = ["State", "District", "Season", "Crop", "Soil_Type", "Year", "Area", "rainfall", "temperature", "ndvi"]
TARGET = "target_yield"


if len(merged) > MAX_TRAIN_ROWS:
    merged = merged.sample(MAX_TRAIN_ROWS, random_state=42)
    print(f"Sampled merged data to {len(merged)} rows for training.")

X = merged[FEATURES]
y = merged[TARGET]

print(f"Training on {X.shape} samples with features: {FEATURES}")


cat_cols = ["State", "District", "Season", "Crop", "Soil_Type"]
num_cols = ["Year", "Area", "rainfall", "temperature", "ndvi"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)


X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Model 1: XGBoost
print("3. Training XGBoost...")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
xgb.fit(X_train, y_train)

# Model 2: CatBoost
print("4. Training CatBoost...")
cat = CatBoostRegressor(iterations=100, depth=7, learning_rate=0.1, verbose=0, random_state=42)
cat.fit(X_train, y_train)

# Model 3: LSTM
print("5. Training LSTM...")
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm = Sequential([
    LSTM(64, activation='relu', input_shape=(1, X_train.shape[1])),
    Dense(32, activation='relu'),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train_lstm, y_train, epochs=5, batch_size=32, verbose=0)


preds = (xgb.predict(X_test) * 0.4) + (cat.predict(X_test) * 0.4) + (lstm.predict(X_test_lstm).flatten() * 0.2)
print(f"Hybrid R2 Score: {r2_score(y_test, preds):.4f}")


print("6. Saving Models...")
joblib.dump(preprocessor, MODELS_DIR / "hybrid_preprocessor.pkl")
joblib.dump(xgb, MODELS_DIR / "hybrid_xgb.pkl")
joblib.dump(cat, MODELS_DIR / "hybrid_cat.pkl")
lstm.save(MODELS_DIR / "hybrid_lstm.keras")


joblib.dump(FEATURES, MODELS_DIR / "model_features.pkl")

print("Done! Restart app now.")
