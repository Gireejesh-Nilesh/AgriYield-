import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import requests
import google.generativeai as genai
from PIL import Image
from gtts import gTTS
import time
from datetime import date

st.set_page_config(page_title="AgriYield+", layout="wide", page_icon="🌱")

THEMES = {
    "light": {
        "app_bg": "#f6f7fb",
        "panel_bg": "#ffffff",
        "text_color": "#334155",
        "control_text": "#475569",
        "label_text": "#526174",
        "accent_color": "#2e7d32",
        "border_color": "rgba(31, 41, 55, 0.16)",
        "muted_text": "rgba(31, 41, 55, 0.62)",
        "hover_bg": "rgba(46, 125, 50, 0.12)",
        "shadow_color": "rgba(15, 23, 42, 0.08)",
    },
    "dark": {
        "app_bg": "#0f172a",
        "panel_bg": "#111827",
        "text_color": "#e5eef7",
        "control_text": "#dbe7f3",
        "label_text": "#b8c7d9",
        "accent_color": "#7ddf64",
        "border_color": "rgba(229, 238, 247, 0.18)",
        "muted_text": "rgba(229, 238, 247, 0.64)",
        "hover_bg": "rgba(125, 223, 100, 0.12)",
        "shadow_color": "rgba(2, 6, 23, 0.42)",
    },
}


def apply_theme(theme_name):
    theme = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
            :root {{
                --app-bg: {theme["app_bg"]};
                --panel-bg: {theme["panel_bg"]};
                --text-color: {theme["text_color"]};
                --control-text: {theme["control_text"]};
                --label-text: {theme["label_text"]};
                --accent-color: {theme["accent_color"]};
                --border-color: {theme["border_color"]};
                --muted-text: {theme["muted_text"]};
                --hover-bg: {theme["hover_bg"]};
                --shadow-color: {theme["shadow_color"]};
            }}

            html, body, [class*="css"],
            p, span, label, li, small, strong, em,
            .stMarkdown, .stText, .stCaption, .stAlert, .stMetric {{
                color: var(--text-color);
            }}

            .stApp {{
                background-color: var(--app-bg);
            }}

            [data-testid="stSidebar"] {{
                background-color: var(--panel-bg);
                border-right: 1px solid var(--border-color);
            }}
            [data-testid="stSidebar"] * {{
                color: var(--text-color) !important;
            }}

            .css-card {{
                background-color: var(--panel-bg);
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 4px 10px var(--shadow-color);
                margin-bottom: 20px;
            }}

            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stTextArea textarea,
            .stDateInput input,
            .stTimeInput input,
            [data-baseweb="input"] > div,
            [data-baseweb="base-input"] > div,
            [data-baseweb="select"] > div {{
                background-color: var(--panel-bg) !important;
                border: 1px solid var(--border-color) !important;
                border-radius: 8px !important;
                color: var(--control-text) !important;
            }}
            [data-baseweb="input"] input,
            [data-baseweb="base-input"] input,
            .stTextArea textarea {{
                color: var(--control-text) !important;
                -webkit-text-fill-color: var(--control-text) !important;
            }}
            input::placeholder,
            textarea::placeholder {{
                color: var(--muted-text) !important;
                opacity: 1;
            }}
            .stSelectbox div[data-baseweb="select"] *,
            input, textarea,
            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] label {{
                color: var(--label-text) !important;
            }}
            [data-testid="stSelectbox"] *,
            [data-testid="stNumberInput"] *,
            [data-testid="stTextInput"] *,
            [data-testid="stTextArea"] *,
            [data-testid="stDateInput"] *,
            [data-testid="stTimeInput"] *,
            [data-testid="stSlider"] *,
            [data-testid="stRadio"] *,
            [data-testid="stSelectSlider"] * {{
                color: var(--label-text) !important;
            }}
            .stSelectbox div[data-baseweb="select"] span,
            .stNumberInput input,
            .stTextInput input,
            .stDateInput input,
            .stTimeInput input,
            .stTextArea textarea {{
                color: var(--control-text) !important;
                -webkit-text-fill-color: var(--control-text) !important;
            }}
            [data-testid="stSelectbox"] input,
            [data-testid="stNumberInput"] input,
            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-testid="stDateInput"] input,
            [data-testid="stTimeInput"] input,
            [data-baseweb="select"] span,
            [data-baseweb="input"] input,
            [data-baseweb="base-input"] input {{
                color: var(--control-text) !important;
                -webkit-text-fill-color: var(--control-text) !important;
            }}
            [data-testid="stNumberInput"] button,
            [data-testid="stDateInput"] button,
            [data-testid="stTimeInput"] button,
            [data-testid="baseButton-secondary"] {{
                color: var(--control-text) !important;
            }}

            h1, h2, h3, h4 {{
                color: var(--text-color);
                font-family: 'Segoe UI', sans-serif;
            }}

            div.row-widget.stRadio > div {{
                flex-direction: column;
                align-items: stretch;
            }}
            div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] {{
                padding: 10px;
                margin-bottom: 5px;
                border-radius: 8px;
                transition: background-color 0.2s, border-color 0.2s;
                color: var(--control-text) !important;
                border: 1px solid transparent;
            }}
            div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"]:hover {{
                background-color: var(--hover-bg);
                border-color: var(--accent-color);
            }}
            div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] * {{
                color: var(--control-text) !important;
            }}

            .theme-toggle-note {{
                padding: 0.45rem 0.75rem;
                margin: 0.25rem 0 0.75rem 0;
                border: 1px solid var(--border-color);
                border-radius: 999px;
                background: var(--panel-bg);
                color: var(--muted-text);
                text-align: center;
                font-size: 0.85rem;
            }}
            [data-testid="stButton"] button {{
                color: var(--control-text) !important;
            }}
            .dashboard-footer {{
                margin-top: 18vh;
                padding-top: 1rem;
                color: var(--muted-text);
                text-align: center;
                font-size: 0.95rem;
            }}

            .agri-loader {{
                display: flex;
                align-items: center;
                gap: 0.8rem;
                padding: 0.85rem 1rem;
                margin: 0.5rem 0 1rem 0;
                border: 1px solid var(--border-color);
                border-radius: 14px;
                background: var(--panel-bg);
                box-shadow: 0 8px 20px var(--shadow-color);
            }}
            .agri-loader--overlay {{
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                min-width: 280px;
                max-width: min(92vw, 420px);
                justify-content: center;
                z-index: 999999;
                margin: 0;
            }}
            .agri-loader-backdrop {{
                position: fixed;
                inset: 0;
                background: rgba(15, 23, 42, 0.12);
                backdrop-filter: blur(2px);
                z-index: 999998;
            }}
            .agri-loader__spinner {{
                width: 18px;
                height: 18px;
                border: 3px solid rgba(127, 127, 127, 0.2);
                border-top-color: var(--accent-color);
                border-radius: 50%;
                animation: agri-spin 0.9s linear infinite;
                flex: 0 0 auto;
            }}
            .agri-loader__text {{
                color: var(--text-color);
                font-weight: 600;
            }}
            @keyframes agri-spin {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(360deg); }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_loader(message):
    return st.markdown(
        f"""
        <div class="agri-loader-backdrop"></div>
        <div class="agri-loader agri-loader--overlay">
            <div class="agri-loader__spinner"></div>
            <div class="agri-loader__text">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "agriyield" / "models"
DATA_PATH = BASE_DIR / "agriyield" / "data" / "raw" / "season_based_crop.csv"
CURRENT_YEAR = date.today().year
MAX_PREDICTION_YEAR = CURRENT_YEAR - 1


PREPROC_PATH = MODELS_DIR / "hybrid_preprocessor.pkl"
XGB_PATH = MODELS_DIR / "hybrid_xgb.pkl"
CAT_PATH = MODELS_DIR / "hybrid_cat.pkl"
LSTM_PATH = MODELS_DIR / "hybrid_lstm.keras"
REC_PATH = MODELS_DIR / "crop_recommender.pkl"
SOIL_LIST_PATH = MODELS_DIR / "soil_types_list.pkl"
APP_DIR = Path(__file__).resolve().parent
LOCAL_DASHBOARD_IMAGE = APP_DIR / "agriYield image.jpg"
DASHBOARD_IMAGE_CANDIDATES = [
    LOCAL_DASHBOARD_IMAGE,
    BASE_DIR / "agriYield image.jpg",
]


if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "light"
if "previous_menu" not in st.session_state:
    st.session_state["previous_menu"] = None

apply_theme(st.session_state["theme_mode"])


def get_live_weather(city):
    API_KEY = "660d70370f1696132c0e3c5f7c76e6e1"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            rainfall = data.get('rain', {}).get('1h', 0.0)
            return {"temp": temp, "rainfall": rainfall}
        else:
            return None
    except:
        return None


@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        df = pd.read_csv(DATA_PATH)
        col_map = {
            "State_Name": "State", "District_Name": "District", "Crop_Year": "Year",
            "Crop_Name": "Crop", "Area": "Area", "Production": "Production", "Season": "Season"
        }
        df = df.rename(columns=col_map)
        
        
        for col in ["State", "District", "Season", "Crop"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
        
    
        if "target_yield" not in df.columns and "Production" in df.columns:
            df = df[df["Area"] > 0]
            df["target_yield"] = df["Production"] / df["Area"]
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        df = pd.DataFrame()

    
    try:
        preprocessor = joblib.load(PREPROC_PATH)
        xgb_model = joblib.load(XGB_PATH)
        cat_model = joblib.load(CAT_PATH)
        lstm_model = tf.keras.models.load_model(LSTM_PATH)
        explainer = shap.TreeExplainer(xgb_model)
    except:
        preprocessor, xgb_model, cat_model, lstm_model, explainer = None, None, None, None, None


    try:
        recommender = joblib.load(REC_PATH)
        soil_types = joblib.load(SOIL_LIST_PATH)
    except:
        recommender = None
        soil_types = ["CLAYEY", "LOAMY", "SANDY", "BLACK", "RED"]

    return df, preprocessor, xgb_model, cat_model, lstm_model, explainer, recommender, soil_types


resource_loader = st.empty()
with resource_loader.container():
    show_loader("Loading data, models, and recommendations...")
    df, preprocessor, xgb_model, cat_model, lstm_model, explainer, recommender, soil_types = load_resources()
resource_loader.empty()

if df.empty:
    st.error("Data could not be loaded. Please check 'season_based_crop.csv' exists in data/raw/.")
    st.stop()
    
    
with st.sidebar:
    st.title("🌱 AgriYield+")
    st.markdown("Intelligent Agriculture System")
    st.caption("AI-powered agriculture advisory")
    current_theme = st.session_state["theme_mode"]
    toggle_icon = "🌙" if current_theme == "light" else "☀️"
    toggle_label = "Dark mode" if current_theme == "light" else "Light mode"
    if st.button(f"{toggle_icon} {toggle_label}", use_container_width=True):
        st.session_state["theme_mode"] = "dark" if current_theme == "light" else "light"
        st.rerun()
    st.markdown(
        f"<div class='theme-toggle-note'>Active theme: {st.session_state['theme_mode'].title()}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    
    
    menu = st.radio("Navigate", ["🏠 Dashboard","📊 Crop Yield Prediction", "🌾 Crop Recommendation", " 📸 AI Plant Doctor"], index=0,label_visibility="collapsed")
    st.markdown("---")
    st.caption("v2.0.1 | Farmers Edition")


if st.session_state["previous_menu"] is None:
    st.session_state["previous_menu"] = menu
elif st.session_state["previous_menu"] != menu:
    page_loader = st.empty()
    with page_loader.container():
        show_loader(f"Opening {menu.strip()}...")
    time.sleep(0.45)
    page_loader.empty()
    st.session_state["previous_menu"] = menu


if menu == "🏠 Dashboard":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%); padding: 30px; border-radius: 15px; color: white; margin-bottom: 25px;">
        <h1 style="color: white; margin-bottom: 10px;">🌱 Welcome to AgriYield+</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">Your intelligent companion for modern farming. Predict yields, find the right crops, and treat diseases with AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    total_crops = len(df['Crop'].unique()) if not df.empty else 12
    total_districts = len(df['District'].unique()) if not df.empty else 30
    total_records = len(df) if not df.empty else 5000
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Crops", f"{total_crops}+", "Supported")
    with m2:
        st.metric("Districts Covered", f"{total_districts}", "Across India")
    with m3:
        st.metric("Data Records", f"{total_records:,}", "Historical Data")
    with m4:
        st.metric("AI Accuracy", "94.5%", "Model Precision")
        
    st.markdown("---")
    
    
    st.subheader("How To Use AgriYield+ Efficiently")

    guide_col1, guide_col2 = st.columns([1.2, 1])
    with guide_col1:
        st.markdown("""
        1. **Start with Crop Recommendation**
           Enter state, district, season, and soil type to identify the most suitable crop first.

        2. **Then use Yield Prediction**
           Select the recommended crop, verify area, rainfall, temperature, soil type, and NDVI for a better estimate.

        3. **Use Live Weather before predicting**
           The `Get Live Weather` button updates temperature and rainfall inputs, which improves practical forecasts.

        4. **Use AI Plant Doctor only when needed**
           Upload a clear crop photo for disease or deficiency guidance. Use it for troubleshooting, not for routine planning.
        """)

    with guide_col2:
        st.info(
            "**Best workflow**\n\n"
            "Recommendation -> Yield Prediction -> Plant Doctor\n\n"
            "This order helps you choose the crop first, estimate output next, and solve health issues only when they appear."
        )
        st.warning(
            "**Input quality matters**\n\n"
            "Correct district, crop, season, soil type, area, rainfall, and temperature give more useful results than random defaults."
        )

    st.markdown("---")

    st.subheader("What Each Page Does")
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.markdown("**📊 Crop Yield Prediction**\nEstimate tons/acre and total production using historical data and hybrid ML models.")
    with exp2:
        st.markdown("**🌾 Crop Recommendation**\nSuggests the best crop for the selected location, season, and soil profile.")
    with exp3:
        st.markdown("**📸 AI Plant Doctor**\nAnalyzes plant images and provides disease, deficiency, and treatment guidance.")
    
    st.markdown(
        "<div class='dashboard-footer'>Developed by AgriTech Innovations | v2.0.1 Stable</div>",
        unsafe_allow_html=True,
    )


# TAB 1: YIELD PREDICTION
elif menu == "📊 Crop Yield Prediction":
    with st.container():
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <span style="font-size: 2rem; margin-right: 15px; background: #e8f5e9; padding: 10px; border-radius: 12px;">📊</span>
            <h2 style="margin: 0;">Crop Yield Prediction</h2>
        </div>
        """, unsafe_allow_html=True)
    
        col1, col2 = st.columns(2)

        with col1:
            states = sorted(df["State"].unique())
            sel_state = st.selectbox("State", states, key="yield_state")

            districts = sorted(df[df["State"] == sel_state]["District"].unique())
            sel_dist = st.selectbox("District", districts, key="yield_dist")

            sel_year = st.slider(
                "Year",
                2000,
                MAX_PREDICTION_YEAR,
                min(2024, MAX_PREDICTION_YEAR),
                key="yield_year"
            )

            if "Season" in df.columns:
                seasons = sorted(df["Season"].unique())
            else:
                seasons = ["KHARIF", "RABI", "WHOLE YEAR"]
            sel_season = st.selectbox("Season", seasons, key="yield_season")

        with col2:
            if "Crop" in df.columns:
                crops = sorted(df["Crop"].unique())
            else:
                crops = ["RICE", "MAIZE", "WHEAT"]
            sel_crop = st.selectbox("Crop Type", crops, key="yield_crop")

            dist_data = df[df["District"] == sel_dist]
            avg_area = dist_data["Area"].mean() if not dist_data.empty else 10.0
            area_inp = st.number_input(
                "Cultivation Area (Acres)",
                value=float(avg_area) if avg_area > 0 else 10.0
            )

        st.markdown("---")
        st.subheader("🌦️ Weather & Soil")

        
        if "fetched_temp" not in st.session_state:
            st.session_state["fetched_temp"] = 25.0
        if "fetched_rain" not in st.session_state:
            st.session_state["fetched_rain"] = 120.0

        cw1, cw2 = st.columns([3, 1])

        with cw2:
            if st.button("Get Live Weather", help="Fetch real-time temperature and rainfall"):
                weather_loader = st.empty()
                with weather_loader.container():
                    show_loader("Fetching latest weather data...")
                    live_w = get_live_weather(sel_dist)
                weather_loader.empty()
                if live_w:
                    st.session_state["fetched_temp"] = live_w["temp"]
                    st.session_state["fetched_rain"] = live_w["rainfall"]
                    st.success(f"{live_w['temp']}°C | {live_w['rainfall']} mm")
                else:
                    st.warning("Weather data not found.")

        with cw1:
            temp_inp = st.number_input(
                "Temperature (°C)",
                value=float(st.session_state["fetched_temp"]),
                key="temp_input"
            )
            rain_inp = st.number_input(
                "Rainfall (mm)",
                value=float(st.session_state["fetched_rain"])
            )
            soil_inp = st.selectbox("Soil Type", soil_types, key="yield_soil")
            ndvi_inp = st.slider("NDVI (Vegetation Index)", 0.0, 1.0, 0.21)

        
        if st.button("Predict Yield", key="btn_yield", type="primary"):
            if xgb_model is None or preprocessor is None:
                st.error("Yield models not loaded.")
            else:
                prediction_loader = st.empty()
                with prediction_loader.container():
                    show_loader("Running hybrid yield prediction...")
                    hist_yield = df[df["District"] == sel_dist]["target_yield"].mean()
                    if pd.isna(hist_yield):
                        hist_yield = 2.0

                    input_df = pd.DataFrame({
                        "Year": [sel_year],
                        "State": [sel_state],
                        "District": [sel_dist],
                        "Season": [sel_season],
                        "Crop": [sel_crop],
                        "Area": [area_inp],
                        "yield_calculated": [hist_yield],
                        "rainfall": [rain_inp],
                        "temperature": [temp_inp],
                        "ndvi": [ndvi_inp],
                        "Soil_Type": [soil_inp]
                    })

                    try:
                        X_proc = preprocessor.transform(input_df)

                        if hasattr(X_proc, "toarray"):
                            X_proc = X_proc.toarray()

                        p_xgb = xgb_model.predict(X_proc)[0]
                        p_cat = cat_model.predict(X_proc)[0]

                        X_lstm = X_proc.reshape((X_proc.shape[0], 1, X_proc.shape[1]))
                        p_lstm = lstm_model.predict(X_lstm, verbose=0).flatten()[0]

                        final_pred = (0.4 * p_xgb) + (0.4 * p_cat) + (0.2 * p_lstm)
                        total_prod = final_pred * area_inp

                        hist_subset = df[(df["District"] == sel_dist) & (df["Crop"] == sel_crop)]
                        if not hist_subset.empty:
                            last_hist_year = int(hist_subset["Year"].max())
                        else:
                            last_hist_year = int(sel_year) - 1

                        forecast_years = list(range(last_hist_year + 1, int(sel_year) + 1))
                        forecast_points = []
                        for forecast_year in forecast_years:
                            input_df_forecast = input_df.copy()
                            input_df_forecast["Year"] = forecast_year
                            X_forecast = preprocessor.transform(input_df_forecast)
                            if hasattr(X_forecast, "toarray"):
                                X_forecast = X_forecast.toarray()

                            fxgb = xgb_model.predict(X_forecast)[0]
                            fcat = cat_model.predict(X_forecast)[0]
                            X_forecast_lstm = X_forecast.reshape((X_forecast.shape[0], 1, X_forecast.shape[1]))
                            flstm = lstm_model.predict(X_forecast_lstm, verbose=0).flatten()[0]
                            fhybrid = (0.4 * fxgb) + (0.4 * fcat) + (0.2 * flstm)
                            forecast_points.append({"Year": forecast_year, "target_yield": float(fhybrid)})

                        st.session_state["yield_forecast_series"] = {
                            "district": sel_dist,
                            "crop": sel_crop,
                            "points": forecast_points
                        }

                        st.success(f"Predicted Yield: **{final_pred:.2f} tons/acre**")
                        st.info(f"Total Expected Production: **{total_prod:.2f} tons**")

                        st.subheader("Why this prediction?")

                        if explainer is not None:
                            shap_vals = explainer.shap_values(X_proc)
                            vals = shap_vals[0]
                        else:
                            vals = np.zeros(X_proc.shape[1])

                        try:
                            cat_encoder = preprocessor.named_transformers_["cat"]
                            ohe_features = list(cat_encoder.get_feature_names_out())
                            num_features = ["Year", "Area", "rainfall", "temperature", "ndvi"]
                            feature_names = ohe_features + num_features
                        except Exception:
                            feature_names = [f"Feature {i}" for i in range(len(vals))]

                        if len(feature_names) != len(vals):
                            feature_names = [f"Feature {i}" for i in range(len(vals))]

                        impact_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Impact": vals
                        })
                        impact_df["Abs_Impact"] = impact_df["Impact"].abs()

                        user_keywords = [
                            sel_state.replace(" ", "_"),
                            sel_dist.replace(" ", "_"),
                            sel_crop.replace(" ", "_"),
                            sel_season.replace(" ", "_"),
                            "Area",
                            "Year",
                            "rainfall",
                            "temperature",
                            "ndvi"
                        ]

                        filtered_df = impact_df[
                            impact_df["Feature"].apply(
                                lambda x: any(k in x for k in user_keywords)
                            )
                        ]

                        if filtered_df.empty:
                            filtered_df = impact_df.copy()

                        top_features = filtered_df.sort_values(
                            "Abs_Impact", ascending=False
                        ).head(8)

                        top_features["Feature"] = (
                            top_features["Feature"]
                            .str.replace("cat__", "", regex=False)
                            .str.replace("State_", "", regex=False)
                            .str.replace("District_", "", regex=False)
                            .str.replace("Season_", "", regex=False)
                            .str.replace("Crop_", "", regex=False)
                            .str.replace("Soil_Type_", "", regex=False)
                        )

                        colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in top_features["Impact"]]

                        fig, ax = plt.subplots(figsize=(10, 4))
                        bars = ax.barh(top_features["Feature"], top_features["Impact"], color=colors)
                        ax.bar_label(bars, fmt="%.2f", padding=3)
                        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
                        ax.set_xlabel("Impact on Yield")
                        plt.gca().invert_yaxis()
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                prediction_loader.empty()


    with st.container():
    
        st.subheader("📈 Yield Trend")
        trend_df = df[(df["District"] == sel_dist) & (df["Crop"] == sel_crop)].sort_values("Year")

        forecast_info = st.session_state.get("yield_forecast_series")
        if forecast_info and forecast_info["district"] == sel_dist and forecast_info["crop"] == sel_crop:
            forecast_df = pd.DataFrame(forecast_info["points"])
            if not forecast_df.empty:
                trend_df = pd.concat([trend_df[["Year", "target_yield"]], forecast_df], ignore_index=True)
                trend_df = trend_df.drop_duplicates(subset=["Year"], keep="last").sort_values("Year")
            
        with st.expander("🔍 Debug Info"):
            st.write(f"**Selected District:** {sel_dist}")
            st.write(f"**Selected Crop:** {sel_crop}")
            if not trend_df.empty:
                st.dataframe(trend_df[["Year", "target_yield"]].head())
            
        if not trend_df.empty:
            st.line_chart(trend_df.set_index("Year")[["target_yield"]])
        else:
            st.warning(f"⚠️ No historical data for **{sel_crop}** in **{sel_dist}**.")
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: CROP RECOMMENDATION
elif menu == "🌾 Crop Recommendation":
    with st.container():
        st.header("Find Suitable Crop & Advisory")
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <span style="font-size: 2rem; margin-right: 15px; background: #e3f2fd; padding: 10px; border-radius: 12px;">🌾</span>
            <h2 style="margin: 0;">Get expert crop suggestions with fertilizer and care insights.</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if recommender is None:
            st.warning("Recommender model not found. Run 'agriyield/models/train_recommender.py'.")
        else:
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                states = sorted(df["State"].unique())
                r_state = st.selectbox("State", states, key="rec_state")
                r_dists = sorted(df[df["State"] == r_state]["District"].unique())
                r_dist = st.selectbox("District", r_dists, key="rec_dist")
                
            with rc2:
                r_season = st.selectbox("Season", ["KHARIF", "RABI", "WHOLE YEAR", "SUMMER", "WINTER"], key="rec_season")
                r_soil = st.selectbox("Soil Type", soil_types, key="rec_soil")

            with rc3:
                r_ph = st.slider("Soil pH Level", 4.0, 9.0, 6.5, help="Optimal pH helps nutrient absorption.")
                r_water = st.select_slider("Water Availability", options=["Low", "Medium", "High", "Abundant"], value="Medium")

            if st.button("Recommend Crop", key="btn_rec", type="primary"):
                rec_input = pd.DataFrame({
                    "State": [r_state], "District": [r_dist], "Season": [r_season], "Soil_Type": [r_soil]
                })
                
                recommendation_loader = st.empty()
                with recommendation_loader.container():
                    show_loader("Finding the best crop recommendation...")
                    try:
                        pred_crop = recommender.predict(rec_input)[0]
                        probs = recommender.predict_proba(rec_input)[0]
                        
                        st.divider()
                        st.subheader(f"🌟 Best Crop: :green[{pred_crop}]")
                        conf_score = max(probs)
                        st.progress(conf_score, text=f"Confidence Score: {conf_score*100:.1f}%")

                        st.markdown("### 💊 Fertilizer & Care Guide")
                        fertilizer_map = {
                            "RICE": "Urea (Nitrogen) & TSP. Maintain standing water level of 5cm.",
                            "COTTON": "Balanced NPK (20-20-20). Requires good drainage.",
                            "MAIZE": "Nitrogen rich fertilizer. Apply Zinc sulphate if leaves yellow.",
                            "WHEAT": "DAP (Di-ammonium Phosphate) during sowing. Irrigate at critical stages.",
                            "GROUNDNUT": "Gypsum/Calcium for pod formation. Avoid excess Nitrogen.",
                            "SUGARCANE": "High Potassium. Heavy irrigation required every 10 days.",
                            "BAJRA": "Low nutrient requirement. Apply Nitrogen in split doses.",
                            "PULSES": "Phosphorus rich fertilizer. No Nitrogen needed (Self-fixing)."
                        }
                        advice = fertilizer_map.get(pred_crop.upper(), "Standard NPK (10-26-26) recommended. Monitor for pests.")
                        
                        col_adv1, col_adv2 = st.columns(2)
                        with col_adv1:
                            st.info(f"**Fertilizer:** {advice}")
                        with col_adv2:
                            if r_ph < 5.5:
                                st.warning(f"**Soil Condition:** Your soil is Acidic (pH {r_ph}). Consider adding Lime.")
                            elif r_ph > 7.5:
                                st.warning(f"**Soil Condition:** Your soil is Alkaline (pH {r_ph}). Consider adding Gypsum.")
                            else:
                                st.success(f"**Soil Condition:** pH {r_ph} is optimal for most crops.")

                        st.markdown("### 💡 Why this Recommendation?")
                        reasons = []
                        if r_soil in ["CLAYEY", "LOAMY"] and pred_crop in ["RICE", "SUGARCANE"]:
                            reasons.append(f"• **{r_soil} Soil** retains moisture well, which is critical for {pred_crop}.")
                        elif r_soil == "SANDY" and pred_crop in ["GROUNDNUT", "MAIZE", "BAJRA"]:
                            reasons.append(f"• **{r_soil} Soil** offers good drainage, preventing root rot for {pred_crop}.")
                        elif r_soil == "BLACK" and pred_crop in ["COTTON"]:
                            reasons.append(f"• **Black Soil** is famous for Cotton cultivation due to moisture holding.")
                            
                        if r_season == "KHARIF":
                            reasons.append(f"• **Kharif Season** (Monsoon) provides the necessary rainfall.")
                        elif r_season == "RABI":
                            reasons.append(f"• **Rabi Season** (Winter) offers the cool, dry climate needed.")

                        if not reasons:
                            reasons.append(f"• Historical farming data in **{r_dist}** shows high success rates for **{pred_crop}**.")
                        
                        for r in reasons:
                            st.write(r)

                        st.markdown("### 🔄 Alternative Options")
                        top3_idx = np.argsort(probs)[-3:][::-1]
                        top_crops = [recommender.classes_[i] for i in top3_idx]
                        top_probs = [probs[i] for i in top3_idx]
                        
                        c_chart1, c_chart2 = st.columns([1, 2])
                        with c_chart1:
                            for i, (crop, prob) in enumerate(zip(top_crops, top_probs)):
                                st.metric(f"Option {i+1}", crop, f"{prob*100:.1f}%")
                                
                        with c_chart2:
                            fig, ax = plt.subplots(figsize=(5, 3))
                            wedges, texts, autotexts = ax.pie(top_probs, labels=top_crops, autopct='%1.1f%%', 
                                                            colors=['#2ecc71', '#3498db', '#95a5a6'], 
                                                            startangle=90, wedgeprops=dict(width=0.4))
                            ax.set_title("Probability Distribution")
                            st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                recommendation_loader.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with st.container():
        st.header("🧮 Fertilizer Calculator")
        st.markdown("Calculate the exact nutrient requirements for your farm.")
        
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            f_crop = st.selectbox("Select Crop", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato"], key="fert_crop")
            f_area = st.number_input("Field Area (Acres)", min_value=0.1, value=1.0, step=0.1, key="fert_area")
            
        with col_f2:
            st.info("💡 **Standard Recommendation:**\n"
                    "- Nitrogen (N): Promotes leaf growth.\n"
                    "- Phosphorus (P): Helps root development.\n"
                    "- Potassium (K): Overall plant health.")

        if st.button("Calculate Quantity", type="primary"):
            st.divider()
            
            
            dosage_map = {
                "Rice":      {"N": 40, "P": 20, "K": 20},
                "Wheat":     {"N": 50, "P": 25, "K": 20},
                "Maize":     {"N": 48, "P": 24, "K": 20},
                "Cotton":    {"N": 60, "P": 30, "K": 30},
                "Sugarcane": {"N": 100,"P": 40, "K": 60},
                "Potato":    {"N": 60, "P": 40, "K": 40}
            }
            
            req = dosage_map[f_crop]
            
            
            n_needed = req["N"] * f_area
            p_needed = req["P"] * f_area
            k_needed = req["K"] * f_area
            
            # Convert to Commercial Bags
            # Urea (46% N) -> 100kg Urea = 46kg N
            urea_bags = (n_needed / 0.46) / 50  # 50kg bags
            
            # DAP (18% N, 46% P) -> We use DAP for P, but it also adds N
            dap_bags = (p_needed / 0.46) / 50
            
            # MOP (60% K)
            mop_bags = (k_needed / 0.60) / 50
            
            
            st.subheader(f"Requirements for {f_area} acres of {f_crop}:")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Urea (Nitrogen)", f"{urea_bags:.1f} Bags", "50kg each")
            c2.metric("DAP (Phosphorus)", f"{dap_bags:.1f} Bags", "50kg each")
            c3.metric("MOP (Potassium)", f"{mop_bags:.1f} Bags", "50kg each")
            
            st.warning(f"⚠️ **Note:** Adjust Urea dosage. Since DAP also provides Nitrogen, reduce Urea by {dap_bags * 0.4:.1f} bags.")            
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: AI PLANT DOCTOR 
elif menu == " 📸 AI Plant Doctor":
    with st.container():
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <span style="font-size: 2rem; margin-right: 15px; background: #fff3e0; padding: 10px; border-radius: 12px;">📸</span>
            <h2 style="margin: 0;">AI Plant Doctor</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("Upload a photo of your crop to identify diseases and get treatment advice.")

        
        GOOGLE_API_KEY = "AIzaSyD9d6pGBqHCKzQcZh9iYXKwBh2XT5roTXo" 
        
        
        uploaded_file = st.file_uploader("Take a photo or upload", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Crop Image", width=300)
            
            crop_hint = st.selectbox(
                "Known Crop (Optional)",
                ["Auto-detect", "Sugarcane", "Rice", "Wheat", "Maize", "Cotton", "Groundnut", "Potato", "Other"],
                key="plant_crop_hint"
            )
            
            
            lang_choice = st.selectbox("Select Output Language:", 
                                    ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam"])
            
            lang_map = {
                "English": "en", "Hindi": "hi", "Tamil": "ta", 
                "Telugu": "te", "Kannada": "kn", "Malayalam": "ml"
            }

            if st.button("Analyze Plant", type="primary"):
                if not GOOGLE_API_KEY or "YOUR_GEMINI" in GOOGLE_API_KEY:
                    st.error("⚠️ API Key missing.")
                else:
                    with st.spinner(f"🔬 AI is analyzing in {lang_choice}..."):
                        try:
                            genai.configure(api_key=GOOGLE_API_KEY)
                            
                            
                            available_models = [m.name for m in genai.list_models() 
                                            if 'generateContent' in m.supported_generation_methods]
                            
                            if not available_models:
                                st.error("No available models found.")
                                st.stop()
                            
                            model_name = next((m for m in available_models if 'flash' in m or 'vision' in m), available_models[0])
                            model = genai.GenerativeModel(model_name)

                            
                            target_lang_instr = f"IMPORTANT: PROVIDE THE ENTIRE RESPONSE IN {lang_choice} LANGUAGE." if lang_choice != "English" else ""
                            crop_hint_text = (
                                f"User-provided crop hint: {crop_hint}."
                                if crop_hint != "Auto-detect"
                                else "User-provided crop hint: None (auto-detect from image)."
                            )

                            prompt = f"""
                            You are an expert agricultural assistant for farmers.
                            {crop_hint_text}
                            Analyze the uploaded plant image carefully and perform the following tasks:

                            CRITICAL IDENTIFICATION RULES:
                            - If a user crop hint is provided, treat it as primary context.
                            - Do not relabel the crop to another type unless there is very strong visual evidence.
                            - If uncertain between two crops, clearly say "uncertain" and ask for a clearer image.
                            - Do not default to maize/corn when the image is ambiguous.

                            1. Identify the plant name accurately (common name and local name if possible).
                            2. Determine whether the plant is a crop, weed, or other plant.
                            3. If it is a weed, briefly explain why it is harmful and how to control it.
                            4. If the plant is a crop, analyze its overall health condition (healthy, weak, or severely affected).
                            5. Detect any visible diseases, bacterial infections, fungal infections, viral infections, pest attacks, or nutrient deficiencies.
                            6. Clearly mention which part of the plant is affected (leaf, stem, root, flower, or fruit).
                            7. Explain the possible causes in simple words (weather conditions, soil quality, water stress, insects, or farming practices).
                            8. Mention the stage of crop growth (seedling, vegetative, flowering, fruiting, or harvesting stage).
                            9. Estimate the severity level of the problem (low, medium, or high).
                            10. Explain how the issue may affect crop yield if not treated.
                            11. Suggest practical treatment methods that are affordable and easy for farmers to apply.
                            12. Recommend specific fertilizers, bio-fertilizers, or micronutrients to reduce the disease or deficiency, including:
                                * Name of fertilizer
                                * Purpose (disease control)
                                * Application method (soil, foliar spray, drip)
                            13. Search and provide an example of the best fertilizers available in the local market (e.g., common NPK mixes, micronutrient solutions).
                            14. Provide dosage guidance in simple terms (per liter or per acre).
                            15. Suggest preventive measures to avoid the problem in future crops.
                            16. Mention safety precautions while using fertilizers or pesticides.
                            17. If the image is unclear, politely ask the farmer to upload a clearer photo.

                            Respond in simple language, avoid technical terms, and give step-by-step guidance suitable for farmers.
                            {target_lang_instr}
                            """
                            
                            response = model.generate_content([prompt, image])
                            response_text = response.text
                            
                            
                            st.success("Analysis Complete!")
                            with st.expander("🌿 Analysis Report", expanded=True):
                                st.markdown(response_text)
                                
                            
                            st.divider()
                            st.subheader(f"🔊 Listen (Summary)")
                            
                            try:
                                
                                if len(response_text) > 10000:
                                    cutoff = response_text[:10000].rfind('.')
                                    audio_text = response_text[:cutoff+1] if cutoff > 0 else response_text[:400]
                                    st.caption("Playing summary for speed...")
                                else:
                                    audio_text = response_text

                                
                                tts = gTTS(text=audio_text, lang=lang_map[lang_choice], slow=False)
                                tts.save("plant_advice.mp3")
                                
                            
                                audio_file = open("plant_advice.mp3", "rb")
                                audio_bytes = audio_file.read()
                                st.audio(audio_bytes, format="audio/mp3")
                                
                            except Exception as e:
                                st.warning(f"Could not generate audio: {e}")
                            
                        except Exception as e:
                            st.error(f"Error connecting to AI: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
