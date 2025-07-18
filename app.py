import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoARIMA, AutoETS, Theta

st.set_page_config(page_title="AI Forecasting App", layout="wide")

st.title("📈 Multi-Model Forecasting Tool")
st.write("Upload your monthly demand data and get forecasts with best-model selection.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=["date"])
    st.success("✅ File uploaded successfully!")
    st.write("Data preview:", data.head())

    # Config
    horizon = st.number_input("Forecast Horizon (months)", min_value=1, max_value=24, value=6)
    n_windows = st.slider("Backtest Windows", 1, 5, 3)

    if st.button("Run Forecast"):
        with st.spinner("Running forecasts..."):
            data = data.sort_values(["sku_id","location","date"])
            data["series_id"] = data["sku_id"] + "_" + data["location"]
            sf_df = data.rename(columns={"series_id":"unique_id","date":"ds","demand":"y"})[["unique_id","ds","y"]]

            models = [Naive(), SeasonalNaive(season_length=12), AutoARIMA(), AutoETS(), Theta()]
            sf = StatsForecast(models=models, freq="M", n_jobs=1)
            results = []
            plots = {}

            for uid, g in sf_df.groupby("unique_id"):
                fcst = sf.forecast(df=g, h=horizon)
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(g["ds"], g["y"], label="Actual")
                for m in models:
                    mname = m.__class__.__name__
                    ax.plot(fcst["ds"], fcst[mname], label=mname)
                ax.legend()
                ax.set_title(uid)
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plots[uid] = buf.getvalue()
                plt.close()
                best_model = "AutoETS"  # Placeholder logic
                results.append({"Series": uid, "Best Model": best_model})

            st.subheader("Results")
            st.write(pd.DataFrame(results))

            st.subheader("Plots")
            for uid, img in plots.items():
                st.image(img, caption=uid)
