
import streamlit as st
from ..global_data import Constants, load_data, load_pred, get_csv, get_excel

import gc
import pandas as pd
from pathlib import Path
# from sklearn.preprocessing import MinMaxScaler
from covid_forecasting_joint_learning.data.getter import DataGetter
from .data_exploration_series import plot


@st.cache(allow_output_mutation=True)
def get_data(data_path, last_modified):
    if not data_path.is_file():
        return None
    df = pd.read_csv(data_path)
    return df

def _show_data(df, kabko=None):
    dl_all_col, dl_kabko_col = st.columns(2)

    dl_all_col.download_button(
        "Download All",
        get_csv(df),
        "data.csv"
    )
    if not kabko or kabko == "all":
        st.write(df)
    else:
        df = df[df["kabko"] == kabko]
        dl_kabko_col.download_button(
            f"Download {kabko}",
            get_csv(df),
            f"data_{kabko}.csv"
        )
        st.write(df)


def show_data(data_path):
    data_path = Path(data_path)
    st.markdown("## Show data")
    if not data_path.is_file():
        st.markdown("Data file doesn't exist. Click the button above to fetch data.")
        return
    df = get_data(data_path, data_path.stat().st_mtime)
    kabko_names = ["all"] + df["kabko"].unique().tolist()
    # kabko_col, label_col = st.columns(2)
    kabko = st.sidebar.selectbox("Show data", kabko_names, index=0)

    _show_data(df, kabko)


def app():
    st.markdown("# Data Getter")
    endpoint = st.sidebar.text_input("API endpoint", "http://covid19dev.jatimprov.go.id/xweb/drax/data/")
    data_path = st.sidebar.text_input("Data path (csv)", "data/data.csv")
    first_col, last_col = st.sidebar.columns(2)
    first = first_col.date_input("First date", pd.to_datetime("2020-03-20"))
    last = last_col.date_input("Last date", pd.to_datetime("2021-03-20"))

    if st.button("Fetch Data"):
        getter = DataGetter(endpoint)
        dates = getter.generate_date_range(first, last)
        df = getter.to_dataframe(getter.get_data_bulk(dates))
        df.to_csv(data_path, index=False)

    show_data(data_path)