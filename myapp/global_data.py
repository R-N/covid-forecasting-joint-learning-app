import gc
import pandas as pd
import streamlit as st
from io import BytesIO
from covid_forecasting_joint_learning.data.center import DataCenter
from covid_forecasting_joint_learning.data import exploration as DataExplorator
from covid_forecasting_joint_learning.data import cols as DataCol
from covid_forecasting_joint_learning.pipeline import main as Pipeline
from covid_forecasting_joint_learning import main as Main

class Constants:
    data_folder = 'data/'
    data_path = data_folder + 'data.xlsx'
    pred_path = data_folder + "pred.xlsx"
    date_json = data_folder + "date_corr.json"

@st.cache(allow_output_mutation=True)
def load_data():
    loader = DataCenter()
    loader.load_excel(Constants.data_path)
    loader = Pipeline.preprocessing_0(loader)
    gc.collect()
    return loader

@st.cache(allow_output_mutation=True)
def load_pred():
    try:
        pred_loader = pd.read_excel(Constants.pred_path, sheet_name="pred")
        pred_loader[DataCol.DATE] = pd.to_datetime(pred_loader[DataCol.DATE])
        gc.collect()
        return pred_loader
    except FileNotFoundError:
        return None

@st.cache(allow_output_mutation=True)
def init():
    device = Main.init(False)

    _markdown = st.markdown
    def markdown_wrapper(x, *args, **kwargs):
        if isinstance(x, str):
            # return _markdown(x.replace(r"\n", r" \n"), *args, **kwargs)
            xs = x.split("\n")
            for x in xs:
                _markdown(x, *args, **kwargs)
        else:
            return _markdown(x, *args, **kwargs)
    st.markdown = markdown_wrapper

    _write = st.write
    def write_wrapper(x, *args, **kwargs):
        if isinstance(x, str):
            # return _write(x.replace(r"\n", r" \n"), *args, **kwargs)
            xs = x.split("\n")
            for x in xs:
                _write(x, *args, **kwargs)
        else:
            return _write(x, *args, **kwargs)
    st.write = write_wrapper

    load_data()
    load_pred()

def get_excel(df, sheet_name="Sheet 1"):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_csv(df):
    return df.to_csv(index=False).encode('utf-8')
