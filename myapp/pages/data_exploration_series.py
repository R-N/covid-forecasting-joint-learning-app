
import streamlit as st
from ..global_data import Constants, load_data, load_pred

import gc
import pandas as pd
from matplotlib.figure import Figure
# from sklearn.preprocessing import MinMaxScaler
from covid_forecasting_joint_learning.pipeline import main as Pipeline
from covid_forecasting_joint_learning.data import cols as DataCol
from covid_forecasting_joint_learning.data import exploration as DataExplorator
from streamlit import components
from covid_forecasting_joint_learning.data.kabko import KabkoData
from covid_forecasting_joint_learning.pipeline.preprocessing import Group
from covid_forecasting_joint_learning.pipeline.clustering import Cluster

@st.cache(
    hash_funcs={
        KabkoData: id,
        Cluster: id,
        Group: id,
        type(KabkoData): id,
        type(load_data): id
    },
    allow_output_mutation=True
)
def init(preprocessing_3=True):
    kabkos = Pipeline.get_kabkos(load_data())
    kabkos = Pipeline.preprocessing_1(kabkos)
    kabko_names = [kabko.name for kabko in kabkos]
    kabko_dict = {kabko.name: kabko for kabko in kabkos}

    # kabkos = Pipeline._preprocessing_2([(k, k.data) for k in kabkos])

    if preprocessing_3:
        for kabko in kabkos:
            # kabko.data = kabko.data.loc[:kabko.split_indices[2]].copy()
            Pipeline.preprocessing_3([kabko], limit_split=False, scale=True)
    gc.collect()

    return kabkos, kabko_names, kabko_dict

@st.cache(allow_output_mutation=True)
def get_dates(
    single_dates=DataCol.SINGLE_DATES,
    labeled_dates=DataCol.LABELED_DATES
):
    single_dates = single_dates + [d for d in labeled_dates.keys() if d not in single_dates]
    return single_dates, labeled_dates

@st.cache(allow_output_mutation=True)
def get_adf(series, series_label):
    adf = DataExplorator.adf(series)
    return DataExplorator.print_adf(adf)

@st.cache(hash_funcs={Figure: hash}, allow_output_mutation=True)
def plot_rolling_stats(series, series_label, smoothing):
    mean, std = DataExplorator.rolling_stats(series, smoothing)
    fig = DataExplorator.plot_rolling_stats(series, mean, std, series_label, smoothing)
    return fig

@st.cache(hash_funcs={Figure: hash}, allow_output_mutation=True)
def plot_acf(series, series_label):
    return DataExplorator.plot_acf(series, name=series_label)

@st.cache(hash_funcs={Figure: hash}, allow_output_mutation=True)
def plot_pacf(series, series_label):
    return DataExplorator.plot_pacf(series, name=series_label)

@st.cache(allow_output_mutation=True)
def seasonal_decompose(series, seasonality):
    decomposed = DataExplorator.classical_decompose(pd.DataFrame(series), period=seasonality, model="additive")
    decomposed.trend.dropna(inplace=True)
    decomposed.seasonal.dropna(inplace=True)
    decomposed.resid.dropna(inplace=True)
    return decomposed

@st.cache(hash_funcs={Figure: hash}, allow_output_mutation=True)
def plot_decompose(decomposed, series_label):
    return DataExplorator.plot_classical_decompose(decomposed, name=series_label)

@st.cache(allow_output_mutation=True)
def get_residual_stats(resid, series_label):
    return DataExplorator.print_residual_stats(resid.dropna(), name=series_label)

@st.cache(hash_funcs={Figure: hash}, allow_output_mutation=True)
def plot_interactive(data, lines=[], fills=[], title="", line_colors=None, width=12, height=6):
    return DataExplorator.plot_fill(
        data,
        lines=lines,
        fills=fills,
        title=title,
        figsize=(width, height),
        line_colors=line_colors,
        ipython=False
    )

@st.cache(hash_funcs={Figure: hash}, allow_output_mutation=True)
def plot(data, lines=[], fills=[], title="", line_colors=None):
    return DataExplorator.plot_fill(
        data,
        lines=lines,
        fills=fills,
        title=title,
        figsize=(12, 6),
        interactive=False,
        line_colors=line_colors
    )

def app():
    st.markdown("# Data Exploration Series")

    kabkos, kabko_names, kabko_dict = init(True)

    #kabko_col, series_col = st.sidebar.columns(2)
    kabko_name = st.sidebar.selectbox(
        'Kabupaten/Kota',
        kabko_names
    )
    kabko = kabko_dict[kabko_name]

    single_dates, labeled_dates = get_dates()

    first_col, last_col = st.sidebar.columns(2)
    first = first_col.date_input("First date", pd.to_datetime("2020-03-20"))
    last = last_col.date_input("Last date", pd.to_datetime("2021-03-20"))

    series_col, order_col = st.sidebar.columns(2)
    smoothing_col, seasonality_col = st.sidebar.columns(2)
    date_expander = st.sidebar.expander(label='Dates', expanded=False)
    with date_expander:
        single_dates = st.multiselect(
            'Dates',
            single_dates,
            single_dates
        )
    labeled_dates = {x: y for x, y in labeled_dates.items() if x in single_dates}

    data = kabko.add_dates(
        kabko.data,
        dates=labeled_dates
    )
    data = data[first:last]

    pred = None
    pred_loader = load_pred()
    if pred_loader is not None:
        pred = pred_loader[pred_loader.kabko == kabko.name][DataCol.SIRD_VARS + DataCol.IRD + [DataCol.DATE]]
        pred.set_index(DataCol.DATE, inplace=True)
        pred = pred[first:last]
        pred = None if len(pred) == 0 else pred

    series_label = series_col.selectbox(
        'Series Label',
        DataCol.IRD + DataCol.SIRD_VARS
    )
    order = order_col.number_input(
        "Differencing Order",
        min_value=0,
        value=0,
        step=1
    )
    series = DataExplorator.diff(
        data[series_label],
        order=order
    )

    smoothing = smoothing_col.number_input(
        "Smoothing",
        min_value=1,
        value=30,
        max_value=len(series),
        step=1
    )

    stationarity_expander = st.expander(label='Stationarity', expanded=True)
    with stationarity_expander:
        st.markdown("## Stationarity")
        st.markdown(get_adf(series, series_label))
        st.pyplot(plot_rolling_stats(series, series_label, smoothing))

    autocorrelation_expander = st.expander(label='Autocorrelation', expanded=True)
    with autocorrelation_expander:
        st.markdown("## Autocorrelation")
        st.pyplot(plot_acf(series, series_label))
        st.pyplot(plot_pacf(series, series_label))

    seasonality = seasonality_col.number_input(
        "Seasonality",
        min_value=1,
        value=7,
        step=1
    )

    decomposition_expander = st.expander(label='Classical Seasonal Decomposition', expanded=True)
    with decomposition_expander:
        st.markdown("## Classical Seasonal Decomposition")
        decomposed = seasonal_decompose(series, seasonality)
        st.pyplot(plot_decompose(decomposed, series_label))
        if seasonality > 1:
            if st.checkbox("Seasonal", False):
                st.pyplot(plot_acf(decomposed.seasonal, series_label + "_seasonal"))
                st.pyplot(plot_pacf(decomposed.seasonal, series_label + "_seasonal"))
        if st.checkbox("Residual", False):
            st.markdown(get_adf(decomposed.resid, series_label + "_resid"))
            st.pyplot(plot_rolling_stats(decomposed.resid, series_label + "_resid", smoothing))
            st.markdown(get_residual_stats(decomposed.resid, series_label + "_resid"))

    dates_expander = st.expander(label='Dates Observation', expanded=True)
    with dates_expander:
        st.markdown("## Dates Observation")

        pwcol, phcol, pdpicol = st.columns(3)
        pwidth = pwcol.number_input(
            "Plot Width",
            min_value=1.0,
            value=8.5,
            step=0.5
        )

        pheight = phcol.number_input(
            "Plot Height",
            min_value=1.0,
            value=4.5,
            step=0.5
        )

        pdpi = pdpicol.number_input(
            "Plot DPI",
            min_value=1.0,
            value=100.0,
            step=1.0
        )

        components.v1.html(
            plot_interactive(
                data,
                [series_label],
                single_dates,
                title="Dates no differencing",
                width=pwidth,
                height=pheight
            ),
            width=pwidth * pdpi,
            height=pheight * pdpi
        )
        if order > 0:
            components.v1.html(
                plot_interactive(
                    data,
                    [series],
                    single_dates,
                    title="Dates with differencing",
                    width=pwidth,
                    height=pheight
                ),
                width=pwidth * pdpi,
                height=pheight * pdpi
            )

    if pred is not None:
        pred_expander = st.expander(label='Forecast Comparison', expanded=True)
        with pred_expander:
            series = data[series_label]
            lines = [series, pred[series_label]]
            lines[0].name = series_label + "_true"
            lines[-1].name = series_label + "_pred"

            st.markdown("## Forecast Comparison")
            st.pyplot(plot(
                data,
                lines=[lines[1]],
                title="Forecasted " + series_label
            ))
            st.pyplot(plot(
                data,
                lines=lines,
                line_colors=["#ff7f0e", "#1f77b4"],
                title="Comparison of " + series_label
            ))
            st.pyplot(plot(
                data,
                lines=[lines[0]],
                line_colors=["#ff7f0e"],
                title="True " + series_label
            ))
