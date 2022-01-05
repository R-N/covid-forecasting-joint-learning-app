import streamlit as st
from ..global_data import Constants, load_data, load_pred

import gc
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from covid_forecasting_joint_learning.data import cols as DataCol
from covid_forecasting_joint_learning.data import exploration as DataExplorator
from .data_exploration_series import init, get_dates

@st.cache(allow_output_mutation=True)
def get_cols(
    single_dates=DataCol.SINGLE_DATES,
    labeled_dates=DataCol.LABELED_DATES,
    cols_non_date=DataCol.COLS_NON_DATE,
    y_cols=DataCol.SIRD_VARS
):
    single_dates, labeled_dates = get_dates(single_dates, labeled_dates)
    x_cols = [
        *cols_non_date,
        *single_dates
    ]
    cols = x_cols + y_cols
    return cols_non_date, cols

@st.cache(allow_output_mutation=True)
def get_dfs(kabkos, labeled_dates, cols, first, last):
    dfs = [kabko.add_dates(kabko.data, labeled_dates)[cols] for kabko in kabkos]
    dfs = [df[first:last] for df in dfs]
    return dfs

@st.cache(allow_output_mutation=True)
def count_valid_dates(dfs, single_dates):
    valid_date_count = {d: sum([1 for df in dfs if max(df[d]) > 0]) for d in single_dates}
    valid_date_count_df = pd.DataFrame(valid_date_count.items(), columns=["date", "count"])
    return valid_date_count_df

@st.cache(allow_output_mutation=True)
def corr_lag_best_multi_dfs(
    dfs,
    x_cols,
    y_cols,
    lag_start=0,
    lag_end=-14,
    method="kendall"
):
    return DataExplorator.corr_lag_best_multi_dfs(
        dfs,
        x_cols=x_cols,
        y_cols=y_cols,
        lag_start=lag_start,
        lag_end=lag_end,
        method=method
    )

@st.cache(hash_funcs={Figure: hash}, allow_output_mutation=True)
def corr_matrix(corr, figsize=(8, 5)):
    return DataExplorator.corr_matrix(corr, figsize=figsize)

def app():
    st.markdown("# Data Exploration All")

    kabkos, kabko_names, kabko_dict = init(True)
    single_dates, labeled_dates = get_dates()
    cols_non_date = DataCol.COLS_NON_DATE

    method_col, minimum_col = st.sidebar.columns(2)
    corr_method = method_col.selectbox(
        "Correlation Method",
        ["pearson", "spearman", "kendall"],
        index=2
    )
    min_corr = minimum_col.number_input(
        "Minimum correlation",
        value=0.3,
        step=0.1,
        min_value=0.0,
        max_value=1.0
    )

    first_col, last_col = st.sidebar.columns(2)
    lag_start_col, lag_end_col = st.sidebar.columns(2)

    #kabko_col, series_col = st.sidebar.columns(2)
    kabko_expander = st.sidebar.expander(label='Kabupaten/Kota', expanded=False)
    with kabko_expander:
        kabko_names = st.multiselect(
            'Kabupaten/Kota',
            kabko_names,
            kabko_names
        )

    date_expander = st.sidebar.expander(label='Dates', expanded=False)
    with date_expander:
        single_dates = st.multiselect(
            'Dates',
            single_dates,
            single_dates
        )
    labeled_dates = {x: y for x, y in labeled_dates.items() if x in single_dates}

    cols_non_date = DataCol.COLS_NON_DATE + DataCol.DAYS
    cols_nd_expander = st.sidebar.expander(label='Cols Non Date', expanded=False)
    with cols_nd_expander:
        cols_non_date = st.multiselect(
            'Cols Non Date',
            cols_non_date,
            cols_non_date
        )

    y_cols = DataCol.IRD + DataCol.DELTA_IRD + DataCol.SIRD_VARS
    y_expander = st.sidebar.expander(label='Y Cols', expanded=False)
    with y_expander:
        y_cols = st.multiselect(
            'Y Cols',
            y_cols,
            y_cols
        )

    cols_non_date, cols = get_cols(single_dates, labeled_dates, cols_non_date, y_cols)

    first = first_col.date_input("First date", pd.to_datetime("2020-03-20"))
    last = last_col.date_input("Last date", pd.to_datetime("2021-03-20"))

    dfs = get_dfs(kabkos, labeled_dates, cols, first, last)

    st.markdown("## Valid date count")
    st.write(count_valid_dates(dfs, single_dates))

    # y_col = y_cols[0]
    """
    y_col = st.sidebar.selectbox(
        'Y Col',
        y_cols
    )
    """

    lag_start = lag_start_col.number_input("Lag Start", value=0, step=1)
    lag_end = lag_end_col.number_input("Lag End", value=-14, step=1)

    if st.checkbox("Correlation Matrix for cols_non_date", False):
        corr = corr_lag_best_multi_dfs(
            dfs,
            x_cols=cols_non_date,
            y_cols=y_cols,
            lag_start=lag_start,
            lag_end=lag_end,
            method=corr_method
        )

        st.markdown("## Correlation Matrix for cols_non_date")
        st.pyplot(corr_matrix(corr, figsize=(8, 5)))

        fair_vars = sorted(list({k for k, v in corr.T.items() if max(np.abs(v)) >= min_corr}))
        st.markdown("Fair vars: " + ", ".join(fair_vars))


    if st.checkbox("Correlation Matrix for dates", False):
        corr_dates = corr_lag_best_multi_dfs(
            dfs,
            x_cols=single_dates,
            y_cols=y_cols,
            lag_start=lag_start,
            lag_end=lag_end,
            method=corr_method
        )

        st.markdown("## Correlation Matrix for dates")
        st.pyplot(corr_matrix(corr_dates, figsize=(8, 10)))

        fair_dates = sorted(list({k for k, v in corr_dates.T.items() if max(np.abs(v)) >= min_corr}))
        st.markdown("Fair dates: " + ", ".join(fair_dates))

    if st.checkbox("Correlation Matrix for all, without days", False):
        corr_all = corr_lag_best_multi_dfs(
            dfs,
            x_cols=[d for d in DataCol.COLS_NON_DATE if d in cols_non_date] + single_dates,
            y_cols=y_cols,
            lag_start=lag_start,
            lag_end=lag_end,
            method=corr_method
        )

        st.markdown("## Correlation Matrix for all, without days")
        st.pyplot(corr_matrix(corr_all, figsize=(8, 10)))

        fair_all = sorted(list({k for k, v in corr_all.T.items() if max(np.abs(v)) >= min_corr}))
        st.markdown("Fair vars: " + ", ".join(fair_all))
