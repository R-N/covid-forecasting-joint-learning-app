
import streamlit as st
from ..global_data import Constants, load_data, load_pred

import pandas as pd
from pathlib import Path
import datetime
# from sklearn.preprocessing import MinMaxScaler
from covid_forecasting_joint_learning.pipeline import main as Pipeline, sird
from covid_forecasting_joint_learning.data import cols as DataCol
from matplotlib import pyplot as plt
from .eval import app as __app
from matplotlib.figure import Figure
from matplotlib.spines import Spines
from covid_forecasting_joint_learning.data.kabko import KabkoData
from covid_forecasting_joint_learning.pipeline.preprocessing import Group
from covid_forecasting_joint_learning.pipeline.clustering import Cluster
from covid_forecasting_joint_learning.model.general import DEFAULT_FUTURE_EXO_COLS, DEFAULT_PAST_COLS, prepare_params, DEFAULT_ACTIVATIONS
import io
from contextlib import redirect_stdout
from .eval import main_1, create_set_targets, show_group_excel
from .pred import pred, fig_names, make_figs, save_combined_pred, preprocess_pred
import json
from covid_forecasting_joint_learning.model.modules.main import SingleModel
import torch

def str_path(path, relative_to=None):
    if not relative_to:
        ret = str(path)
    else:
        ret = f"{str_path(relative_to)}/{str(path.relative_to(relative_to))}"
    return ret.replace("\\", "/")

def set_activations(
    params,
    keys=[
        "activation",
        "conv_activation",
        "fc_activation",
        "residual_activation"
    ],
    activations=DEFAULT_ACTIVATIONS
):
    for key in keys:
        if key in params:
            params[key] = activations[params[key]]
    for value in params.values():
        if isinstance(value, dict):
            set_activations(value, keys=keys, activations=activations)
    return params


def app(
    model_dir="model/pred",
    state_file="model.pt",
    model_dir_b="model/pred_load"
):
    st.markdown("# Forecast (Load Model)")

    model_dir_path = Path(model_dir)
    pred_trials = [f for f in model_dir_path.iterdir() if f.is_dir()]

    if len(pred_trials) == 0:
        st.error("No trial found. Please use the normal forecast module first")
        return

    trial_col, group_col = st.sidebar.columns(2)
    trial_path = trial_col.selectbox(
        "Trial",
        pred_trials,
        format_func=lambda x: x.name
    )

    pred_groups = [f for f in trial_path.iterdir() if f.is_dir()]
    if len(pred_groups) == 0:
        st.error(f"No valid group in trial {trial_path.name}")
        return

    pred_group_path = group_col.selectbox(
        "Group",
        pred_groups,
        format_func=lambda x: x.name
    )

    group_numbers = [x.name for x in pred_groups]

    pred_clusters = [f for f in pred_group_path.iterdir() if f.is_dir()]
    pred_targets = [f for g in pred_clusters for f in g.iterdir() if f.is_dir()]
    pred_targets = [t for t in pred_targets if len([f for f in t.iterdir() if f.name == state_file and f.is_file()])]
    target_names = [t.name for t in pred_targets]

    if len(pred_targets) == 0:
        st.error(f"No valid target in trial {trial_path.name}")
        return

    loader = load_data()
    kabko_names = loader.kabko.tolist()

    trial_path_str = str_path(trial_path, model_dir_path)
    with open(f"{trial_path_str}/model.json", 'r', encoding='utf-8') as f:
        model_info = json.load(f)

    with open(f"{trial_path_str}/hparams.json", 'r', encoding='utf-8') as f:
        hparams = json.load(f)

    past_cols = hparams["past_cols"]
    past_cols = DEFAULT_PAST_COLS[past_cols] if isinstance(past_cols, int) else past_cols
    future_exo_cols = hparams["future_exo_cols"]
    future_exo_cols = DEFAULT_FUTURE_EXO_COLS[future_exo_cols] if isinstance(future_exo_cols, int) else future_exo_cols
    # hparams = prepare_params(hparams)

    kabko_names = model_info["kabko_names"]
    kabko_ms_expander = st.sidebar.expander(label='Learned kabkos', expanded=False)
    # kabko_col, series_col = st.sidebar.columns(2)
    with kabko_ms_expander:
        st.write("Learned kabkos: ", kabko_names)

    past_cols_expander = st.sidebar.expander(label='Learned past_cols', expanded=False)
    with past_cols_expander:
        st.write("Learned past_cols: ", hparams["past_cols"])

    future_exo_cols_expander = st.sidebar.expander(label='Learned future_exo_cols', expanded=False)
    with future_exo_cols_expander:
        st.write("Learned future_exo_cols: ", hparams["future_exo_cols"])

    last = pd.to_datetime(model_info["last_date"])
    ## last = st.sidebar.date_input("Last date", last, disabled=True)
    st.sidebar.write("Last learned date: ", last.strftime("%Y-%m-%d"))

    preprocessing_expander = st.expander(label='Preprocessing', expanded=False)
    with preprocessing_expander:
        st.markdown("## Preprocessing")
        with io.StringIO() as buf, redirect_stdout(buf):
            groups = main_1(
                loader,
                limit_length=[],
                limit_date=[last],
                limit_data=False,
                n_clusters=None,
                kabkos=kabko_names,
                clustering_callback=create_set_targets(target_names)
            )
            print_out = buf.getvalue()
        st.write(print_out)

    group = groups[0]

    trial_id = trial_path.name
    model_dir_2 = f"{model_dir_b}/{trial_id}/{group.id}/"
    Path(model_dir_2).mkdir(parents=True, exist_ok=True)

    # model_dir_2a = f"{model_dir}/{trial_id}"
    show_group_excel(
        title="RMSSE loss",
        group=group.id,
        model_dir_2=trial_path_str,
        file_name="result.xlsx",
        sheet_name="eval",
        expanded=True
    )

    show_group_excel(
        title="Epoch Log",
        group=group.id,
        model_dir_2=trial_path_str,
        file_name="epoch.xlsx",
        sheet_name="epoch",
        expanded=True
    )

    pred_expander = st.expander(label='Forecast', expanded=True)
    with pred_expander:
        st.markdown("## Forecast")
        target_col, pred_col = st.columns(2)
        target_path = target_col.selectbox(
            'Kabupaten/Kota',
            pred_targets,
            format_func=lambda x: x.name
        )
        target_name = target_path.name
        fig_name = st.multiselect(
            'Label',
            fig_names,
            DataCol.IRD
        )

        # pred_date = "2021-12-31"
        # pred_date = datetime.date.today()

        targets = [t for t in group.targets if t.name in target_names]
        target = [t for t in targets if t.name == target_name][0]
        pred_date = target.data.last_valid_index()

        pred_date = pred_date + datetime.timedelta(days=14)
        pred_date = pred_col.date_input("Forecast until date", pd.to_datetime(pred_date))

        cluster_path_str = str_path(target_path.parent, model_dir_path)

        with open(f"{cluster_path_str}/sizes.json", 'r', encoding='utf-8') as f:
            sizes = json.load(f)
        with open(f"{cluster_path_str}/model_kwargs.json", 'r', encoding='utf-8') as f:
            model_kwargs = json.load(f)
        model_kwargs = set_activations(model_kwargs)

        target.model = SingleModel(**sizes, **model_kwargs)

        target_path_str = str_path(target_path, model_dir_path)
        target.model.load_state_dict(torch.load(f"{target_path_str}/{state_file}"))

        preprocess_pred(
            targets,
            end_date=pred_date,
            past_size=30 + hparams["additional_past_length"],
            past_cols=past_cols,
            future_exo_cols=future_exo_cols
        )

        pred_dict = {}
        fig_dict = {}

        model_dir_3 = f"{model_dir_2}/{target.cluster.id}/{target.name}"

        # for target in targets:
        df = pred(target, model_dir_3)
        pred_dict[target.name] = df
        df["kabko"] = pd.Series(target.name, index=df.index)

        fig_dict[target.name] = make_figs(df, model_dir_3)

        # preds = [pred_dict[t] for t in target_names]
        # save_combined_pred(preds, model_dir_2)

        for f in fig_name:
            fig = fig_dict[target_name][f]
            st.pyplot(fig)

