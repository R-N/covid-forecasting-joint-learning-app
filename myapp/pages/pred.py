
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
from covid_forecasting_joint_learning.model.general import DEFAULT_FUTURE_EXO_COLS, DEFAULT_PAST_COLS

def _app():
    return __app(
        title="# Forecast",
        log_dir="logs/pred",
        model_dir="model/pred",
        trial_id=-2,
        limit_data=False,
        val=2,
        early_stopping_kwargs={
            "rise_patience": 25,
            "still_patience": 25,
            "both_patience": 75
        },
        show_loss=False,
        show_epoch=True,
        show_tb=False
    )

@st.cache(
    hash_funcs={
        KabkoData: hash,
        Cluster: hash,
        Group: hash
    },
    allow_output_mutation=True
)
def pred(target, model_dir_3):
    data_torch = target.datasets_torch[0][0]
    target.model.future_length = data_torch[4].size(1)
    target.model.eval()
    pred_vars = target.model(*data_torch[:5]).detach().numpy()

    data_np = target.datasets[0][0]
    indices = data_np[-1]
    df_vars = pd.DataFrame(pred_vars[0], columns=DataCol.SIRD_VARS, index=indices)

    prev = data_torch[5]
    pred_final = target.model.rebuild(pred_vars, prev, target.population, sird.rebuild)

    indices = data_np[-1]
    df_final = pd.DataFrame(pred_final[0], columns=DataCol.IRD, index=indices)

    Path(model_dir_3).mkdir(parents=True, exist_ok=True)
    df = pd.concat([df_vars, df_final], axis=1)
    df.to_excel(f"{model_dir_3}/pred.xlsx", sheet_name="pred")
    return df


@st.cache(
    hash_funcs={
        KabkoData: hash,
        Cluster: hash,
        Group: hash
    },
    allow_output_mutation=True
)
def save_combined_pred(preds, model_dir_2):
    df = pd.concat(preds)
    df.to_excel(f"{model_dir_2}/pred.xlsx", sheet_name="pred")


label_name = [
    (DataCol.IRD, "pred_final"),
    (DataCol.SIRD_VARS, "pred_vars")
]
fig_names = [ln[1] for ln in label_name] + DataCol.IRD + DataCol.SIRD_VARS

def plot_etc(fig, ax, name, model_dir_3):
    ax.legend(loc='best')
    ax.title.set_text(name)
    # ax.tick_params(labelrotation=90)
    ax.grid(which="both", alpha=0.3)
    fig.savefig(f"{model_dir_3}/{name}.jpg", bbox_inches="tight")

@st.cache(hash_funcs={Figure: hash, Spines: hash}, allow_output_mutation=True)
def plot_many(df, labels, name, model_dir_3):
    fig, ax = plt.subplots(1, 1)
    for k in labels:
        ax.plot(df[k], label=k)
    plot_etc(fig, ax, name, model_dir_3)
    return fig

@st.cache(hash_funcs={Figure: hash, Spines: hash}, allow_output_mutation=True)
def plot_single(df, k, model_dir_3):
    fig, ax = plt.subplots(1, 1)
    ax.plot(df[k], label=k)
    plot_etc(fig, ax, k, model_dir_3)
    return fig

@st.cache(hash_funcs={Figure: hash, Spines: hash}, allow_output_mutation=True)
def make_figs(df, model_dir_3):
    fig_dict_1 = {}
    plt.close('all')

    for labels, name in label_name:
        fig = plot_many(df, labels, name, model_dir_3)
        fig_dict_1[name] = fig

    for k in DataCol.IRD + DataCol.SIRD_VARS:
        fig = plot_single(df, k, model_dir_3)
        fig_dict_1[k] = fig

    return fig_dict_1

def app():
    groups, hparams, model_dir, trial_id, target_names = _app()
    groups = [g.copy() for g in groups]

    group = groups[0]

    model_dir_2 = f"{model_dir}/{trial_id}/{group.id}/"
    Path(model_dir_2).mkdir(parents=True, exist_ok=True)

    pred_expander = st.expander(label='Forecast', expanded=True)
    with pred_expander:
        st.markdown("## Forecast")
        target_col, pred_col = st.columns(2)
        target_name = target_col.selectbox(
            'Kabupaten/Kota',
            target_names
        )
        fig_name = st.multiselect(
            'Label',
            fig_names,
            DataCol.IRD
        )

        # pred_date = "2021-12-31"
        # pred_date = datetime.date.today()
        target = [t for t in group.targets if t.name == target_name][0]
        pred_date = target.data.last_valid_index()

        pred_date = pred_date + datetime.timedelta(days=14)
        pred_date = pred_col.date_input("Forecast until date", pd.to_datetime(pred_date))

        past_cols = hparams["past_cols"]
        past_cols = DEFAULT_PAST_COLS[past_cols] if isinstance(past_cols, int) else past_cols
        future_exo_cols = hparams["future_exo_cols"]
        future_exo_cols = DEFAULT_FUTURE_EXO_COLS[future_exo_cols] if isinstance(future_exo_cols, int) else future_exo_cols
        Pipeline.preprocessing_7(
            group.targets,
            end_date=pred_date,
            past_size=30 + hparams["additional_past_length"],
            past_cols=past_cols,
            future_exo_cols=future_exo_cols
        )

        pred_dict = {}
        fig_dict = {}

        for cluster in group.clusters:
            for target in cluster.targets:
                model_dir_3 = f"{model_dir_2}/{target.cluster.id}/{target.name}"
                df = pred(target, model_dir_3)
                pred_dict[target.name] = df
                df["kabko"] = pd.Series(target.name, index=df.index)

                fig_dict[target.name] = make_figs(df, model_dir_3)

        preds = [pred_dict[t] for t in target_names]
        save_combined_pred(preds, model_dir_2)

        for f in fig_name:
            fig = fig_dict[target_name][f]
            st.pyplot(fig)

