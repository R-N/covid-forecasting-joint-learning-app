
import streamlit as st
from ..global_data import Constants, load_data, load_pred, get_csv, get_excel

import pandas as pd
from pathlib import Path
# from sklearn.preprocessing import MinMaxScaler
import io
import json
from contextlib import redirect_stdout
from covid_forecasting_joint_learning.data import cols as DataCol
from covid_forecasting_joint_learning import main as Main
from covid_forecasting_joint_learning.model import general as general_model
from covid_forecasting_joint_learning.model.baseline import lstm, no_representation, source_longest, source_all, fully_shared, fully_private
from covid_forecasting_joint_learning.data.kabko import KabkoData
from covid_forecasting_joint_learning.data.center import DataCenter
from covid_forecasting_joint_learning.pipeline.preprocessing import Group
from covid_forecasting_joint_learning.pipeline.clustering import Cluster
from .data_exploration_all import get_dates, get_cols
from covid_forecasting_joint_learning.model.general import DEFAULT_FUTURE_EXO_COLS, DEFAULT_PAST_COLS
from streamlit_tensorboard import st_tensorboard
import torch


model_names = [
    "Main/Full",
    "LSTM Encoder-Decoder",
    "No Representation",
    "Source Longest",
    "Source All",
    "Fully Shared",
    "Fully Private"
]


model_codes = [
    "main",
    "baseline_1",
    "baseline_2",
    "baseline_3",
    "baseline_4",
    "baseline_5",
    "baseline_6"
]

model_funcs = [
    general_model,
    lstm,
    no_representation,
    source_longest,
    source_all,
    fully_shared,
    fully_private,
]

model_name_dict = dict(zip(model_codes, model_names))
model_func_dict = dict(zip(model_codes, model_funcs))

@st.cache(
    hash_funcs={
        DataCenter: id,
        KabkoData: id,
        Cluster: id,
        Group: id,
        type(KabkoData): id,
        type(load_data): id
    },
    allow_output_mutation=True
)
def main_1(
    loader,
    limit_length=[],
    limit_date=["2020-03-20"],
    limit_data=True,
    n_clusters=None,
    clustering_callback=None,
    kabkos=None,
    **kwargs
):
    groups = Main.main_1(
        loader,
        limit_length=limit_length,
        limit_date=limit_date,
        limit_data=limit_data,
        n_clusters=n_clusters,
        kabkos=kabkos,
        clustering_callback=clustering_callback,
        **kwargs
    )
    return groups

"""
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
"""
def set_targets(groups, targets):
    # groups = [g.copy() for g in groups]
    for group in groups:
        for cluster in group.clusters:
            cluster.set_targets(targets)

    return groups


@st.cache(
    hash_funcs={
        KabkoData: id,
        Cluster: id,
        Group: id,
        torch.nn.Module: id,
        type(torch.nn.Module): id,
        type(set_targets): id
    },
    allow_output_mutation=True
)
def create_set_targets(targets):
    def __set_targets(groups):
        return set_targets(groups, targets)
    return __set_targets

def show_clustering(groups):
    target_expander = st.expander(label='Clustering', expanded=False)
    with target_expander:
        st.markdown("## Clustering")
        for group in groups:
            st.markdown(f"### Group {group.id}")
            for cluster in group.clusters:
                st.markdown(f"#### Cluster {cluster.id}")
                st.write("Targets: ", [f"{t.name} ({len(t.data)}, {t.data.first_valid_index()})"for t in cluster.targets])
                st.write("Sources: ", [s.name for s in cluster.sources])

            mems = [k for c in group.clusters for k in c.members]
            removed = [k for k in group.members if k not in mems]
            st.write("Removed: ", [r.name for r in removed])
    return groups

@st.cache(
    hash_funcs={
        KabkoData: id,
        Cluster: id,
        Group: id,
        torch.nn.Module: id,
        type(torch.nn.Module): id,
        type(load_data): id
    },
    allow_output_mutation=True
)
def eval(
    model_code,
    groups,
    hparams,
    log_dir_copy,
    model_dir_copy,
    trial_id=-1,
    min_epoch=0,
    max_epoch=None,
    val=0,
    early_stopping_kwargs={}
):
    model_func = model_func_dict[model_code]
    result, epoch_log = model_func.eval(
        groups,
        hparams,
        log_dir_copy=log_dir_copy,
        model_dir_copy=model_dir_copy,
        trial_id=trial_id,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
        continue_train=False,
        val=val,
        early_stopping_kwargs=early_stopping_kwargs
    )

    result_1 = clean_result(result)
    return result_1, epoch_log

def clean_result(result):
    result_1 = {
        gk: {
            k: v for cluster, target in gv.items() for k, v in target.items()
        } for gk, gv in result.items()
    }
    return result_1

@st.cache(allow_output_mutation=True)
def recap_loss(result_1, targets, group=0):
    pre_df = []
    for i in range(3):
        label = DataCol.IRD[i]
        for target in targets:
            pre_df.append({
                "label": label,
                "target": target,
                "loss": result_1[group][target][i]
            })
    df = pd.DataFrame(pre_df)
    return df

@st.cache(allow_output_mutation=True)
def recap_epoch(epoch_log, targets, clusters, group=0):
    labels = ["last_epoch", "stop_reason", "best_epoch"]
    pre_df = []
    for i in range(3):
        label = labels[i]
        n_targets = len(targets)
        for j in range(n_targets):
            target = targets[j]
            # keys = sorted(epoch_log[group].keys())
            try:
                value = epoch_log[group][clusters[group][target]][label]
            except KeyError:
                print("[A]", epoch_log)
                print("[B]", clusters)
                print("[C]", epoch_log[group])
                print("[D]", clusters[group][target])
                print("[E]", epoch_log[group][clusters[group][target]])
                raise
            pre_df.append({
                "label": label,
                "target": target,
                "value": value
            })
    df = pd.DataFrame(pre_df)
    return df

def write_json(obj, file_name, groups, model_dir_2):
    with open(f"{model_dir_2}{file_name}", 'w', encoding='utf-8') as f:
        json.dump(obj, f)

    for group in groups:
        for cluster in group.clusters:
            model_dir_3 = f"{model_dir_2}/{group.id}/{cluster.id}/"
            Path(model_dir_3).mkdir(parents=True, exist_ok=True)
            with open(f"{model_dir_3}{file_name}", 'w', encoding='utf-8') as f:
                json.dump(obj, f)

def show_group_excel(title, group, model_dir_2, file_name, sheet_name, expanded=True):
    loss_expander = st.expander(label=title, expanded=expanded)
    with loss_expander:
        title_col, group_col = st.columns(2)
        title_col.markdown(f"## {title}")

        if isinstance(group, list):
            group = group_col.selectbox(
                "Group",
                group,
                key=title.lower().replace(" ", "-") + "-group-select"
            )

        title_col, download_col = st.columns(2)
        title_col.markdown(f"### Group {group}")

        file_path = f"{model_dir_2}/{group}/{file_name}"
        if not Path(file_path).is_file():
            st.write("Excel file missing: ", file_path)
            return

        df = pd.read_excel(file_path, sheet_name=sheet_name)
        download_col.download_button(
            "Download",
            get_excel(df, sheet_name=sheet_name),
            file_name
        )
        st.write(df)

def app(
    title="# Evaluation",
    log_dir="logs/eval",
    model_dir="model/eval",
    trial_id=-1,
    limit_data=True,
    val=0,
    early_stopping_kwargs={},
    show_loss=True,
    show_epoch=True,
    show_tb=False,
    hparam_dir="model/hparams/",
    min_epoch=0,
    max_epoch=None
):
    st.markdown(title)

    model_col, last_col = st.sidebar.columns(2)
    model_code = model_col.selectbox("Model", model_codes, format_func=lambda x: model_name_dict[x], index=0)
    last = last_col.date_input("Last date", pd.to_datetime("2021-03-20"))
    hparam_path = f"{hparam_dir}{model_code}.json"

    trial_id_col, min_epoch_col = st.sidebar.columns(2)
    trial_id = trial_id_col.number_input("Trial ID", value=trial_id, step=1)
    min_epoch = min_epoch_col.number_input("Minimum Epoch", value=min_epoch, min_value=0, step=1)


    with open(hparam_path, 'r', encoding='utf-8') as f:
        hparams = json.load(f)

    loader = load_data()
    kabko_names = loader.kabko.tolist()

    kabko_ms_expander = st.sidebar.expander(label='Kabupaten/Kota', expanded=False)
    # kabko_col, series_col = st.sidebar.columns(2)
    with kabko_ms_expander:
        kabko_names = st.multiselect(
            'Kabupaten/Kota',
            kabko_names,
            kabko_names
        )

    if len(kabko_names) == 0:
        st.error(f"Please select at least one kabupaten/kota")
        return

    target_names = [
        'KAB. BOJONEGORO',
        'KAB. SAMPANG',
        'KAB. TUBAN',
        'KOTA MADIUN',
        'KOTA PASURUAN'
    ]
    target_names = [t for t in target_names if t in kabko_names]

    target_expander = st.sidebar.expander(label='Targets', expanded=False)
    with target_expander:
        target_names = st.multiselect(
            'Targets',
            kabko_names,
            target_names
        )

    if len(target_names) == 0:
        st.error(f"Please select at least one target")
        return

    preprocessing_expander = st.expander(label='Preprocessing', expanded=False)
    with preprocessing_expander:
        st.markdown("## Preprocessing")
        with io.StringIO() as buf, redirect_stdout(buf):
            groups = main_1(
                loader,
                limit_length=[],
                limit_date=[last],
                limit_data=limit_data,
                n_clusters=None,
                kabkos=kabko_names,
                clustering_callback=create_set_targets(target_names)
            )
            print_out = buf.getvalue()
        st.write(print_out)

    show_clustering(groups)

    group = groups[0]
    kabkos = group.members

    past_cols = hparams["past_cols"]
    past_cols = DEFAULT_PAST_COLS[past_cols] if isinstance(past_cols, int) else past_cols
    future_exo_cols = hparams["future_exo_cols"]
    future_exo_cols = DEFAULT_FUTURE_EXO_COLS[future_exo_cols] if isinstance(future_exo_cols, int) else future_exo_cols

    past_col_expander = st.sidebar.expander(label='Past Cols', expanded=False)
    # kabko_col, series_col = st.sidebar.columns(2)
    with past_col_expander:
        past_cols = st.multiselect(
            'Past Cols',
            DEFAULT_PAST_COLS[0],
            past_cols
        )

    future_exo_cols_expander = st.sidebar.expander(label='Future Exo Cols', expanded=False)
    # kabko_col, series_col = st.sidebar.columns(2)
    with future_exo_cols_expander:
        future_exo_cols = st.multiselect(
            'Future Exo Cols',
            DEFAULT_FUTURE_EXO_COLS[0],
            future_exo_cols
        )

    hparams["past_cols"] = past_cols
    hparams["future_exo_cols"] = future_exo_cols

    model_dir_2 = f"{model_dir}/{trial_id}/"
    Path(model_dir_2).mkdir(parents=True, exist_ok=True)

    write_json(hparams, "hparams.json", groups, model_dir_2)
    write_json({
        "model_code": model_code,
        "last_date": last.strftime("%Y-%m-%d"),
        "kabko_names": kabko_names,
        "target_names": target_names
    }, "model.json", groups, model_dir_2)

    eval_expander = st.expander(label='Eval', expanded=False)
    with eval_expander:
        st.markdown("## Eval exec")
        with io.StringIO() as buf, redirect_stdout(buf):
            result_1, epoch_log = eval(
                model_code,
                groups,
                hparams,
                log_dir_copy=log_dir,
                model_dir_copy=model_dir,
                trial_id=trial_id,
                min_epoch=min_epoch,
                max_epoch=max_epoch,
                val=val,
                early_stopping_kwargs=early_stopping_kwargs
            )
            print_out = buf.getvalue()
        st.write(print_out)


    cols = {group.id: [t.name for t in group.targets] for group in groups}
    group_numbers = sorted(list(cols.keys()))

    file_name = "result.xlsx"
    sheet_name = "eval"

    for group, targets in cols.items():
        df = recap_loss(result_1, targets, group)
        model_dir_3 = f"{model_dir_2}/{group}/"
        Path(model_dir_3).mkdir(parents=True, exist_ok=True)
        df.to_excel(f"{model_dir_3}{file_name}", index=False, sheet_name=sheet_name)

    show_group_excel(
        title="RMSSE Loss",
        group=group_numbers,
        model_dir_2=model_dir_2,
        file_name=file_name,
        sheet_name=sheet_name,
        expanded=show_loss
    )

    clusters = {group.id: {t.name: t.cluster.id for t in group.targets} for group in groups}

    sheet_name = "epoch"
    file_name = "epoch.xlsx"

    for group, targets in cols.items():
        df = recap_epoch(epoch_log, targets, clusters, group)
        model_dir_3 = f"{model_dir_2}/{group}/"
        Path(model_dir_3).mkdir(parents=True, exist_ok=True)
        df.to_excel(f"{model_dir_3}{file_name}", index=False, sheet_name=sheet_name)

    show_group_excel(
        title="Epoch Log",
        group=group_numbers,
        model_dir_2=model_dir_2,
        file_name=file_name,
        sheet_name=sheet_name,
        expanded=show_loss
    )

    log_dir_2 = f"{log_dir}/T{trial_id}/"
    tb_expander = st.expander(label='Tensorboard', expanded=show_tb)
    with tb_expander:
        heading_col, width_col = st.columns(2)
        heading_col.markdown("## Tensorboard")
        width = width_col.number_input("Viewport Width", min_value=0, value=800, step=1)
        st_tensorboard(logdir=log_dir_2, port=6006, width=width)

    return groups, hparams, model_dir, trial_id, target_names
