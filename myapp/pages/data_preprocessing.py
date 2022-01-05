
import streamlit as st
from ..global_data import Constants, load_data, load_pred

import gc
import pandas as pd
from pathlib import Path
# from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import io
from contextlib import redirect_stdout
from covid_forecasting_joint_learning.pipeline import main as Pipeline, preprocessing, postprocessing, sird
from covid_forecasting_joint_learning.data import cols as DataCol
from .data_exploration_all import get_dates, get_cols, init
from matplotlib import pyplot as plt
import pandas as pd
from covid_forecasting_joint_learning.data.kabko import KabkoData
from covid_forecasting_joint_learning.pipeline.preprocessing import Group
from covid_forecasting_joint_learning.pipeline.clustering import Cluster

@st.cache(
    hash_funcs={
        KabkoData: hash,
        Cluster: hash,
        Group: hash
    },
    allow_output_mutation=True
)
def preprocess(kabkos, labeled_dates, last):
    for k in kabkos:
        k.data = k.add_dates(k.data, dates=labeled_dates)

    groups = Pipeline.preprocessing_2(kabkos, limit_length=[], limit_date=[last])

    for group in groups:
        Pipeline.preprocessing_3(group.members)

    group = groups[0]
    kabkos = group.members
    kabko_dict = {k.name: k for k in kabkos}
    gc.collect()

    return groups, kabkos, kabko_dict


def print_clustering_result(groups):
    for g in groups:
        st.markdown(f"### Group {g.id}")
        mems = [k for c in g.clusters for k in c.members]
        removed = [k for k in g.members if k not in mems]
        st.write("Removed", [r.name for r in removed])
        for c in g.clusters:
            st.markdown(f"#### Cluster {c.id}")
            st.write("Targets: ", [f"{t.name} ({len(t.data)}, {t.data.first_valid_index()})"for t in c.targets])
            st.write("Sources: ", [s.name for s in c.sources])


@st.cache(
    hash_funcs={
        KabkoData: hash,
        Cluster: hash,
        Group: hash
    },
    allow_output_mutation=True
)
def cluster(groups):
    with io.StringIO() as buf, redirect_stdout(buf):
        clusters = [Pipeline.clustering_1(group) for group in groups]
        print_out = buf.getvalue()
    return clusters, print_out


@st.cache(
    hash_funcs={
        KabkoData: hash,
        Cluster: hash,
        Group: hash
    },
    allow_output_mutation=True
)
def preprocess_2(groups, x_cols, future_exo_cols):
    for group in groups:
        for cluster in group.clusters:
            Pipeline.preprocessing_4(cluster)
            Pipeline.preprocessing_5(
                cluster.members,
                past_cols=x_cols,
                future_exo_cols=future_exo_cols
            )
            Pipeline.preprocessing_6(cluster.members)
    return groups

@st.cache(
    hash_funcs={
        KabkoData: hash,
        Cluster: hash,
        Group: hash
    },
    allow_output_mutation=True
)
def sample_rebuild(kabko):
    sample = kabko.get_batch_sample(single=True, last=True)
    future, prev, final, indices, population = sample[3][0].numpy(), sample[5][0].numpy(), sample[6][0].numpy(), sample[7][0], sample[8]
    future = kabko.scaler_2.inverse_transform(future)
    y_data = pd.DataFrame(final, columns=DataCol.IRD, index=indices)
    rebuilt = postprocessing.sird.rebuild(future, prev[-1], population, index=indices)
    return y_data, rebuilt


def app():
    st.markdown("# Data Preprocessing & Clustering")


    loader = load_data()

    global_expander = st.expander(label='Data Global', expanded=True)
    with global_expander:
        st.markdown("## Data Global")
        st.write(loader.data_global)

    single_dates, labeled_dates = get_dates()
    kabkos, kabko_names, kabko_dict = init()
    kabkos = [k.copy() for k in kabkos]

    kabko_ms_expander = st.sidebar.expander(label='Kabupaten/Kota', expanded=False)
    # kabko_col, series_col = st.sidebar.columns(2)
    with kabko_ms_expander:
        kabko_names = st.multiselect(
            'Kabupaten/Kota',
            kabko_names,
            kabko_names
        )

    kabko_col, last_col = st.sidebar.columns(2)
    kabko_name = kabko_col.selectbox("Kabupaten/kota", kabko_names)
    last = last_col.date_input("Last date", pd.to_datetime("2021-03-20"))

    cols_non_date = DataCol.COLS_NON_DATE + DataCol.DAYS

    exo_expander = st.sidebar.expander(label='Exogen Variables', expanded=False)
    # kabko_col, series_col = st.sidebar.columns(2)
    with exo_expander:
        single_dates = st.multiselect(
            'Dates',
            single_dates,
            single_dates
        )
        cols_non_date = st.multiselect(
            'Cols Non Date',
            cols_non_date,
            DataCol.COLS_NON_DATE
        )
    labeled_dates = {x: y for x, y in labeled_dates.items() if x in single_dates}

    y_cols = DataCol.SIRD
    cols_non_date, x_cols = get_cols(single_dates, labeled_dates, cols_non_date, y_cols)

    kabko_expander = st.expander(label='Data Kabko', expanded=True)
    with kabko_expander:
        st.markdown("## Data Kabko")

        groups, kabkos, kabko_dict = preprocess(kabkos, labeled_dates, last)

        kabko = kabko_dict[kabko_name]

        st.write(kabko.data)

    clustering_expander = st.expander(label='Clustering', expanded=True)
    with clustering_expander:
        st.markdown("## Clustering")

        clusters, print_out = cluster(groups)

        targets = set([cluster.target.name for group in groups for cluster in group.clusters])
        # st.markdown("Targets: " + ", ".join(sorted(list(targets))))
        st.write("Targets: ", sorted(list(targets)))

        longests = set([cluster.source_longest.name for group in groups for cluster in group.clusters])
        # st.markdown("Longests: " + ", ".join(sorted(list(longests))))
        st.write("Longests: ", sorted(list(longests)))

        print_clustering_result(groups)


    clustering_log_expander = st.expander(label='Clustering log', expanded=True)
    with clustering_log_expander:
        st.write(print_out)

    rebuild_expander = st.expander(label='Rebuild', expanded=True)
    with rebuild_expander:
        st.markdown("## Rebuild")
        groups = preprocess_2(
            groups,
            x_cols=x_cols,
            future_exo_cols=[c for c in x_cols if c not in y_cols and c not in DataCol.COLS_NON_DATE]
        )

        y_data, rebuilt = sample_rebuild(kabko)

        for k in DataCol.IRD:
            fig, ax = plt.subplots(1, 1)
            ax.plot(y_data[k], label="true")
            ax.plot(rebuilt[k], label="rebuilt")
            ax.legend(loc='best')
            ax.title.set_text("Rebuilt comparison for " + k)
            # ax.xticks(rotation=90)
            st.pyplot(fig)
