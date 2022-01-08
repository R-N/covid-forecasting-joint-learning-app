
import streamlit as st

from myapp.global_data import init
from myapp.pages import data_exploration_series, data_exploration_all, data_getter, data_preprocessing, eval, pred, home, pred_load  # import your pages here

st.set_page_config(layout="wide")

page_dict = {}
page_names = []

def add_page(title, func):
    page_names.append(title)
    page_dict[title] = func


def run():
    # Drodown to select the page to run
    page_name = st.sidebar.selectbox(
        'App Navigation',
        page_names
    )

    # run the app function
    page_dict[page_name]()


init()

# Title of the main page
# st.title("Covid Forecasting Joint Learning")

# Add all your applications (pages) here
add_page("Home", home.app)
add_page("Data Getter", data_getter.app)
add_page("Exploration Series", data_exploration_series.app)
add_page("Exploration All", data_exploration_all.app)
add_page("Preprocessing & Clustering", data_preprocessing.app)
add_page("Evaluation", eval.app)
add_page("Forecast", pred.app)
add_page("Forecast (Load)", pred_load.app)

# The main app
run()
