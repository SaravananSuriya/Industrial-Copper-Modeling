import streamlit as st
import pandas as pd
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain

st.set_page_config(
    page_title = "Project Description",
    page_icon = "📚",
    layout = 'wide'
)

st.markdown("# :orange[Project Title :]")
st.markdown("## &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Industrial Copper Modeling")
st.markdown("# :orange[Skills take away From This Project :]")
st.markdown("## &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Python scripting, Data Preprocessing, EDA, Streamlit.")
st.markdown("# :orange[Domain :]")
st.markdown("## &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Manufacturing")
st.markdown("# :orange[Problem Statement :]")
st.markdown("## &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.")
st.markdown("## &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.")
url = "https://docs.google.com/spreadsheets/d/18eR6DBe5TMWU9FnIewaGtsepDbV4BOyr/edit?rtpof=true&sd=true#gid=462557918"
st.markdown("# :orange[Dataset Link : [Data Link](%s)]"% url)