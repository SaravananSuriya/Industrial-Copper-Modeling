import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_extras.colored_header import colored_header
import pandas as pd
import pickle

@st.cache_data
def data():
    df = pd.read_csv(r'Copper_Modeling.csv')
    df1 = pd.read_csv(r'Cleaned_Copper_Modeling.csv')
    df2 = pd.read_csv(r'Copper_Modeling_Original.csv')
    return df,df1,df2
df,df1,df2 = data()


with st.sidebar:
    st.sidebar.markdown("# :orange[Select an option to Predict:]")
    option = st.selectbox("", ("Price Prediction", "Status Prediction"))
        
if option == "Price Prediction":
    
    st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; :red[Prediction of selling price]")
    st.markdown("")
    st.info(" &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;**Note: Enter value Greater than of minimum value and Lesser than of maximum value**")
    column1,column2 = st.columns([2,2], gap = 'small')
    with column1:
        min_ton = df2['quantity tons'].min()
        max_ton = df2['quantity tons'].max()
        min_thick = df2['thickness'].min()
        max_thick = df2['thickness'].max()
        min_width = df2['width'].min()
        max_width = df2['width'].max()
        min_customer = df2['customer'].min()
        max_customer = df2['customer'].max()
        ton = st.number_input(f'**Enter Quantity Tons (Minimum : {min_ton}, Maximun : {max_ton})**')
        thickness = st.number_input(f'**Enter Thickness (Minimum : {min_thick}, Maximun : {max_thick})**')
        width = st.number_input(f'**Enter Width (Minimum : {min_width}, Maximun : {max_width})**')
        customer = st.selectbox(
            "**Select a Customer Id**",
            options = df1['customer'].unique(),
        )
    with column2:
        type = st.selectbox(
            "**Select an Item Type**",
            options = df1['item_type'].unique(),
        )
        country = st.selectbox(
            "**Select a Country**",
            options = df2['country'].unique(),
        )
        application = st.selectbox(
            "**Select an Application**",
            options = df2['application'].unique(),
        )
        stat = st.selectbox(
            "**Select a Status**",
            options = df1['status'].unique(),
        )
        product = st.selectbox(
            "**Select a Product Reference**",
            options = df1['product_ref'].unique(),
        )
    with column1:
        for i in df1.columns:
            if df1[i].dtype == 'object' and i != 'id':
                col_name = i
                decode = df1[i].sort_values().unique() # status
                encode = df2[i].sort_values().unique() # 0,1,2
                globals()[col_name] = {}
                globals()[i] = dict(zip(decode, encode))

        with open('dt_reg_model.pkl', 'rb') as file:
            model = pickle.load(file)
        result = model.predict([[ton, 1/customer, country, status[stat], item_type[type], application, thickness, width, 1/product]])
        st.markdown("")
        st.markdown("")
        button = st.button(label = 'Predict',use_container_width = True, type = "primary")
    if button:
        st.markdown(f"## Predicted Selling Price is :green[{result[0]}]")
    
elif option == "Status Prediction":
    st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; :red[Prediction of Product Status]")
    st.markdown("")
    st.info("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;**Note: Enter value Greater than of minimum value and Lesser than of maximum value**")
    column1,column2 = st.columns([2,2], gap = 'small')
    with column1:
        df3 = df2[(df2['status']==0.0) | (df2['status']==6.0)]
        min_ton = df3['quantity tons'].min()
        max_ton = df3['quantity tons'].max()
        min_thick = df3['thickness'].min()
        max_thick = df3['thickness'].max()
        min_width = df3['width'].min()
        max_width = df3['width'].max()
        min_customer = df3['customer'].min()
        max_customer = df3['customer'].max()
        min_price = df3['selling_price'].min()
        max_price = df3['selling_price'].max()
        ton = st.number_input(f'**Enter Quantity Tons (Minimum : {min_ton}, Maximun : {max_ton})**')
        thickness = st.number_input(f'**Enter Thickness (Minimum : {min_thick}, Maximun : {max_thick})**')
        width = st.number_input(f'**Enter Width (Minimum : {min_width}, Maximun : {max_width})**')
        price = st.number_input(f'**Enter Selling Price (Minimum : {min_price}, Maximun : {max_price})**')
        st.markdown("")
        st.markdown("")
        button1 = st.button(label = 'Predict',use_container_width = True, type = "primary")
    with column2:
        customer = st.selectbox(
            "**Select a Customer Id**",
            options = df1['customer'].unique(),
        )
        type = st.selectbox(
            "**Select an Item Type**",
            options = df1['item_type'].unique(),
        )
        country = st.selectbox(
            "**Select a Country**",
            options = df2['country'].unique(),
        )
        application = st.selectbox(
            "**Select an Application**",
            options = df2['application'].unique(),
        )
        product = st.selectbox(
            "**Select a Product Reference**",
            options = df1['product_ref'].unique(),
        )
        
    with column1:
        for i in df1.columns:
            if df1[i].dtype == 'object' and i != 'id':
                col_name = i
                decode = df1[i].sort_values().unique()
                encode = df2[i].sort_values().unique()
                globals()[col_name] = {}
                globals()[i] = dict(zip(decode, encode))
        with open('dt_class_model.pkl', 'rb') as file:
            model = pickle.load(file)
        result1 = model.predict([[ton, 1/customer, country, price, item_type[type], application, thickness, width, 1/product]])
    if button1:
        if result1 == 0.0:
            st.markdown("## Product Status is :green[Loss]")
        else:
            st.markdown("## Product Status is :green[Won]")