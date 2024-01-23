import streamlit as st
import pandas as pd
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain
import plotly.express as px

st.set_page_config(
    page_title = "Analysis",
    page_icon = "",
    layout = 'wide'
)

@st.cache_data
def data():
    df = pd.read_csv(r'Copper_Modeling.csv')
    df1 = pd.read_csv(r'Cleaned_Copper_Modeling.csv')
    df2 = pd.read_csv(r'Copper_Modeling_Original.csv')
    return df,df1,df2
df,df1,df2 = data()

st.markdown("#  Customer who buy product for highest selling price")
column1,column2 = st.columns([3,2], gap = 'small')
with column1:

    data = df1[['selling_price','customer','country']].sort_values('selling_price',ascending=False,ignore_index=True).head(50)
    pie = px.pie(data, names = 'customer', values= 'selling_price', hover_data = 'country', width=500)
    st.plotly_chart(pie)
with column2:
    st.markdown('')
    st.markdown('')
    st.dataframe(data.style.background_gradient(cmap='Purples'),use_container_width=True)


st.markdown("#  Customer who buy product for lowest selling price")
column1,column2 = st.columns([3,2], gap = 'small')
with column1:
    data = df1[['selling_price','customer','country']].sort_values('selling_price',ascending=True,ignore_index=True).head(20)
    pie = px.pie(data, names = 'customer', values= 'selling_price', hover_data = 'country', width=500)
    st.plotly_chart(pie)
with column2:
    st.markdown('')
    st.markdown('')
    st.dataframe(data.style.background_gradient(cmap='Purples'),use_container_width=True)


st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Total count of product status")
status_count = df.groupby('status').count().reset_index()[['status','id']].sort_values('id',ascending=False)
bar = px.bar(status_count, x = 'status', y = 'id', orientation = 'v', labels = {'status':'Status', 'id':'Total Count'}, width = 1100, height = 450, color_continuous_scale = 'ylorbr',color = 'id')
st.plotly_chart(bar)


st.markdown("#  Which Item Type is Highly Produced in Company")
item_type_count = df.groupby('item type').count().reset_index()[['item type','id']].sort_values('id',ascending=False)
bar = px.bar(item_type_count, x = 'item type', y = 'id', orientation = 'v', labels = {'item type':'Item Type', 'id':'Total Count'}, width = 1100, height = 450, color_continuous_scale = 'ylorbr',color = 'id')
st.plotly_chart(bar)


st.markdown("# &nbsp; &nbsp; &nbsp;  In which date most number of items Produced")
item_date_count = df2.groupby('item_date').count().reset_index()[['item_date','id']].sort_values('id',ascending=False).head(10)
line = px.line(item_date_count, x = 'item_date', y = 'id', orientation = 'v', labels = {'item_date':'Item Date', 'id':'Number of Item Produced'}, width = 1050, height = 500)
st.plotly_chart(line)


st.markdown("# &nbsp; &nbsp; &nbsp;  In which date most number of items Delivered")
delivery_date_count = df2.groupby('delivery date').count().reset_index()[['delivery date','id']].sort_values('id',ascending=False).head(10)
line = px.line(delivery_date_count, x = 'delivery date', y = 'id', orientation = 'v', labels = {'delivery date':'Delivery Date', 'id':'Number of Item Deliverd'}, width = 1050, height = 500)
st.plotly_chart(line)


st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; which delivery date has highest sold")
delivery_date_count = df2.groupby(['delivery date','selling_price','customer']).count().reset_index()[['delivery date','selling_price','customer']].sort_values('selling_price',ascending=False).head(40)
pie = px.pie(delivery_date_count, names = 'delivery date', values = 'selling_price', labels = {'delivery date':'Delivery Date', 'id':'Number of Item Deliverd'}, width = 1000, height = 500, hover_data = 'customer', color_discrete_sequence = px.colors.sequential.RdBu)
st.plotly_chart(pie)

# @st.cache_data(experimental_allow_widgets=True)
# def price():
#     column1,column2 = st.columns([2,2], gap = 'small')
#     with column1:
#         st.markdown("# Price without Outlier")
#         box = px.box(df2,y='selling_price',width=550)
#         st.plotly_chart(box)
#     with column2:
#         st.markdown("# Price with Outlier")
#         box = px.box(df,y='selling_price',width=550)
#         st.plotly_chart(box)
# price()

# @st.cache_data(experimental_allow_widgets=True)
# def tons():
#     column1,column2 = st.columns([2,2], gap = 'small')
#     with column1:
#         outlier = df1[(df1['quantity tons']>-73.42875229999999) & (df1['quantity tons']<152.52067025999997)]
#         st.markdown("# Quantity Tons without Outlier")
#         box = px.box(outlier,y='quantity tons',width=550)
#         st.plotly_chart(box)
#     with column2:
#         st.markdown("# Quantity Tons with Outlier")
#         box = px.box(df,y='quantity tons',width=550)
#         st.plotly_chart(box)
# tons()


st.markdown("# &nbsp; &nbsp; &nbsp;  Selling Price vs Item Type produced in company")
outlier = df[(df['selling_price']>243.0) & (df['selling_price']<1379.0)]
violin = px.violin(outlier,x = 'item type', y = 'selling_price',color = 'item type',box=True,width = 1100,height = 600)
st.plotly_chart(violin)

st.markdown("# Product thickness vs Selling Price")
hist = px.histogram(df2, y = 'selling_price', x = 'thickness',width = 1000, height = 500)
st.plotly_chart(hist)

st.markdown("# Product width vs Selling Price")
hist = px.histogram(df2, y = 'selling_price', x = 'width',width = 1000, height = 500)
st.plotly_chart(hist)

st.markdown("# Which application is giving more selling price to the company")
hist = px.histogram(df2, y = 'selling_price', x = 'application',width = 1000, height = 500)
st.plotly_chart(hist)

st.markdown("# which country has highest market with company")
hist = px.histogram(df2, y = 'selling_price', x = 'country',width = 1000, height = 500)
st.plotly_chart(hist)

st.markdown("# Item produced date vs Selling price")
hist = px.histogram(df2, y = 'selling_price', x = 'item_date',width = 1000, height = 500)
st.plotly_chart(hist)

st.markdown("# Item delivery date vs Selling price")
hist = px.histogram(df2, y = 'selling_price', x = 'delivery date',width = 1000, height = 500)
st.plotly_chart(hist)

hide = """
<style>
footer {visibility: hidden;}
# header {visibility: hidden;}
</style>
"""
st.markdown(hide,unsafe_allow_html = True)