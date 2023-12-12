import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
import pickle


img = Image.open('C:\\Users\\WELCOME\\OneDrive\\Desktop\\saravanan\\Copper_Modeling\\images (3).jpeg')
st.set_page_config(page_title = 'Industrial Copper Modeling', page_icon = img, layout = 'wide')

selected = option_menu('Industrial Copper Modeling', ["Home","Analysis","Predict"], 
    icons=['house', "reception-4","dice-5-fill"], 
    menu_icon='buildings-fill', default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#509191"}, # #008080
        "icon": {"color": "violet", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "center", "margin":"0px", "--hover-color": "#008080"},
        "nav-link-selected": {"background-color": "#008080"}, 
    }
)

df = pd.read_csv(r'Copper_Modeling.csv')
df1 = pd.read_csv(r'Cleaned_Copper_Modeling.csv')
df2 = pd.read_csv(r'Copper_Modeling_Original.csv')

if selected == 'Home':
    st.markdown("# :orange[Project Title :]")
    st.markdown("## &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Industrial Copper Modeling")
    st.markdown("# :orange[Skills take away From This Project :]")
    st.markdown("## &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Python scripting, Data Preprocessing, EDA, Streamlit.")
    st.markdown("# :orange[Domain :]")
    st.markdown("## &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Manufacturing")
    url = "https://docs.google.com/spreadsheets/d/18eR6DBe5TMWU9FnIewaGtsepDbV4BOyr/edit?rtpof=true&sd=true#gid=462557918"
    st.markdown("# :orange[Dataset Link : [Data Link](%s)]"% url)
    st.markdown("# :orange[Problem Statement :]")
    st.markdown("## &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.")
    st.markdown("## &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.")

if selected == 'Predict':
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
                options = df2['customer'].unique(),
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
                options = df2['product_ref'].unique(),
            )
        with column1:
            for i in df1.columns:
                if df1[i].dtype == 'object' and i != 'id':
                    col_name = i
                    decode = df1[i].sort_values().unique() # status
                    encode = df2[i].sort_values().unique() # 0,1,2
                    globals()[col_name] = {}
                    globals()[i] = dict(zip(decode, encode))

            with open('Decision_Tree.pkl', 'rb') as file:
                model = pickle.load(file)
            result = model.predict([[ton, customer, country, status[stat], item_type[type], application, thickness, width, product]])
            st.markdown("")
            st.markdown("")
            button = st.button(label = 'Predict',use_container_width = True, type = "primary")
        if button:
            st.markdown(f"# :green[Predicted Selling Price is {result}]")
        
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
                options = df2['product_ref'].unique(),
            )
            
        with column1:
            for i in df1.columns:
                if df1[i].dtype == 'object' and i != 'id':
                    col_name = i
                    decode = df1[i].sort_values().unique()
                    encode = df2[i].sort_values().unique()
                    globals()[col_name] = {}
                    globals()[i] = dict(zip(decode, encode))
            with open('DecisionTree_classifier.pkl', 'rb') as file:
                model = pickle.load(file)
            result1 = model.predict([[ton, int(customer), country, price, item_type[type], application, thickness, width, product]])
        if button1:
            if result1 == 0.0:
                st.markdown("# :green[Product Status is Loss]")
            else:
                st.markdown("# :green[Product Status is Won]")


elif selected == 'Analysis':
    st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Customer who buy product for highest selling price")
    column1,column2 = st.columns([3,2], gap = 'small')
    with column1:
        df2['customer'] = df2['customer'].astype('str')
        df2['customer'] = df2['customer'].apply(lambda x : x.split('.')[0])
        data = df2[['selling_price','customer','country']].sort_values('selling_price',ascending=False,ignore_index=True).head(20)
        pie = px.pie(data, names = 'customer', values= 'selling_price', hover_data = 'country')
        st.plotly_chart(pie)
    with column2:
        st.dataframe(data.style.background_gradient(cmap='Purples'),use_container_width=True)
   

    st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Customer who buy product for lowest selling price")
    column1,column2 = st.columns([3,2], gap = 'small')
    with column1:
        data = df2[['selling_price','customer','country']].sort_values('selling_price',ascending=True,ignore_index=True).head(50)
        pie = px.pie(data, names = 'customer', values= 'selling_price', hover_data = 'country')
        st.plotly_chart(pie)
    with column2:
        st.dataframe(data.style.background_gradient(cmap='Purples'),use_container_width=True)


    st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Total count of product status")
    status_count = df.groupby('status').count().reset_index()[['status','id']].sort_values('id',ascending=False)
    bar = px.bar(status_count, x = 'status', y = 'id', orientation = 'v', labels = {'status':'Status', 'id':'Total Count'}, width = 1200, height = 600, color_continuous_scale = 'ylorbr',color = 'id')
    st.plotly_chart(bar)


    st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Which Item Type is Highly Produced in Company")
    item_type_count = df.groupby('item type').count().reset_index()[['item type','id']].sort_values('id',ascending=False)
    bar = px.bar(item_type_count, x = 'item type', y = 'id', orientation = 'v', labels = {'item type':'Item Type', 'id':'Total Count'}, width = 1200, height = 600, color_continuous_scale = 'ylorbr',color = 'id')
    st.plotly_chart(bar)


    st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; In which date most number of items Produced")
    item_date_count = df2.groupby('item_date').count().reset_index()[['item_date','id']].sort_values('id',ascending=False).head(10)
    line = px.line(item_date_count, x = 'item_date', y = 'id', orientation = 'v', labels = {'item_date':'Item Date', 'id':'Number of Item Produced'}, width = 1200, height = 600)
    st.plotly_chart(line)


    st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; In which date most number of items Delivered")
    delivery_date_count = df2.groupby('delivery date').count().reset_index()[['delivery date','id']].sort_values('id',ascending=False).head(10)
    line = px.line(delivery_date_count, x = 'delivery date', y = 'id', orientation = 'v', labels = {'delivery date':'Delivery Date', 'id':'Number of Item Deliverd'}, width = 1200, height = 600)
    st.plotly_chart(line)


    st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; which delivery date has highest sold")
    delivery_date_count = df2.groupby(['delivery date','selling_price','customer']).count().reset_index()[['delivery date','selling_price','customer']].sort_values('selling_price',ascending=False).head(40)
    pie = px.pie(delivery_date_count, names = 'delivery date', values = 'selling_price', labels = {'delivery date':'Delivery Date', 'id':'Number of Item Deliverd'}, width = 1000, height = 600, hover_data = 'customer', color_discrete_sequence = px.colors.sequential.RdBu)
    st.plotly_chart(pie)


    column1,column2 = st.columns([2,2], gap = 'small')
    with column1:
        st.markdown("# Selling Price without Outlier")
        box = px.box(df2,y='selling_price')
        st.plotly_chart(box)
    with column2:
        st.markdown("# Selling Price with Outlier")
        box = px.box(df,y='selling_price')
        st.plotly_chart(box)


    column1,column2 = st.columns([2,2], gap = 'small')
    with column1:
        outlier = df1[(df1['quantity tons']>-73.42875229999999) & (df1['quantity tons']<152.52067025999997)]
        st.markdown("# Quantity Tons without Outlier")
        box = px.box(outlier,y='quantity tons')
        st.plotly_chart(box)
    with column2:
        st.markdown("# Quantity Tons with Outlier")
        box = px.box(df,y='quantity tons')
        st.plotly_chart(box)

    
    st.markdown("# &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Selling Price vs Item Type produced in company")
    outlier = df[(df['selling_price']>243.0) & (df['selling_price']<1379.0)]
    violin = px.violin(outlier,x = 'item type', y = 'selling_price',color = 'item type',box=True,width = 1300,height = 600)
    st.plotly_chart(violin)


    st.markdown("# which country has highest market with company")
    hist = px.histogram(df2, y = 'selling_price', x = 'country',width = 1200, height = 600)
    st.plotly_chart(hist)


hide = """
    <style>
    footer {visibility: hidden;}
    # header {visibility: hidden;}
    </style>
    """
st.markdown(hide,unsafe_allow_html = True)