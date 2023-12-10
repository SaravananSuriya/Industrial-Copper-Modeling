# Industrial-Copper-Modeling

**Project Title :** Industrial Copper Modeling

**Skills take away From This Project :** Python scripting, Data Preprocessing, EDA, Streamlit.

**Domain :** Manufacturing

**Data Set :** [Data Link](https://docs.google.com/spreadsheets/d/18eR6DBe5TMWU9FnIewaGtsepDbV4BOyr/edit#gid=462557918)

## Problem Statement :
  The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data. 
      
Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.

**The solution must include the following steps :**

    1) Exploring skewness and outliers in the dataset.
    2) Transform the data into a suitable format and perform any necessary cleaning and pre-processing steps.
    3) ML Regression model which predicts continuous variable ‘Selling_Price’.
    4) ML Classification model which predicts Status: WON or LOST.
    5) Creating a streamlit page where you can insert each column value and you will get the Selling_Price predicted value or Status(Won/Lost).

## Approach :

    1) Data Understanding: Identify the types of variables (continuous, categorical) and their distributions. Some rubbish values are present in ‘Material_Reference’ which starts with ‘00000’ value which should be converted into null. Treat reference columns as categorical variables. INDEX may not be useful.
    2) Data Preprocessing:
  
          *Handle missing values with mean/median/mode.
          
    
