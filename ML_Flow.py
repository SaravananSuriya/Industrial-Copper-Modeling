import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from sklearn.metrics import accuracy_score , confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv(r"Copper_Modeling_Original.csv")
df.drop(['item_date','id','delivery date'],axis = 1, inplace = True)
# print(df.head())

x = df.drop(['selling_price'],axis = 1)
y = df['selling_price']
# print(x.info())

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size = 0.2, random_state = 30)

def randomforest():
    n_estimators_val = [50, 100, 200, 250, 350, 400,450]
    Max_depth = [5,10,15,20, 25, 30, 35]
    Min_sample_split = [15,12,10,8,7,6,5]

    for i,j,k in zip(Max_depth,Min_sample_split,n_estimators_val):
        model_rf = RandomForestRegressor(criterion = 'squared_error', n_estimators = k, max_depth = i, min_samples_split = j,  random_state=30)
        model_rf.fit(x_train,y_train)

        y_train_pred = model_rf.predict(x_train)
        y_test_pred = model_rf.predict(x_test)

        res = pd.DataFrame(data = list(zip(y_train,y_train_pred,y_test,y_test_pred)), columns = ['y_train','y_train_pred','y_test','y_test_pred'])
        print(res)

        train_MAE = mean_absolute_error(y_train_pred, y_train)
        test_MAE = mean_absolute_error(y_test_pred, y_test)

        #   train_MSE = mean_squared_error(y_train_pred, y_train)
        #   test_MSE = mean_squared_error(y_test_pred, y_test)

        # Train_RMSE = mean_squared_error(y_train_pred, y_train)
        # Test_RMSE = mean_squared_error(y_test_pred, y_test)
        # train_RMSE = sqrt(Train_RMSE)
        # test_RMSE = sqrt(Test_RMSE)
        
        train_MAPE = mean_absolute_percentage_error(y_train_pred, y_train)
        test_MAPE = mean_absolute_percentage_error(y_test_pred, y_test)

        with mlflow.start_run():
            mlflow.log_param('n_estimators', k)
            mlflow.log_param('max_depth', i)
            mlflow.log_param('min_sample_split', j)

            mlflow.log_metric('train_MeanAbsoluteError', train_MAE)
            mlflow.log_metric('test_MeanAbsoluteError', test_MAE)

            # mlflow.log_metric('train_MeanSquaredError', train_MSE)
            # mlflow.log_metric('test_MeanSquaredError', test_MSE)

            # mlflow.log_metric('train_RootMeanSquaredError', train_RMSE)
            # mlflow.log_metric('test_RootMeanSquaredError', test_RMSE)

            mlflow.log_metric('train_MeanAbsolutePercentageError', train_MAPE)
            mlflow.log_metric('test_MeanAbsolutePercentageError', test_MAPE)

            mlflow.sklearn.log_model(model_rf, 'model')
# randomforest()


def decisiontree():
    Max_depth = [5,10,15,20, 25, 30, 35,50, 70, 85, 100]
    # Min_sample_split = [40,35,30,25,20,15,10,9,8,5,2]
    Min_sample_split = [2,5,8,9,10,15,20,25,30,35,40]

    for i,j in zip(Max_depth,Min_sample_split):
        model_dt = DecisionTreeRegressor(criterion = 'squared_error', max_depth = i, min_samples_split = j,  random_state=30)
        model_dt.fit(x_train,y_train)

        y_train_pred = model_dt.predict(x_train)
        y_test_pred = model_dt.predict(x_test)

        res = pd.DataFrame(data = list(zip(y_train,y_train_pred,y_test,y_test_pred)), columns = ['y_train','y_train_pred','y_test','y_test_pred'])
        print(i,j)
        print(res)

        train_MAE = mean_absolute_error(y_train_pred, y_train)
        test_MAE = mean_absolute_error(y_test_pred, y_test)

        #   train_MSE = mean_squared_error(y_train_pred, y_train)
        #   test_MSE = mean_squared_error(y_test_pred, y_test)

        # Train_RMSE = mean_squared_error(y_train_pred, y_train)
        # Test_RMSE = mean_squared_error(y_test_pred, y_test)
        # train_RMSE = sqrt(Train_RMSE)
        # test_RMSE = sqrt(Test_RMSE)
        
        train_MAPE = mean_absolute_percentage_error(y_train_pred, y_train)
        test_MAPE = mean_absolute_percentage_error(y_test_pred, y_test)

        with mlflow.start_run():
            mlflow.log_param('max_depth', i)
            mlflow.log_param('min_sample_split', j)

            mlflow.log_metric('train_MeanAbsoluteError', train_MAE)
            mlflow.log_metric('test_MeanAbsoluteError', test_MAE)

            # mlflow.log_metric('train_MeanSquaredError', train_MSE)
            # mlflow.log_metric('test_MeanSquaredError', test_MSE)

            # mlflow.log_metric('train_RootMeanSquaredError', train_RMSE)
            # mlflow.log_metric('test_RootMeanSquaredError', test_RMSE)

            mlflow.log_metric('train_MeanAbsolutePercentageError', train_MAPE)
            mlflow.log_metric('test_MeanAbsolutePercentageError', test_MAPE)

            mlflow.sklearn.log_model(model_dt, 'model')
# decisiontree()

def knn():
    neighbour = [2,3,4,5,6,7,8,9,10]

    for i in neighbour:
        model_knn = KNeighborsRegressor(n_neighbors = i)
        model_knn.fit(x_train,y_train)

        y_train_pred = model_knn.predict(x_train)
        y_test_pred = model_knn.predict(x_test)

        res = pd.DataFrame(data = list(zip(y_train,y_train_pred,y_test,y_test_pred)), columns = ['y_train','y_train_pred','y_test','y_test_pred'])
        print(res)

        train_MAE = mean_absolute_error(y_train_pred, y_train)
        test_MAE = mean_absolute_error(y_test_pred, y_test)

        #   train_MSE = mean_squared_error(y_train_pred, y_train)
        #   test_MSE = mean_squared_error(y_test_pred, y_test)

        # Train_RMSE = mean_squared_error(y_train_pred, y_train)
        # Test_RMSE = mean_squared_error(y_test_pred, y_test)
        # train_RMSE = sqrt(Train_RMSE)
        # test_RMSE = sqrt(Test_RMSE)
        
        train_MAPE = mean_absolute_percentage_error(y_train_pred, y_train)
        test_MAPE = mean_absolute_percentage_error(y_test_pred, y_test)

        with mlflow.start_run():
            mlflow.log_param('n_neighbor', i)

            mlflow.log_metric('train_MeanAbsoluteError', train_MAE)
            mlflow.log_metric('test_MeanAbsoluteError', test_MAE)

            # mlflow.log_metric('train_MeanSquaredError', train_MSE)
            # mlflow.log_metric('test_MeanSquaredError', test_MSE)

            # mlflow.log_metric('train_RootMeanSquaredError', train_RMSE)
            # mlflow.log_metric('test_RootMeanSquaredError', test_RMSE)

            mlflow.log_metric('train_MeanAbsolutePercentageError', train_MAPE)
            mlflow.log_metric('test_MeanAbsolutePercentageError', test_MAPE)

            mlflow.sklearn.log_model(model_knn, 'model')
# knn()


df1 = df[(df['status']==1.0) | (df['status']==7.0)]
# print(df1['status'].value_counts())

X = df1.drop(['status'],axis = 1)
Y = df1['status']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

def randomforest_class():
    # n_estimators_val = [20, 40, 70, 90, 110, 150, 200, 300]
    # Max_depth = [2,5,8,11, 15, 20, 25, 30]
    # Min_sample_split = [2,5,7,9,12,13,15,20]

    n_estimators_val = [350, 400, 450, 500]
    Max_depth = [25, 30, 35, 40]
    Min_sample_split = [2,5,7,9]

    for i,j,k in zip(Max_depth,Min_sample_split,n_estimators_val):
        model_rf = RandomForestClassifier(criterion = 'entropy', n_estimators = k, max_depth = i, min_samples_split = j,  random_state=30)
        model_rf.fit(X_train,Y_train)

        y_train_pred = model_rf.predict(X_train)
        y_test_pred = model_rf.predict(X_test)

        # res = pd.DataFrame(data = list(zip(y_train,y_train_pred,y_test,y_test_pred)), columns = ['y_train','y_train_pred','y_test','y_test_pred'])
        # print(res)

        train_accuracy = accuracy_score(Y_train, y_train_pred)
        test_accuracy = accuracy_score(Y_test, y_test_pred)

        train_precision = precision_score(Y_train, y_train_pred)
        test_precision = precision_score(Y_test, y_test_pred)

        train_recall = recall_score(Y_train, y_train_pred)
        test_recall = recall_score(Y_test, y_test_pred)

        train_f1 = f1_score(Y_train, y_train_pred)
        test_f1 = f1_score(Y_test, y_test_pred)

        with mlflow.start_run():
            mlflow.log_param('n_estimators', k)
            mlflow.log_param('max_depth', i)
            mlflow.log_param('min_sample_split', j)

            mlflow.log_metric('train_accuracy',train_accuracy)
            mlflow.log_metric('test_accuracy',test_accuracy)
            mlflow.log_metric('train_precision',train_precision)
            mlflow.log_metric('test_precision',test_precision)
            mlflow.log_metric('train_recall',train_recall)
            mlflow.log_metric('test_recall',test_recall)
            mlflow.log_metric('train_f1',train_f1)
            mlflow.log_metric('test_f1',test_f1)
            

            mlflow.sklearn.log_model(model_rf, 'model')
# randomforest_class()

def decisiontree_class():
    Max_depth = [5,10,15,20, 25, 30, 35,50, 70, 85, 100]
    Min_sample_split = [40,35,30,25,20,15,10,9,8,5,2]
    # Min_sample_split = [2,5,8,9,10,15,20,25,30,35,40]

    for i,j in zip(Max_depth,Min_sample_split):
        model_dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = i, min_samples_split = j,  random_state=30)
        model_dt.fit(X_train,Y_train)

        y_train_pred = model_dt.predict(X_train)
        y_test_pred = model_dt.predict(X_test)

        # res = pd.DataFrame(data = list(zip(y_train,y_train_pred,y_test,y_test_pred)), columns = ['y_train','y_train_pred','y_test','y_test_pred'])
        # print(res)

        train_accuracy = accuracy_score(Y_train, y_train_pred)
        test_accuracy = accuracy_score(Y_test, y_test_pred)

        train_precision = precision_score(Y_train, y_train_pred)
        test_precision = precision_score(Y_test, y_test_pred)

        train_recall = recall_score(Y_train, y_train_pred)
        test_recall = recall_score(Y_test, y_test_pred)

        train_f1 = f1_score(Y_train, y_train_pred)
        test_f1 = f1_score(Y_test, y_test_pred)

        with mlflow.start_run():
            mlflow.log_param('max_depth', i)
            mlflow.log_param('min_sample_split', j)

            mlflow.log_metric('train_accuracy',train_accuracy)
            mlflow.log_metric('test_accuracy',test_accuracy)
            mlflow.log_metric('train_precision',train_precision)
            mlflow.log_metric('test_precision',test_precision)
            mlflow.log_metric('train_recall',train_recall)
            mlflow.log_metric('test_recall',test_recall)
            mlflow.log_metric('train_f1',train_f1)
            mlflow.log_metric('test_f1',test_f1)
            

            mlflow.sklearn.log_model(model_dt, 'model')
# decisiontree_class()