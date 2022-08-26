# imports 
from pyforest import * 
from sklearn import datasets
from PIL import Image
import streamlit as st
import io
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


#### Starting app
# title and details 
st.title("Universal Classifier Application")
st.subheader(" This is a simple streamlit application where we can analyse ,transform and train different dataset with ease . Here we will be using different classifier models and evaluate .")
st.text(" ")
st.write(" #### lets explore the dataset !")

# suppressing warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# adding sidebar
st.text(" ")
dataset_type = st.sidebar.selectbox("Dataset Type :",("Preloaded sklearn dataset","Custom dataset"))

# defining dataset loading function

def get_dataset(name):
    if name == "Iris data":
        data = datasets.load_iris()
    elif name == "Wine data":
        data = datasets.load_wine()
    else:
        data = datasets.load_digits(n_class=10)
    
    target = data['target']
    df = pd.DataFrame(data= np.c_[data['data'], data['target']],columns= data['feature_names'] + ['target'])

    return df,target

df = pd.DataFrame({}) # empty dataframe initiation
if dataset_type == 'Custom dataset':
    file = st.file_uploader("Choose the dataset ",type = "csv")
    if file is not None:
        df = pd.read_csv(file,header = 0)
        st.success("file uploaded !")
        st.text(" ")
        st.text(" ")
        st.dataframe(df)
        st.text(" ")
        st.text(" ")
        buffer = io.StringIO() 
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(' ')
        st.write("#### Description : ")
        st.text(s)
    else :
        st.markdown("upload csv file")
else:
    dataset_name= st.selectbox("Select the dataset",("Iris data","Wine data","digits data (upto 10)"))
    df,target = get_dataset(dataset_name)
    st.dataframe(df)
    st.text(" ")


# creating a main sidebar function to handle different operations 

def main () :
    activities = ['Explanatory Data Analysis','Visualization','Model Creation']
    option = st.sidebar.selectbox("Choose your operation : ", activities)
    df1 = pd.DataFrame({}) # df initialization for all !

    # EDA
    if option == 'Explanatory Data Analysis':
        st.text(" ")
        st.text(" ")
        st.text(" ")
        st.subheader("Explanatory Data Analysis (EDA) ")
        st.text(" ")
        st.text(" ")
        if st.checkbox("Data Shape:"):
            st.write(df.shape)
        if st.checkbox("Data Columns Name:"):
            st.write(df.columns)
        st.text(" ")
        st.text(" ")
        st.write("#### Do you want to filter attributes for analysis ")
        st.text("select No to consider entire data set")
        ans1 = st.selectbox("response" ,['Yes','No'])
        
        if ans1 == 'Yes':
            selected_col = st.multiselect("filter Columns : ",df.columns)
            st.text(" ")
            
            df1 = df[selected_col]
            if not df1.empty:
                st.text("selected dataframe")
                st.dataframe(df1)
                if st.checkbox("Data summary (categorical attributes are ignored here )"):
                    st.write(df1.describe())
                st.text(" ")
                st.text(" ")
                if st.checkbox("Data Information with null values ,data types and shape"):
                    buffer = io.StringIO() 
                    df1.info(buf=buffer)
                    s = buffer.getvalue()
                    st.text(' ')
                    st.text(s)
                st.text(" ")
                st.text(" ")
                if st.checkbox("Data Correlation"):
                    st.write(df1.corr())
        else:
            df1 = df.copy()
            st.text(" ")
            st.text("selected dataframe")
            st.dataframe(df1)
            st.text(" ")
            st.text(" ")
            if st.checkbox("Data summary (categorical attributes are ignored here )"):
                st.write(df1.describe())
            st.text(" ")
            st.text(" ")
            if st.checkbox("Data Information with null values ,data types and shape"):
                buffer = io.StringIO() 
                df1.info(buf=buffer)
                s = buffer.getvalue()
                st.text(' ')
                st.text(s)
            st.text(" ")
            st.text(" ")
            if st.checkbox("Data Correlation"):
                st.write(df1.corr())
        

    # Visaulization
    elif option == 'Visualization':
        df1 = df.copy()
        st.write("#### Do you want to filter attributes for Visualization ")
        st.text("select No to consider entire data set")
        ans1 = st.selectbox("response" ,['Yes','No'])
        
        if ans1 == 'Yes':
            selected_col = st.multiselect("filter Columns : ",df.columns)
            st.text(" ")
            
            df1 = df[selected_col]
            if not df1.empty:
                st.text("selected dataframe")
                st.dataframe(df1)
                if st.checkbox("Correlation Heatmap (categorical attributes are ignored here )"):
                    sns.heatmap(df1.corr(),cmap = 'viridis',square = True,annot = True,vmax =1)
                    st.pyplot()
                if st.checkbox("Pair Plot"):
                    sns.pairplot(df1,diag_kind = 'kde')
                    st.pyplot()
                if st.checkbox("Box Plot"):
                    for col in list(df1.describe().columns):
                        sns.boxplot(df1[col],orient = 'h')
                        st.pyplot()
                if st.checkbox("Count Plot Pie Chart"):
                    pie_columns = st.selectbox("Select Columns : ",df1.columns.to_list())
                    piechart = df1[pie_columns].value_counts().plot.pie(autopct = "1.1f%%")
                    st.write(piechart)
                    st.pyplot()
                    
        else:
            df1 = df.copy()
            st.text(" ")
            st.text("selected dataframe")
            st.dataframe(df1)
            st.text(" ")
            st.text(" ")
            if st.checkbox("Correlation Heatmap (categorical attributes are ignored here )"):
                sns.heatmap(df.corr(),cmap = 'viridis',square = True,annot = True,vmax =1)
                st.pyplot()
            if st.checkbox("Pair Plot"):
                sns.pairplot(df,diag_kind = 'kde')
                st.pyplot()
            if st.checkbox("Box Plot"):
                for col in list(df.describe().columns):
                    sns.boxplot(df[col],orient = 'h')
                    st.pyplot()
            if st.checkbox("Count Plot Pie Chart"):
                pie_columns = st.selectbox("Select Columns : ",df.columns.to_list())
                piechart = df[pie_columns].value_counts().plot.pie(autopct = "1.1f%%")
                st.write(piechart)
                st.pyplot()

    #creating ML models             
    elif option == 'Model Creation':
        df1 = df.copy()
        st.write("#### Do you want to filter attributes for Model Creation ")
        st.text("select No to consider entire data set")
        ans1 = st.selectbox("response" ,['Yes','No'])
        
        if ans1 == 'Yes':
            st.text("Always select the target variable at the end !")
            selected_col = st.multiselect("filter Columns : ",df.columns)
            st.text(" ")
            df1 = df[selected_col]
            if not df1.empty:
                st.text("selected dataframe")
                st.dataframe(df1)
                st.text(" ")
                st.text(" ")
                target_var = st.selectbox("Select the target variable",df1.columns)
            
                # model selection
                classifier_name = st.selectbox("Select the classifier : ",['KNN','Logistic Regression','XGBoost','Catboost'])

                # creating parameters
                def create_params(classifier_name):
                    param = {}
                    if classifier_name == 'KNN':
                        param['n_neighbors'] = st.slider('K',1,20)
                    if classifier_name == 'Logistic Regression':
                        param['solver']  = st.selectbox('Solver',['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
                    if classifier_name == 'Catboost':
                        param['loss_function'] = st.selectbox('loss function',[
                        'Logloss','MAE','CrossEntropy','MultiClass'])
                        param['eval_metric'] = st.selectbox('eval_metric',[
                        'Logloss','MAE','CrossEntropy','MultiClass','AUC','Precison','Recall','F1','Accuracy'])
                    if classifier_name == 'XGBoost':
                        param['n_estimators'] = st.slider('n_estimators ',10,1000)
                        param['max_depth'] = st.slider('max_depth  ',2,10)
                        param['verbosity'] = st.slider('verbosity  ',0,3)
                        param['booster'] = st.selectbox('booster   ',['gbtree', 'gblinear', 'dart'])
                        param['learning_rate'] = st.slider('learning_rate  ',0.01,5.00)
                    return param

                param = create_params(classifier_name)

                ans2 = st.selectbox('Do you want to encode columns',[' ','yes','No'])
                # encoding
                if ans2 == 'yes':
                    columns = st.multiselect('Select columns to encode',df1.columns)
                    if classifier_name == 'Catboost':
                        st.text([df1.columns.get_loc(col)for col in columns])
                        param['cat_features'] = [df1.columns.get_loc(col) for col in columns]
                        # splitting df into x and y 
                        X = df1.drop(target_var,axis = 1)
                        y = df1[target_var]

                        #random seed
                        random_seed = st.slider("Select the random state : ",1,50)
                        param['random_seed'] = random_seed

                        # splitting it into train and test
                        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = random_seed)
            
                        #standardize data
                        """if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                            X_train = StandardScaler().fit_transform(X_train)
                            X_test = StandardScaler().fit_transform(X_test)"""
                    
                        #model
                        if st.checkbox("Train model "):
                            model = CatBoostClassifier(**param).fit(X_train,y_train,eval_set = (X_test,y_test),use_best_model = True)
                            #feature importance
                            st.dataframe(model.get_feature_importance(prettified = True))
                            sns.barplot(data = model.get_feature_importance(prettified = True), x = 'Feature Id', y = 'Importances')
                            plt.xticks(rotation = 45)
                            plt.title('Feature Importance')
                            st.pyplot()

                            st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test),average=None))

                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                st.pyplot()
                            if st.checkbox("Precision Score "):
                                st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("Recall Score "):
                                st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("F1 Score "):
                                st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))

                    else:
                        st.text([df1.columns.get_loc(col)for col in columns])
                        cat_features = [df1.columns.get_loc(col) for col in columns]
                        columnTransformer = ColumnTransformer([('encoder',
                                            OneHotEncoder(),
                                            cat_features)],
                                        remainder='passthrough')
                    
                        # splitting df into x and y 
                        X = df1.drop(target_var,axis = 1)
                        y = df1[target_var]

                        #random seed
                        random_seed = st.slider("Select the random state : ",1,50)
                        param['random_seed'] = random_seed

                        # splitting it into train and test
                        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = random_seed)
                        X_train = columnTransformer.fit_transform(X_train)
                        X_test = columnTransformer.transform(X_test)
            
                        #standardize data
                        if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                            X_train = StandardScaler().fit_transform(X_train)
                            X_test = StandardScaler().fit_transform(X_test)
                    
                        #model
                        if st.checkbox("Train model "):
                            if classifier_name == 'XGBoost':
                                model = xgb.XGBClassifier(learning_rate = param['learning_rate'],n_estimators = param['n_estimators'],max_depth = param['max_depth'],verbosity = param['verbosity'],booster = param['booster']).fit(X_train,y_train)
                                #feature importance
                                
                                xgb.plot_importance(model)
                                plt.xticks(rotation = 45)
                                plt.title('Feature Importance')
                                st.pyplot()

                                st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                                if st.checkbox("Precision Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                if st.checkbox("Recall Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                    st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test)))
                                if st.checkbox("F1 Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                    st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test)))

                            elif classifier_name == 'Logistic Regression':
                                model = LogisticRegression(solver = param['solver']).fit(X_train,y_train)
                                st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                                if st.checkbox("Precision Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test)))
                                if st.checkbox("Recall Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                    st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test)))
                                if st.checkbox("F1 Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                    st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test)))

                            else :
                                model = KNeighborsClassifier(n_neighbors = param['n_neighbors']).fit(X_train,y_train)
                                st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                                if st.checkbox("Precision Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test)))
                                if st.checkbox("Recall Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                    st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test)))
                                if st.checkbox("F1 Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                    st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test)))

                else:
                    if classifier_name == 'Catboost':
                        # splitting df into x and y 
                        X = df1.drop(target_var,axis = 1)
                        y = df1[target_var]

                        #random seed
                        random_seed = st.slider("Select the random state : ",1,50)
                        param['random_seed'] = random_seed

                        # splitting it into train and test
                        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = random_seed)
            
                        #standardize data
                        """if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                            X_train = StandardScaler().fit_transform(X_train)
                            X_test = StandardScaler().fit_transform(X_test)"""
                    
                        #model
                        if st.checkbox("Train model "):
                            model = CatBoostClassifier(**param).fit(X_train,y_train,eval_set = (X_test,y_test),use_best_model = True)
                            #feature importance
                            st.dataframe(model.get_feature_importance(prettified = True))
                            sns.barplot(data = model.get_feature_importance(prettified = True), x = 'Feature Id', y = 'Importances')
                            plt.xticks(rotation = 45)
                            plt.title('Feature Importance')
                            st.pyplot()

                            st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))

                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                            if st.checkbox("Precision Score "):
                                st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("Recall Score "):
                                st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("F1 Score "):
                                st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))

                    else:
                        
                        # splitting df into x and y 
                        X = df1.drop(target_var,axis = 1)
                        y = df1[target_var]

                        #random seed
                        random_seed = st.slider("Select the random state : ",1,50)
                        param['random_seed'] = random_seed

                        # splitting it into train and test
                        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = random_seed)
                
                        #standardize data
                        if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                            X_train = StandardScaler().fit_transform(X_train)
                            X_test = StandardScaler().fit_transform(X_test)
                    
                        #model
                        if st.checkbox("Train model "):
                            if classifier_name == 'XGBoost':
                                model = xgb.XGBClassifier(learning_rate = param['learning_rate'],n_estimators = param['n_estimators'],max_depth = param['max_depth'],verbosity = param['verbosity'],booster = param['booster']).fit(X_train,y_train)
                                #feature importance
                                xgb.plot_importance(model)
                                plt.xticks(rotation = 45)
                                plt.title('Feature Importance')
                                st.pyplot()

                                st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                                if st.checkbox("Precision Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                if st.checkbox("Recall Score "):
                                    st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                                if st.checkbox("F1 Score "):
                                    st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))

                            elif classifier_name == 'Logistic Regression':
                                model = LogisticRegression(solver = param['solver']).fit(X_train,y_train)
                                st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                                if st.checkbox("Precision Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                if st.checkbox("Recall Score "):
                                    st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                                if st.checkbox("F1 Score "):
                                    st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))

                            else :
                                model = KNeighborsClassifier(n_neighbors = param['n_neighbors']).fit(X_train,y_train) 
                                st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                                st.text(" ")
                                st.text(" ")
                                if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                                if st.checkbox("Precision Score "):
                                    st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                                if st.checkbox("Recall Score "):
                                    st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                                if st.checkbox("F1 Score "):
                                    st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))
                    
        else:
            df1 = df.copy()
            st.text(" ")
            st.text("selected dataframe")
            st.dataframe(df1)
            st.text(" ")
            st.text(" ")
            target_var = st.selectbox("Select the target variable",df1.columns)
            
            # model selection
            classifier_name = st.selectbox("Select the classifier : ",['KNN','Logistic Regression','XGBoost','Catboost'])

            # creating parameters
            def create_params(classifier_name):
                param = {}
                if classifier_name == 'KNN':
                    param['n_neighbors'] = st.slider('K',1,20)
                if classifier_name == 'Logistic Regression':
                    param['solver']  = st.selectbox('Solver',['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
                if classifier_name == 'Catboost':
                    param['loss_function'] = st.selectbox('loss function',[
                    'Logloss','MAE','CrossEntropy','MultiClass'])
                    param['eval_metric'] = st.selectbox('eval_metric',[
                    'Logloss','MAE','CrossEntropy','MultiClass','AUC','Precison','Recall','F1','Accuracy'])
                if classifier_name == 'XGBoost':
                    param['n_estimators'] = st.slider('n_estimators ',10,1000)
                    param['max_depth'] = st.slider('max_depth  ',2,10)
                    param['verbosity'] = st.slider('verbosity  ',0,3)
                    param['booster'] = st.selectbox('booster   ',['gbtree', 'gblinear', 'dart'])
                    param['learning_rate'] = st.slider('learning_rate  ',0.01,5.00)
                return param

            param = create_params(classifier_name)

            ans2 = st.selectbox('Do you want to encode columns',[' ','yes','No'])
            # encoding
            if ans2 == 'yes':
                columns = st.multiselect('Select columns to encode',df1.columns)
                if classifier_name == 'Catboost':
                    st.text([df1.columns.get_loc(col)for col in columns])
                    param['cat_features'] = [df1.columns.get_loc(col) for col in columns]
                    # splitting df into x and y 
                    X = df1.drop(target_var,axis = 1)
                    y = df1[target_var]

                    #random seed
                    random_seed = st.slider("Select the random state : ",1,50)
                    param['random_seed'] = random_seed

                    # splitting it into train and test
                    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = random_seed)
            
                    #standardize data
                    """if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                        X_train = StandardScaler().fit_transform(X_train)
                        X_test = StandardScaler().fit_transform(X_test)"""
                    
                    #model
                    if st.checkbox("Train model "):
                        model = CatBoostClassifier(**param).fit(X_train,y_train,eval_set = (X_test,y_test),use_best_model = True)
                         #feature importance
                        st.dataframe(model.get_feature_importance(prettified = True))
                        sns.barplot(data = model.get_feature_importance(prettified = True), x = 'Feature Id', y = 'Importances')
                        plt.xticks(rotation = 45)
                        plt.title('Feature Importance')
                        st.pyplot()

                        st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))

                        st.text(" ")
                        st.text(" ")
                        if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                        if st.checkbox("Precision Score "):
                            st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                        if st.checkbox("Recall Score "):
                            st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                        if st.checkbox("F1 Score "):
                            st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))

                else:
                    st.text([df1.columns.get_loc(col)for col in columns])
                    cat_features = [df1.columns.get_loc(col) for col in columns]
                    columnTransformer = ColumnTransformer([('encoder',
                                        OneHotEncoder(),
                                        cat_features)],
                                      remainder='passthrough')
                    
                    # splitting df into x and y 
                    X = df1.drop(target_var,axis = 1)
                    y = df1[target_var]

                    #random seed
                    random_seed = st.slider("Select the random state : ",1,50)
                    param['random_seed'] = random_seed

                    # splitting it into train and test
                    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = random_seed)
                    X_train = columnTransformer.fit_transform(X_train)
                    X_test = columnTransformer.transform(X_test)
            
                    #standardize data
                    if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                        X_train = StandardScaler().fit_transform(X_train)
                        X_test = StandardScaler().fit_transform(X_test)
                    
                    #model
                    if st.checkbox("Train model "):
                        if classifier_name == 'XGBoost':
                            model = xgb.XGBClassifier(learning_rate = param['learning_rate'],n_estimators = param['n_estimators'],max_depth = param['max_depth'],verbosity = param['verbosity'],booster = param['booster']).fit(X_train,y_train)
                            #feature importance
                            
                            xgb.plot_importance(model)
                            plt.xticks(rotation = 45)
                            plt.title('Feature Importance')
                            st.pyplot()

                            st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                            if st.checkbox("Precision Score "):
                                st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("Recall Score "):
                                st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("F1 Score "):
                                st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))

                        elif classifier_name == 'Logistic Regression':
                            model = LogisticRegression(solver = param['solver']).fit(X_train,y_train)
                            st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                            if st.checkbox("Precision Score "):
                                st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("Recall Score "):
                                st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("F1 Score "):
                                st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))
                        
                        else :
                            model = KNeighborsClassifier(n_neighbors = param['n_neighbors']).fit(X_train,y_train)
                            st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                            if st.checkbox("Precision Score "):
                                st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("Recall Score "):
                                st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("F1 Score "):
                                st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))


            else:
                if classifier_name == 'Catboost':
                    # splitting df into x and y 
                    X = df1.drop(target_var,axis = 1)
                    y = df1[target_var]

                    #random seed
                    random_seed = st.slider("Select the random state : ",1,50)
                    param['random_seed'] = random_seed

                    # splitting it into train and test
                    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = random_seed)
            
                    #standardize data
                    """if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                        X_train = StandardScaler().fit_transform(X_train)
                        X_test = StandardScaler().fit_transform(X_test)"""
                    
                    #model
                    if st.checkbox("Train model "):
                        model = CatBoostClassifier(**param).fit(X_train,y_train,eval_set = (X_test,y_test),use_best_model = True)
                         #feature importance
                        st.dataframe(model.get_feature_importance(prettified = True))
                        sns.barplot(data = model.get_feature_importance(prettified = True), x = 'Feature Id', y = 'Importances')
                        plt.xticks(rotation = 45)
                        plt.title('Feature Importance')
                        st.pyplot()

                        st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                        st.text(" ")
                        st.text(" ")
                        if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                        if st.checkbox("Precision Score "):
                            st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                        if st.checkbox("Recall Score "):
                            st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                        if st.checkbox("F1 Score "):
                            st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))

                else:
                    
                    # splitting df into x and y 
                    X = df1.drop(target_var,axis = 1)
                    y = df1[target_var]

                    #random seed
                    random_seed = st.slider("Select the random state : ",1,50)
                    param['random_seed'] = random_seed

                    # splitting it into train and test
                    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = random_seed)
            
                    #standardize data
                    if st.checkbox("Do you want to standardize the data (RECOMMENDED)"):
                        X_train = StandardScaler().fit_transform(X_train)
                        X_test = StandardScaler().fit_transform(X_test)
                    
                    #model
                    if st.checkbox("Train model "):
                        if classifier_name == 'XGBoost':
                            model = xgb.XGBClassifier(learning_rate = param['learning_rate'],n_estimators = param['n_estimators'],max_depth = param['max_depth'],verbosity = param['verbosity'],booster = param['booster']).fit(X_train,y_train)
                            #feature importance

                            xgb.plot_importance(model)
                            plt.xticks(rotation = 45)
                            plt.title('Feature Importance')
                            st.pyplot()

                            st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                            if st.checkbox("Precision Score "):
                                st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("Recall Score "):
                                st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("F1 Score "):
                                st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))


                        elif classifier_name == 'Logistic Regression':
                            model = LogisticRegression(solver = param['solver']).fit(X_train,y_train)
                            st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                            if st.checkbox("Precision Score "):
                                st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("Recall Score "):
                                st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("F1 Score "):
                                st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))


                        else :
                            model = KNeighborsClassifier(n_neighbors = param['n_neighbors']).fit(X_train,y_train)  
                            st.write("ACCURACY : ",metrics.accuracy_score(y_test,model.predict(X_test)))  
                            st.text(" ")
                            st.text(" ")
                            if st.checkbox("Confusion Matrix "):
                                    sns.heatmap(metrics.confusion_matrix(y_test,model.predict(X_test)),cmap = 'viridis',square = True,annot = True,vmax =1)
                                    st.pyplot()
                            if st.checkbox("Precision Score "):
                                st.write("Precision Score :",metrics.precision_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("Recall Score "):
                                st.write("Recall Score :",metrics.recall_score(y_test,model.predict(X_test),average=None))
                            if st.checkbox("F1 Score "):
                                st.write("F1 Score :",metrics.f1_score(y_test,model.predict(X_test),average=None))







            



# main function called 

if __name__ == '__main__':
    if not df.empty:
        main()