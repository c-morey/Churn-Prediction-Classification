import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import pickle
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

matplotlib.use("Agg")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Bank Customers Churn Prediction App")
st.markdown( "This app aims to predict **customer churn**. The model is trained with the Credit Card customers dataset on [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers).")
st.markdown("For more info: [Check GitHub](https://github.com/c-morey/Churn-Prediction-Classification)")

st.image('/Users/cerenmorey/Desktop/At-risk-customers.png')

data_url = "./Bank_Churners_Clean.csv"
data_unclean_url = "./BankChurners.csv"

# Creating a side bar for users to explore
st.sidebar.markdown("## Side Bar")
st.sidebar.markdown("Use this panel to explore the dataset, create viz, and make predictions.")

df = pd.read_csv(data_url)
df_unclean = pd.read_csv(data_unclean_url)


# Showing the original raw data
if st.checkbox("Here, you can check the raw data", False):
    st.subheader('Raw data')
    st.write(df_unclean)

st.title('Quick  Exploration')
st.sidebar.subheader('Quick  Exploration')
st.markdown("Tick the box on the side panel to explore the dataset.")
if st.sidebar.checkbox('Basic info'):
    if st.sidebar.checkbox('Dataset Quick Look'):
        st.subheader('Dataset Quick Look:')
        st.write(df.head(10))
    if st.sidebar.checkbox("Show Columns"):
        st.subheader('Show Columns List')
        all_columns = df.columns.to_list()
        st.write(all_columns)

    if st.sidebar.checkbox('Statistical Description'):
        st.subheader('Statistical Data Descripition')
        st.write(df.describe())
    if st.sidebar.checkbox('Missing Values'):
        st.subheader('Missing values')
        st.write(df.isnull().sum())

# Visualization part
st.title('Explore Data with Visualization')
st.markdown('Tick the box on the side panel to create your own Visualization and explore the data.')
st.sidebar.subheader('Explore with Visualization')
if st.sidebar.checkbox('Data Visualization'):

    if st.sidebar.checkbox('Count Plot'):
        st.subheader('Count Plot')
        st.info("If error, please adjust column name on side panel.")
        column_count_plot = st.sidebar.selectbox(
            "Choose a column to plot count. Try Selecting Categorical Columns (e.g. Gender) ", df.columns)
        hue_opt = st.sidebar.selectbox("Optional categorical variables (countplot hue). Try Selecting Attrition Flag ",
                                       df.columns.insert(0, None))

        fig = sns.countplot(x=column_count_plot, data=df, hue=hue_opt)
        st.pyplot()

    if st.sidebar.checkbox('Histogram | Distplot'):
        st.subheader('Histogram | Distplot')
        st.info("If error, please adjust column name on side panel.")
        if st.checkbox('Dist plot'):
            column_dist_plot = st.sidebar.selectbox(
                "Optional categorical variables (countplot hue). Try Selecting Body Mass", df.columns)
        fig = sns.distplot(df[column_dist_plot])
        st.pyplot()

    if st.sidebar.checkbox('Boxplot'):
        st.subheader('Boxplot')
        st.info("If error, please adjust column name on side panel.")
        column_box_plot_X = st.sidebar.selectbox("X (Choose a column). Try Selecting Attrition Flag:",
                                                 df.columns.insert(0, None))
        column_box_plot_Y = st.sidebar.selectbox("Y (Choose a column - only numerical). Try Selecting Total_Trans_Ct",
                                                 df.columns)

        fig = sns.boxplot(x=column_box_plot_X, y=column_box_plot_Y, data=df, palette="Set3")
        st.pyplot()

    if st.sidebar.checkbox('Correlation Map'):
        st.subheader('Correlation Map')
        st.info("If error, please adjust column name on side panel.")

        plt.figure(figsize=(25, 20), dpi=200)
        fig = sns.heatmap(df.corr(), annot=True, vmin=-0.5, vmax=1, cmap='coolwarm', linewidths=0.75)
        st.pyplot()

# Give prediction based on user input
st.title('Explore Model')
st.markdown('Tick the box on the side panel to explore different model socres, or predict if the customer will churn or not.')
st.sidebar.subheader('Explore Model')
if st.sidebar.checkbox('Features'):
    def user_input_features():
        Total_Relationship_Count = st.sidebar.slider('Total Relationship Count', 0, 10)
        Months_Inactive_12_mon = st.sidebar.slider('Months Inactive 12_month', 0, 12)
        Contacts_Count_12_mon = st.sidebar.slider('Contacts Count 12_month', 0, 10)
        Total_Revolving_Bal = st.sidebar.slider('Total Revolving Balance', 0, 2600)
        Total_Amt_Chng_Q4_Q1 = st.sidebar.slider('Total Amount Change_Q4_Q1', 0, 4)
        Total_Trans_Amt = st.sidebar.slider('Total Transaction Amount', 0, 19000)
        Total_Trans_Ct = st.sidebar.slider('Total Transaction Count', 0, 140)
        Total_Ct_Chng_Q4_Q1 = st.sidebar.slider('Total Change Transaction Count_Q4_Q1', 0, 5)
        Avg_Utilization_Ratio = st.sidebar.slider('Average Card Utilization Ratio', 0, 5)
        data = {'Total Relationship Count': Total_Relationship_Count,
                'Months Inactive 12_month': Months_Inactive_12_mon,
                'Contacts Count 12_month': Contacts_Count_12_mon,
                'Total Revolving Balance': Total_Revolving_Bal,
                'Total Amount Change_Q4_Q1': Total_Amt_Chng_Q4_Q1,
                'Total Transaction Amount': Total_Trans_Amt,
                'Total Transaction Count': Total_Trans_Ct,
                'Total Change Transaction Count_Q4_Q1': Total_Ct_Chng_Q4_Q1,
                'Average Card Utilization Ratio': Avg_Utilization_Ratio}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    # Reads in saved classification model
    load_rfc = pickle.load(open('random_forest.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_rfc.predict(input_df)
    prediction_proba = load_rfc.predict_proba(input_df)

    st.subheader('Prediction')
    customer_types = np.array(['Existing Customer', 'Attrited Customer'])
    st.write(customer_types[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

if st.sidebar.checkbox('Choose Model'):
    st.sidebar.selectbox('Choose a model to check their performance',['Random Forest Classifier', 'Logistic Regression','Support Vector Machines'])
    if ('Random Forest Classifier'):
        st.subheader("Classification Report")
        df['Attrition_Flag'] = df['Attrition_Flag'].replace({"Existing Customer": 0, "Attrited Customer": 1})
        df = df[
            ['Attrition_Flag', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Total_Revolving_Bal', 'Avg_Utilization_Ratio',
             'Total_Trans_Amt',
             'Total_Relationship_Count', 'Total_Amt_Chng_Q4_Q1', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon']]
        X = df.drop('Attrition_Flag', axis=1)
        y = df['Attrition_Flag']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        # Oversample the train dataset
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        rfc = RandomForestClassifier()
        rfc.fit(X_res, y_res)
        predictions = rfc.predict(X_test)

        report = classification_report(y_test, predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.table(report_df)


        # plot the confusion matrix
        st.subheader("Confusion Matrix")
        fig5 = plt.figure()
        conf_matrix = confusion_matrix(rfc.predict(X_test), y_test)
        sns.heatmap(conf_matrix, annot=True, fmt = 'g')
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        st.pyplot(fig5)



