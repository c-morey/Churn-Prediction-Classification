import pandas as pd
import pickle

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix,classification_report, accuracy_score



df = pd.read_csv('Bank_Churners_Clean.csv')
df.info()
# Convert target variable to dummy variables
df['Attrition_Flag']= df['Attrition_Flag'].replace({"Existing Customer":0, "Attrited Customer":1})

# Select columns that have correlations more than 0.1
df = df[['Attrition_Flag','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Total_Revolving_Bal','Avg_Utilization_Ratio','Total_Trans_Amt',
           'Total_Relationship_Count','Total_Amt_Chng_Q4_Q1','Months_Inactive_12_mon','Contacts_Count_12_mon']]
# Define the X target and y features
X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Oversample the train dataset
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample (X_train,y_train)
print ("After SMOTE: % of 1s in label:", y_res.mean())


# Define the model with default parameters
rfc = RandomForestClassifier()
# Fit the model
rfc.fit(X_res,y_res)

# Create the predictions
predictions = rfc.predict(X_test)

# print the scores on training and test set
print('Training set score: {:.4f}'.format(rfc.score(X_res, y_res)))
print('Test set score: {:.4f}'.format(rfc.score(X_test, y_test)))

# print the classification report
print(classification_report(y_test,predictions))

print(type(classification_report))
# plot the confusion matrix
plot_confusion_matrix(rfc,X_test,y_test)

# saving the model
pickle_out = open("random_forest.pkl", mode = "wb")
pickle.dump(rfc, pickle_out)
pickle_out.close()
