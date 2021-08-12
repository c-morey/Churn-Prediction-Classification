import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


df = pd.read_csv('BankChurners.csv')

#Drop the CLIENTNUM and the last 2 columns
df = df.drop(['CLIENTNUM', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
             'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
            axis = 1)

df.info() #check overall view

# Create Pandas Profiling for detailed view
profile = ProfileReport(df, title="Pandas Profiling Report - Churn Prediction Dataset")
print(profile)
#Check the correlation among numeric columns
corr= df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr,annot= True,vmin=-0.5,vmax=1, cmap='coolwarm',linewidths=0.75)
plt.show()

#Group and plot certain Numerical variables together for a comparison study with the target variable
cols1 = df[['Total_Relationship_Count',
           'Months_Inactive_12_mon',
            'Contacts_Count_12_mon']].columns.tolist()
cols2 = df[['Credit_Limit',
           'Total_Revolving_Bal',
           'Avg_Open_To_Buy',
            'Avg_Utilization_Ratio'
            ]].columns.tolist()
cols3 = df[['Total_Trans_Amt',
           'Total_Trans_Ct',
           'Total_Ct_Chng_Q4_Q1',
            'Total_Amt_Chng_Q4_Q1']].columns.tolist()
cols4 = df[['Dependent_count','Customer_Age','Months_on_book']]

#Function that will plot a boxplot with the numeric columns and the target variable
def bi_plot(x):
    plt.figure(figsize=(9,7))
    for i,count in enumerate(x):
        plt.subplot(2,2,i+1)
        sns.boxplot(df['Attrition_Flag'],df[count],showmeans=True)
        plt.title('Attrition_Flag Vs '+count,fontsize=12,fontweight = 'bold')
        plt.tight_layout()
#call the function to plot the boxplots
bi_plot(cols1), bi_plot(cols2), bi_plot(cols3), bi_plot(cols4)

# Total_Trans_Ct Vs Total_Trans_Amt
plt.figure(figsize=(15,7))
sns.lineplot(df.Total_Trans_Ct,df.Total_Trans_Amt,hue=df.Attrition_Flag)
plt.show()

# Total_Ct_Chng_Q4_Q1 Vs Total_Amt_Chng_Q4_Q1
plt.figure(figsize=(8,6))
sns.scatterplot(x='Total_Ct_Chng_Q4_Q1', y='Total_Amt_Chng_Q4_Q1',hue='Attrition_Flag',
             data=df)
plt.show()

# Avg_Open_To_Buy Vs Credit_Limit -- Their correlation is 1
plt.figure(figsize=(12,10), dpi=100)
sns.scatterplot(x='Avg_Open_To_Buy', y='Credit_Limit',hue='Attrition_Flag',
             data=df)
plt.show()

# Same features with lineplot, to see more in detail
plt.figure(figsize=(12,10))
sns.lineplot(df.Avg_Open_To_Buy,df.Credit_Limit,hue=df.Attrition_Flag)
plt.show()

# Compute crosstabs to make sure the conditions for the Chi-squared test of independence are met
df_cat = df[['Gender', 'Education_Level','Marital_Status','Income_Category','Card_Category']]
cols=df_cat.columns.to_list()
for i in range(len(cols)-1):
    for j in range(i+1, len(cols)):
        print(pd.crosstab(df[cols[i]], df[cols[j]]), "\n")

# Definition of Cramer's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

# Cramer's Vs matrix
rows = []
for var1 in df_cat.columns:
    col = []
    for var2 in df_cat.columns :
        col.append(round(cramers_v(df_cat[var1], df_cat[var2]),2))
    rows.append(col)
# Make a dataframe from the cramer matrix
cramer_matrix = pd.DataFrame(np.array(rows), columns = df_cat.columns, index = df_cat.columns)
print("Cramer matrix")
 # check the dataframe
print(cramer_matrix)

