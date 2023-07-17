#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[99]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Import data

# In[100]:


df=pd.read_csv(r"D:\job\internship\bankruptcy\data.csv\data.csv")


# ### EDA

# In[101]:


pd.set_option('display.max_columns',None)


# In[102]:


df.head()


# In[103]:


df.info()


# In[104]:


df.describe()


# In[105]:


df.isnull().sum()


# In[106]:


df.columns


# In[107]:


sns.countplot(x='Bankrupt?', data=df)


# In[108]:


df[' Liability-Assets Flag'].unique()


# In[109]:


sns.countplot(x=' Liability-Assets Flag',data=df)


# In[110]:


sns.countplot(x=' Net Income Flag',data=df)


# We can note that the data is unbalanced here.

# In[111]:


corr_matrix = df.corr()
plt.figure(figsize=(40,40))
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# we can see that the following columns are highly correlated.
# 1) ROA(C) before interest and depreciation before interest and ROA(B) before interest and depreciation after tax
# 2) Pre-tax net Interest Rate and Continuous interest rate (after tax)
# 3) Net Income to Total Assets and ROA(A) before interest and % after tax
# etc.,

# But, here we cannot take insights effectively. Lets check other correlation methods to decide.

# In[116]:


company_corr = df.corr().loc[:, 'Bankrupt?'].to_frame()


# In[117]:


indices_to_remove = [' Liability-Assets Flag', ' Net Income Flag', 'Bankrupt?']
company_corr.drop(indices_to_remove, inplace=True)


# In[118]:


plt.figure(figsize=(8, 17))
sns.barplot(y=company_corr.index, x='Bankrupt?', data=company_corr)
plt.title("Pearson correlation with Bankruptcy")
plt.show()


# In[119]:


selected_features = company_corr[abs(company_corr['Bankrupt?']) > 0.10].index.tolist()


# In[120]:


selected_corr = df[selected_features].corr()


# In[121]:


# Plot correlation heatmap between the selected columns
plt.figure(figsize=(20, 10))
sns.heatmap(df[selected_features].corr(), annot=True, fmt=".2f")
plt.title("Correlation between selected columns")
plt.show()


# In[122]:


# Add 'Bankrupt?' to the selected columns set
selected_columns_set_linear = selected_features + ['Bankrupt?']


# In[123]:


df1=df[selected_columns_set_linear]


# In[124]:


imp_col=['Bankrupt?',' Borrowing dependency',' Current Liability to Current Assets',' Debt ratio %',' Net Value Per Share (A)',' Net profit before tax/Paid-in capital',
' Operating Gross Margin',
' Per Share Net profit before tax (Yuan ¥)',
' Persistent EPS in the Last Four Seasons',
' ROA(A) before interest and % after tax',
' Working Capital to Total Assets']


# In[125]:


sns.heatmap(df1[imp_col].corr(),annot=True)


# since, the dataset is imbalanced, we need to balance them using some techniques like oversampling/undersampling, ensemble methods, weighted loss methods.
# Here we shall use oversampling.

# ### xy split and PCA

# In[126]:


x=df.drop(['Bankrupt?'],axis=1)
y=df['Bankrupt?']


# since, we have too many columns, we need to take those columns which has high variance and contributes most to the dependent variable

# In[127]:


from sklearn.decomposition import PCA
pca = PCA(n_components= 10)
dataN = pca.fit_transform(x)
print(dataN)
dataN.shape


# In[128]:


y=pd.DataFrame(y)
new=np.concatenate((dataN,y),axis=1)
new=pd.DataFrame(new)
print(new.columns)


# In[129]:


print(new)
y=new[10]
x=new.drop([10],axis=1)
y.value_counts()


# ### oversampling

# In[130]:


from imblearn.over_sampling import RandomOverSampler
import imblearn
oversample = RandomOverSampler(sampling_strategy='minority')

# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy=1.0)

# fit and apply the transform
X_over, y_over = oversample.fit_resample(x, y)

y=y_over
x=X_over
y=pd.DataFrame(y)
y.value_counts()


# ### Train test split

# In[131]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)


# ### 1.decision tree

# In[132]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
import time
debut=time.time()
dtree.fit(x_train,y_train)
fin=time.time()-debut
prediction = dtree.predict(x_test)
y_pred_prob = dtree.predict_proba(x_test)[:, 1]


# ### evaluation for decision tree

# In[133]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,prediction)*100
print('Decision tree accuracy is: ',acc)


# In[134]:


from sklearn.metrics import roc_auc_score,roc_curve

# Calculate the AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred_prob)

# Print the AUC-ROC score
print("AUC-ROC Score:", auc_roc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[135]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,prediction)
sns.heatmap(cm,annot=True)
plt.title("Confusion matrix for decision tree")


# In[136]:


from sklearn.metrics import classification_report
cr=classification_report(y_test,prediction)
print(cr)


# ### 2. Random forest

# In[137]:


from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")
classifier.fit(x_train, y_train)
prediction1= classifier.predict(x_test)
y_pred_prob1 = classifier.predict_proba(x_test)[:, 1]


# ### evaluation for random forest

# In[138]:


from sklearn.metrics import accuracy_score
acc1=accuracy_score(y_test,prediction1)*100
print('random forest accuracy is: ',acc1)


# In[139]:


from sklearn.metrics import roc_auc_score,roc_curve

# Calculate the AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred_prob1)

# Print the AUC-ROC score
print("AUC-ROC Score:", auc_roc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob1)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[140]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,prediction1)
sns.heatmap(cm,annot=True)
plt.title("Confusion matrix for random forest")


# In[141]:


from sklearn.metrics import classification_report
cr=classification_report(y_test,prediction1)
print(cr)


# ### 3. Logistic regression

# In[142]:


from sklearn.linear_model import LogisticRegression
cl=LogisticRegression()
cl.fit(x_train, y_train)
prediction2= cl.predict(x_test)
y_pred_prob2 = cl.predict_proba(x_test)[:, 1]


# ### evaluation for logistic regression

# In[143]:


from sklearn.metrics import accuracy_score
acc2=accuracy_score(y_test,prediction2)*100
print('random forest accuracy is: ',acc2)


# In[144]:


from sklearn.metrics import roc_auc_score,roc_curve

# Calculate the AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred_prob2)

# Print the AUC-ROC score
print("AUC-ROC Score:", auc_roc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob2)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[145]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,prediction2)
sns.heatmap(cm,annot=True)
plt.title("Confusion matrix for random forest")


# In[146]:


from sklearn.metrics import classification_report
cr=classification_report(y_test,prediction2)
print(cr)


# ### We can see that random forest gives better accuracy.
