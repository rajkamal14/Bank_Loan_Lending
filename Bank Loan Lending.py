#!/usr/bin/env python
# coding: utf-8

# # Capstone Project - Bank Loan Lending

# In[1]:


import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_columns', 100)


# In[3]:


os.chdir("D:\\Imarticus\\Python Cls\\Python Project - Bank Lending")
os.getcwd()


# # Importing Dataset 

# In[4]:


data= pd.read_csv("XYZCorp_LendingData.txt",delimiter = "\t" )


# In[5]:


data.head(2)


# In[6]:


data.insert(0, 'Sl.No', data.index)
data.set_index('Sl.No', inplace = True)


# In[7]:


data.shape


# In[8]:


data.info()


# # checking for Missing values in the dataset

# In[9]:


pd.DataFrame(data).isnull().sum()


# # Finding the columns having more than 25% missing data's

# In[10]:


missing_cols_name=data.loc[:, data.isnull().any()].columns
missing_cols_name


# In[11]:


cols= data.columns[(data.isnull().sum()/len(data))*100>25]
cols.shape


# # droping columns which has more than 25% missing data's

# In[12]:


new_data= data.drop(columns=cols, axis=1)
new_data.dtypes


# # Missing Value Treatment

# In[13]:


pd.DataFrame(new_data).isnull().sum()


# # filling the missing values in categorical columns using mode

# In[14]:


new_data['emp_length'] = new_data['emp_length'].fillna(new_data['emp_length'].mode()[0])
new_data['emp_title'] = new_data['emp_title'].fillna(new_data['emp_title'].mode()[0])
new_data['title'] = new_data['title'].fillna(new_data['title'].mode()[0])
new_data['last_pymnt_d'] = new_data['last_pymnt_d'].fillna(new_data['last_pymnt_d'].mode()[0])
new_data['last_credit_pull_d'] = new_data['last_credit_pull_d'].fillna(new_data['last_credit_pull_d'].mode()[0])


# # Outlier checking

# In[15]:


sns.boxplot(x=data['revol_util'])


# In[16]:


sns.boxplot(x=data['tot_coll_amt'])


# In[17]:


sns.boxplot(x=data['tot_cur_bal'])


# In[18]:


sns.boxplot(x=data['total_rev_hi_lim'])


# In[19]:


sns.boxplot(x=data['collections_12_mths_ex_med'])


# # filling the numerical columns which has missing data's, and they are having outliers so using median to fill the data

# In[20]:


new_data['tot_coll_amt'] = new_data['tot_coll_amt'].fillna(new_data.tot_coll_amt.median())
new_data['tot_cur_bal'] = new_data['tot_cur_bal'].fillna(new_data.tot_cur_bal.median())
new_data['total_rev_hi_lim'] = new_data['total_rev_hi_lim'].fillna(new_data.total_rev_hi_lim.median())
new_data['collections_12_mths_ex_med'] = new_data['collections_12_mths_ex_med'].fillna(new_data.collections_12_mths_ex_med.median())
new_data['revol_util'] = new_data['revol_util'].fillna(new_data.revol_util.median())


# In[21]:


pd.DataFrame(new_data).isnull().sum()


# In[22]:


new_data.shape


# # coorelation Matrix

# In[23]:


corr = new_data.corr()
plt.figure(figsize=(30,30))
sns.heatmap(corr, annot = True , cmap="BuPu")


# # Exploratory Data Analysis

# # Categorical attributes Visualization

# In[24]:


sns.countplot(new_data['default_ind'])


# In[25]:


new_data['default_ind'].value_counts()


# # Numerical attributes visualization 

# In[26]:


sns.distplot(new_data['loan_amnt'])


# In[27]:


sns.distplot(new_data['funded_amnt'])


# In[28]:


sns.distplot(new_data['funded_amnt_inv'])


# In[29]:


sns.distplot(new_data['installment'])


# In[30]:


sns.distplot(new_data['out_prncp'])


# In[31]:


sns.distplot(new_data['out_prncp_inv'])


# In[32]:


sns.distplot(new_data['total_pymnt'])


# In[33]:


sns.distplot(new_data['total_pymnt_inv'])


# In[34]:


sns.distplot(new_data['total_rec_prncp'])


# In[35]:


sns.distplot(new_data['recoveries'])


# In[36]:


sns.distplot(new_data['collection_recovery_fee'])


# In[37]:


sns.distplot(new_data['last_pymnt_amnt'])


# In[38]:


new_data.columns


# # Removing Unwanted columns

# In[39]:


un_cols=['id','member_id','emp_title','grade','inq_last_6mths','sub_grade','zip_code','addr_state','purpose','earliest_cr_line','last_pymnt_d','last_credit_pull_d']
new_data= new_data.drop(columns=un_cols, axis=1)


# # To change issue_d  datatype  (object to datetime)

# In[40]:


new_data['issue_d']= pd.to_datetime(new_data.issue_d)
new_data_sort= new_data.sort_values(["issue_d"], ascending= True)
new_data_sort.head(2)


# In[41]:


new_data_sort = new_data_sort.reset_index(drop=True)
new_data_sort.insert(0, 'Sl.No', new_data_sort.index)
new_data_sort.set_index('Sl.No', inplace = True)


# # Spliting datatable by issue_d into training and test as given in problem statement

# In[42]:


# by using this data we need to build model
new_data_training=new_data_sort[(new_data_sort['issue_d']>='Jan-2007')& (new_data_sort['issue_d']<='May-2015')]


# In[43]:


new_data_training.shape


# In[44]:


new_data_test=new_data_sort[(new_data_sort['issue_d']>'May-2015')]
new_data_test = new_data_test.reset_index(drop=True)
new_data_test.insert(0, 'Sl.No', new_data_test.index)
new_data_test.set_index('Sl.No', inplace = True)


# In[45]:


new_data_test.shape


# In[46]:


cols_dt=['issue_d']
new_data_training= new_data_training.drop(columns=cols_dt, axis=1)
new_data_test= new_data_test.drop(columns=cols_dt, axis=1)


# # Label Encoding techinque  - to change character attributes to intiger values to built model
# 

# In[47]:


from sklearn.preprocessing import LabelEncoder
cols=['term','emp_length','home_ownership','verification_status','pymnt_plan','title','initial_list_status','application_type']
le= LabelEncoder()
for col in cols:
    new_data_training[col]=le.fit_transform(new_data_training[col])


# In[48]:


new_data_training.head(2)


# # We used Baruta Algorithm to find the significant and non-significant variables, and we will remove the non-significant variables before building a model

# In[49]:


x_br= new_data_training.drop(columns=['default_ind'], axis=1)
y_br= new_data_training['default_ind']


# In[50]:


print(x_br.shape, y_br.shape)


# In[51]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[52]:


select = SelectFromModel(RandomForestClassifier(n_estimators=50))
x_br = new_data_training.iloc[:,0:37]
y_br = new_data_training.iloc[:,-1]


# In[53]:


select.fit(x_br,y_br)


# In[54]:


select.get_support()


# In[55]:


list1 = list(select.get_support())
list1


# In[56]:


new_data_feat = pd.DataFrame({"Feature_Name": x_br.columns, "Importance": list1})
new_data_feat.sort_values(["Importance"], ascending = False)


# In[57]:


cols_uimp = ['collections_12_mths_ex_med','total_rev_hi_lim','term','int_rate','emp_length','home_ownership','annual_inc','verification_status','pymnt_plan','title','dti','delinq_2yrs','open_acc','pub_rec','revol_bal','revol_util','initial_list_status','total_acc','tot_cur_bal','tot_coll_amt','total_rec_int','total_rec_late_fee','acc_now_delinq','application_type','policy_code']
new_data1 = new_data_training.drop(columns= cols_uimp, axis=1)


# In[58]:


new_data1.head(2)


# # Train_Test_split

# In[59]:


new_data1.default_ind.value_counts()


# In[60]:


Yes =new_data1[new_data1["default_ind"]==1]
No =new_data1[new_data1["default_ind"]==0]


# In[61]:


print(Yes.shape, No.shape)


# In[62]:


X1= new_data1.drop(columns=['default_ind'], axis=1)
y1= new_data1['default_ind']


# # Using MinMaxScaler we are standardizing data in independent variables

# In[63]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X1= scaler.fit_transform(X1)


# In[64]:


X1


# In[65]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X1,y1, test_size=0.20, random_state=42)


# In[66]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Logistic Regression

# In[67]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X_train,y_train)


# In[68]:


y_pred_lg = log_model.predict(X_test)


# In[69]:


from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred_lg))


# In[70]:


print(classification_report(y_test,y_pred_lg))


# In[71]:


pd.crosstab(y_test,y_pred_lg)


# In[72]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
y_pred_log_roc = log_model.predict_proba(X_train)
print("logistic regression train roc-auc:{}".format (roc_auc_score(y_train,y_pred_log_roc[:,1])))
y_pred_log_roc = log_model.predict_proba(X_test)
print("logistic regression test roc-auc:{}".format (roc_auc_score(y_test,y_pred_log_roc[:,1])))


# # Random Forest

# In[73]:


class_weight= dict({0:1,1:100})


# In[74]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(class_weight=class_weight)


# In[75]:


rf_classifier.fit(X_train,y_train)


# In[76]:


y_pred_rf=rf_classifier.predict(X_test)


# In[77]:


from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred_rf))


# In[78]:


pd.crosstab(y_test,y_pred_rf)


# In[79]:


print(classification_report(y_test,y_pred_rf))


# In[80]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
y_pred_rf_roc = rf_classifier.predict_proba(X_train)
print("Random forest train roc-auc:{}".format (roc_auc_score(y_train,y_pred_rf_roc[:,1])))
y_pred_rf_roc = rf_classifier.predict_proba(X_test)
print("Random forest test roc-auc:{}".format (roc_auc_score(y_test,y_pred_rf_roc[:,1])))


# # Since it's a imbalanced dataset we use sampling techniques to make it as a balanced dataset
# # Under Sampling

# In[81]:


from imblearn.under_sampling import NearMiss


# In[82]:


nm = NearMiss(0.8)
X_train_ns,y_train_ns = nm.fit_sample(X_train,y_train)


# In[83]:


from collections import Counter
print ("Before sampling:",Counter(y_train))
print ("After sampling:" ,Counter(y_train_ns))


# In[84]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier_ns = RandomForestClassifier()
rf_classifier_ns.fit(X_train_ns,y_train_ns)


# In[85]:


y_pred_ns = rf_classifier_ns.predict(X_test)


# In[86]:


print(accuracy_score(y_test,y_pred_ns))


# In[87]:


pd.crosstab(y_test,y_pred_ns)


# In[88]:


print(classification_report(y_test,y_pred_ns))


# In[89]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
y_pred_ns_roc = rf_classifier_ns.predict_proba(X_train_ns)
print("Random forest-under_sample train roc-auc:{}".format (roc_auc_score(y_train_ns,y_pred_ns_roc[:,1])))
y_pred_ns_roc = rf_classifier_ns.predict_proba(X_test)
print("Random forest- under_sample test roc-auc:{}".format (roc_auc_score(y_test,y_pred_ns_roc[:,1])))


# # SMOTETomek technique

# In[90]:


pip install imblearn


# In[91]:


from imblearn.combine import SMOTETomek
smote = SMOTETomek(0.75)


# In[92]:


X_train_sm,y_train_sm = smote.fit_sample(X_train,y_train)


# In[93]:


from collections import Counter
print ("Before sampling:",Counter(y_train))
print ("After sampling:" ,Counter(y_train_sm))


# In[94]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier_sm = RandomForestClassifier()
rf_classifier_sm.fit(X_train_sm,y_train_sm)


# In[95]:


y_pred_sm = rf_classifier_sm.predict(X_test)


# In[96]:


print(accuracy_score(y_test,y_pred_sm))


# In[97]:


pd.crosstab(y_test,y_pred_sm)


# In[98]:


print(classification_report(y_test,y_pred_sm))


# In[99]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
y_pred_sm_roc = rf_classifier_sm.predict_proba(X_train_sm)
print("Random forest-SMOTE train roc-auc:{}".format (roc_auc_score(y_train_sm,y_pred_sm_roc[:,1])))
y_pred_sm_roc = rf_classifier_sm.predict_proba(X_test)
print("Random forest- SMOTE test roc-auc:{}".format (roc_auc_score(y_test,y_pred_sm_roc[:,1])))


# # XGBoost 

# In[100]:


from xgboost import XGBClassifier
classifier_xgb = XGBClassifier()
classifier_xgb.fit(X_train,y_train)


# In[101]:


y_pred_xgb= classifier_xgb.predict(X_test)


# In[102]:


from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred_xgb))
print(classification_report(y_test,y_pred_xgb))


# In[103]:


pd.crosstab(y_test,y_pred_xgb)


# In[104]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
y_pred_xgb_roc = classifier_xgb.predict_proba(X_train)
print("XGBoost train roc-auc:{}".format (roc_auc_score(y_train,y_pred_xgb_roc[:,1])))
y_pred_xgb_roc = classifier_xgb.predict_proba(X_test)
print("XGBoost SMOTE test roc-auc:{}".format (roc_auc_score(y_test,y_pred_xgb_roc[:,1])))


# # We can select the best thresold Value for maximum accuracy

# In[105]:


predict = []
for model in [log_model,rf_classifier,rf_classifier_ns,rf_classifier_sm,classifier_xgb]:
    predict.append(pd.Series(model.predict_proba(X_test)[:,1]))
final_prediction= pd.concat(predict,axis=1).mean(axis=1)
print("Ensemble test roc-auc:{}".format(roc_auc_score(y_test,final_prediction)))


# In[106]:


pd.concat(predict, axis=1)


# In[107]:


fpr, tpr , thresholds = roc_curve(y_test, final_prediction)
thresholds


# In[108]:


from sklearn.metrics import accuracy_score
accuracy_score_th= []
for thres in thresholds:
    y_pred = np.where(final_prediction>thres,1,0)
    accuracy_score_th.append(accuracy_score(y_test,y_pred, normalize =True))
accuracy_score_th= pd.concat([pd.Series(thresholds),pd.Series(accuracy_score_th)], axis=1)
accuracy_score_th.columns= ['thresholds','accuracy']
accuracy_score_th.sort_values(by='accuracy', ascending = False, inplace = True)
accuracy_score_th.head()


# In[110]:


def plot_roc_curve(fpr,tpr):
    plt.plot(fpr,tpr, color= 'red', label = 'ROC')
    plt.plot([0,1],[0,1] ,color= 'darkblue', linestyle= '--')
    plt.xlabel('False Postive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic(ROC) Curve ')
    plt.legend()
    plt.show()    


# In[111]:


plot_roc_curve(fpr,tpr)


# In[113]:


import pickle


# In[114]:


with open('rf_classifier_pickle','wb') as f:
    pickle.dump(rf_classifier,f)

