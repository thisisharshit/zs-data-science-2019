
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import std
from numpy import mean


# In[20]:


data=pd.read_csv("Downloads/ZS data science 2019/data.csv")


# In[21]:


data.head(10)


# In[22]:


(data.columns)


# In[23]:


del data['Unnamed: 0']


# In[24]:


(data.columns)


# In[25]:


data.describe()


# In[26]:


data.info()


# In[27]:


data.apply(lambda x: sum(x.isnull()),axis=0)


# In[28]:


sum(data['is_goal'].isna())


# In[29]:


data.shape


# In[30]:


test=pd.DataFrame(data[data['is_goal'].isnull()])


# In[31]:


test


# In[32]:


data.drop(data[data['is_goal'].isnull()].index,inplace=True)


# In[33]:


data.shape


# In[34]:


print(data)


# In[35]:


test.reset_index(drop=True)


# In[36]:


train=data.reset_index(drop=True)


# In[37]:


train.head()


# In[38]:


train.shape


# In[39]:


test.shape


# In[40]:


dataset=pd.concat([train,test],ignore_index=True)


# In[41]:


dataset.shape


# In[42]:


dataset


# In[43]:


dataset['power_of_shot'].value_counts()


# In[44]:


dataset['knockout_match'].value_counts()


# In[45]:


dataset['game_season'].value_counts()


# In[46]:


dataset['type_of_shot'].value_counts()
len(set(dataset['type_of_shot']))


# In[47]:


dataset['type_of_combined_shot'].value_counts()


# In[48]:


dataset['area_of_shot'].value_counts()


# In[49]:


dataset['shot_basics'].value_counts()


# In[50]:


dataset['team_name'].value_counts()


# In[51]:


dataset['date_of_game'].value_counts()
#len(set(dataset['date_of_game']))


# In[52]:


dataset['team_id'].value_counts()


# In[53]:


dataset['home/away'].value_counts()
len(set(dataset['home/away']))


# In[54]:


len(set(dataset['team_id']))


# In[55]:


dataset['range_of_shot'].value_counts()


# In[56]:


dataset['shot_id_number'].value_counts()
len(set(dataset['shot_id_number']))


# In[57]:


dataset['match_id'].value_counts()
len(set(dataset['match_id']))


# In[58]:


dataset['remaining_min'].value_counts()
#len(set(dataset['remaining_min']))


# In[59]:


dataset['match_event_id'].value_counts()


# In[60]:


newdataset=dataset[['location_x','location_y','remaining_min','power_of_shot','knockout_match','game_season','remaining_sec','distance_of_shot','area_of_shot','shot_basics','range_of_shot','date_of_game','home/away','lat/lng','type_of_shot','type_of_combined_shot','remaining_min.1','power_of_shot.1','knockout_match.1','remaining_sec.1','distance_of_shot.1','is_goal']]


# In[61]:


newdataset


# In[62]:


sum(newdataset['location_x'].isna())


# In[63]:


newdataset.describe()


# In[64]:


train.describe()


# In[65]:


print(newdataset['location_y'].mean())
print(newdataset['location_y'].median())
print(newdataset['location_y'].mode()[0])


# In[66]:


newdataset['location_x'].fillna(newdataset['location_x'].mean(),inplace=True)


# In[67]:


print(newdataset['location_y'].mean())
print(newdataset['location_y'].median())
print(newdataset['location_y'].mode()[0])


# In[68]:


variable=['location_x','location_y','remaining_min','remaining_sec','distance_of_shot','remaining_min.1','power_of_shot.1','knockout_match.1','remaining_sec.1','distance_of_shot.1']


# In[69]:


for i in variable:
    print(i)
    print(train[i].mean(),test[i].mean(),newdataset[i].mean())
    print(train[i].median(),test[i].median(),newdataset[i].median())
    print(train[i].mode()[0],test[i].mode()[0],newdataset[i].mode()[0])
    print(std(train[i]),std(test[i]),std(newdataset[i]))
    print("\n")


# In[70]:


pd.crosstab(newdataset['is_goal'],newdataset['remaining_sec'])


# In[71]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,12))
A=train['remaining_min']
Anan=A[~np.isnan(A)]
sns.distplot(Anan)


# In[72]:


train


# In[73]:


test


# In[74]:


train=train.drop(['match_event_id','match_id','team_id','shot_id_number','team_name'],axis=1)


# In[75]:


test=test.drop(['match_event_id','match_id','team_id','shot_id_number','team_name','is_goal'],axis=1)


# In[76]:


test=test.reset_index(drop=True)


# In[77]:


test


# In[78]:


newdataset


# In[79]:


newdataset.apply(lambda x:sum(x.isnull()),axis=0)


# In[80]:


discrete=['power_of_shot','knockout_match','game_season','area_of_shot','shot_basics','range_of_shot','date_of_game','home/away','power_of_shot.1','type_of_shot','type_of_combined_shot']
for i in discrete:
    newdataset[i].fillna(newdataset[i].mode()[0],inplace=True)


# In[81]:


newdataset.apply(lambda x:sum(x.isnull()),axis=0)


# In[82]:


for i in variable:
    newdataset[i].fillna(newdataset[i].mean(),inplace=True)
    


# In[83]:


newdataset['remaining_min.1'].fillna(newdataset['remaining_min.1'].mode()[0],inplace=True)
newdataset['remaining_min'].fillna(newdataset['remaining_min'].mode()[0],inplace=True)
newdataset['remaining_sec.1'].fillna(newdataset['remaining_sec.1'].median(),inplace=True)
newdataset['remaining_sec'].fillna(newdataset['remaining_sec'].median(),inplace=True)
newdataset['knockout_match.1'].fillna(newdataset['knockout_match.1'].median(),inplace=True)
newdataset['distance_of_shot.1'].fillna(newdataset['distance_of_shot.1'].median(),inplace=True)
newdataset['distance_of_shot'].fillna(newdataset['distance_of_shot'].median(),inplace=True)
newdataset['location_y'].fillna(newdataset['location_y'].mean(),inplace=True)


# In[84]:


newdataset


# In[85]:


newdataset[['lat','lng']]=newdataset['lat/lng'].str.split(expand=True,)


# In[86]:


del newdataset['lat/lng']


# In[87]:


newdataset


# In[88]:


newdataset=pd.DataFrame(newdataset)


# In[89]:


newdataset['lng']=newdataset['lng'].astype(str)
newdataset['lat']=newdataset['lat'].astype(str)


# In[90]:


newdataset['lng']=newdataset['lng'].astype(float)
#newdataset['lat']=newdataset['lat'].astype(float)


# In[91]:


newdataset["lat"] = newdataset["lat"].str.replace(",","").astype(float)


# In[92]:


newdataset.info()


# In[93]:


print(newdataset['lng'].mean(),newdataset['lng'].median(),newdataset['lng'].mode()[0])
print(newdataset['lat'].mean(),newdataset['lat'].median(),newdataset['lat'].mode()[0])


# In[94]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,12))
A=newdataset['lat']
Anan=A[~np.isnan(A)]
sns.distplot(Anan)


# In[95]:


newdataset['lng'].fillna(newdataset['lng'].median(),inplace=True)
newdataset['lat'].fillna(newdataset['lat'].median(),inplace=True)


# In[96]:


newdataset.info()


# In[97]:


numeric_features = newdataset.select_dtypes(include=['float64','int64'])
numeric_features.dtypes


# In[98]:


corr=numeric_features.corr()
corr


# In[99]:


ax=plt.subplots(figsize=(12,9))
sns.heatmap(corr)


# In[100]:


from sklearn.preprocessing import LabelEncoder
discrete=['game_season','area_of_shot','shot_basics','range_of_shot','type_of_shot','type_of_combined_shot']
le = LabelEncoder()
for i in discrete:
    newdataset[i] = le.fit_transform(newdataset[i])


# In[101]:


newdataset


# In[102]:


#Dummy Variables:
newdata = pd.get_dummies(newdataset, columns =['game_season','area_of_shot','shot_basics','range_of_shot','type_of_shot','type_of_combined_shot'])
newdata.dtypes


# In[103]:


newdata.shape


# In[104]:


newdata['date_of_game']=pd.to_datetime(newdata['date_of_game'])


# In[105]:


newdata['date_of_game'].dt.year.value_counts()


# In[106]:


newdata['date_of_game'].dt.month.value_counts()


# In[107]:


newdata['year']=newdata['date_of_game'].dt.year
newdata['month']=newdata['date_of_game'].dt.month


# In[108]:


newdata.shape


# In[109]:


del newdata['date_of_game']


# In[110]:


newdata=pd.get_dummies(newdata,columns=['year','month'])


# In[111]:


newdata.dtypes


# In[112]:


newdata.to_csv("Downloads/ZS data science 2019/newdata.csv",index=False)


# In[113]:


numeric_features = newdata.select_dtypes(include=['float64','int64','uint8'])
numeric_features.dtypes


# In[114]:


newdata['home/away'] = le.fit_transform(newdata['home/away'])


# In[115]:


newdata=pd.get_dummies(newdata,columns=['home/away'])


# In[116]:


train=newdata.iloc[0:24429,:]
test=newdata.iloc[24429:,:]
del test['is_goal']
test=test.reset_index(drop=True)
y=train[['is_goal']]
del train['is_goal']


# In[117]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.20, random_state = 0)


# In[118]:


print(train.shape,test.shape)


# In[119]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[120]:


X_train=X_train.reset_index(drop=True)
X_test=X_test.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)


# In[122]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train)
X_test2 = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train2, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test2)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[123]:


(2050+928)/(2050+928+1330+578)


# In[124]:


y_pred


# In[125]:


y_pred=classifier.predict_proba(X_test)


# In[126]:


y_pred=pd.DataFrame(y_pred)


# In[127]:


y_pred


# In[128]:


y_pred=y_pred[1]


# In[129]:


y_pred
#y_test=y_test.reset_index(drop=True)


# In[130]:


y_pred=pd.DataFrame(y_pred)
mae=abs(y_pred[1]-y_test['is_goal'])


# In[131]:


mae[3]


# In[132]:


sum=0
for i in range(4886):
    sum+=mae[i]
mae=sum/4886


# In[133]:


mae


# In[134]:


predictions=classifier.predict_proba(test)


# In[135]:


predictions


# In[136]:


predictions=pd.DataFrame(predictions)


# In[137]:


predictions=predictions[1]


# In[139]:


predictions=pd.DataFrame(predictions)


# In[140]:


predictions.to_csv("Downloads/ZS data science 2019/submission.csv",index=False)


# In[141]:


# Importing the Keras libraries and packages
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


# In[145]:


X_train.shape[1]


# In[146]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 120, kernel_initializer='normal', activation = 'relu', input_dim = X_train.shape[1]))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 100, kernel_initializer='normal', activation = 'relu'))
classifier.add(Dense(output_dim = 80, kernel_initializer='normal', activation = 'relu'))
classifier.add(Dense(output_dim = 60, kernel_initializer='normal', activation = 'relu'))
classifier.add(Dense(output_dim = 40, kernel_initializer='normal', activation = 'relu'))
classifier.add(Dense(output_dim = 20, kernel_initializer='normal', activation = 'relu'))
classifier.add(Dense(output_dim = 10, kernel_initializer='normal', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer='normal', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mean_absolute_error'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 100, nb_epoch = 500)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[147]:


y_pred


# In[148]:


predictions=classifier.predict(test)


# In[149]:


predictions=pd.DataFrame(predictions)
predictions.to_csv("Downloads/ZS data science 2019/submission2.csv",index=False)

