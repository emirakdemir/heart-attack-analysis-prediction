#libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve


df = pd.read_csv("heart.csv")

df.head(11)
df.describe().T
df.info()
df.isnull().sum()


#unique value analysis
for i in list(df.columns):
    print(i , ": " , df[i].value_counts().shape[0])
"""
age:       41
sex:        2
cp:         4
trtbps:    49
chol:     152
fbs:        2
restecg:    3
thalachh:  91
exng:       2
oldpeak:   40
slp:        3
caa:        5
thall:      4
output:     2
"""    
    
#categorical feature analysis
categorical_features =["sex","cp","fbs","restecg","exng","slp","caa","thall","output"]
numerical_features = ["age","trtbps","chol","thalachh","oldpeak","output"]

df_categorical = df[categorical_features]
df_numerical = df[numerical_features]

for i in df_categorical:
    plt.figure()
    sns.countplot(x = i, hue="output",data = df_categorical)
    plt.title(i)
    plt.show()
    
#numerical feture analysis
sns.pairplot(data = df_numerical, 
             hue = "output", 
             diag_kind="kde")
plt.show()


#standardization
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df[numerical_features[:-1]])
scaled_array


#box plot analysis
df_dummy = pd.DataFrame(data = scaled_array,columns = numerical_features[:-1])
df_dummy = pd.concat([df_dummy,df.loc[:,"output"]],axis=1)
df_dummy.head(6)
"""
        age    trtbps      chol  thalachh   oldpeak  output  output
0  0.952197  0.763956 -0.256334  0.015443  1.087338       1       1
1 -1.915313 -0.092738  0.072199  1.633471  2.122573       1       1
2 -1.474158 -0.092738 -0.816773  0.977514  0.310912       1       1
3  0.180175 -0.663867 -0.198357  1.239897 -0.206705       1       1
4  0.290464 -0.663867  2.082050  0.583939 -0.379244       1       1
5  0.290464  0.478391 -1.048678 -0.072018 -0.551783       1       1
"""
data_melted = pd.melt(df_dummy,id_vars="output", var_name="features", value_name="value")
data_melted.head(5)

plt.figure()
sns.boxplot(x="features",y="value",hue="output", data=data_melted)
plt.show()


#Swarm plot analysis
plt.figure()
sns.swarmplot(x="features",y="value",hue="output", data=data_melted)
plt.show()


#Cat plot analysis
plt.figure()
sns.catplot(x="fbs",y="age",hue="output", col="sex", kind="swarm", data=df)
plt.show()


#Correlation analysis
plt.figure(figsize=(20,8))
sns.heatmap(data=df.corr(),annot=True, fmt=".2f",linewidths=.7)
plt.show()


#Outlier Detection
numerical_features = ["age","trtbps","chol","thalachh","oldpeak"]
df_numerical = df.loc[:,numerical_features]
df_numerical.head(6)
"""
   age  trtbps  chol  thalachh  oldpeak
0   63     145   233       150      2.3
1   37     130   250       187      3.5
2   41     130   204       172      1.4
3   56     120   236       178      0.8
4   57     120   354       163      0.6
5   57     140   192       148      0.4
"""

for i in numerical_features:
    #IQR
    Q1 = np.percentile(df.loc[:,i],25)
    Q3 = np.percentile(df.loc[:,i],75)
    IQR = Q3-Q1
    print("Old shape --> ",df.loc[:,i].shape)
    #Upper bounds
    upper = np.where(df.loc[:,i]>=Q1 + 2.5 * IQR)
    #Lower bounds
    lower = np.where(df.loc[:,i]<=Q3 - 2.5 * IQR)
    
    print("upper--> {} -- lower-->{}".format(upper,lower))
    
    try:
        df.drop(upper[0], inplace=True)
    except: print("KeyError: {} not found in axis".format(upper[0]))
    try:
        df.drop(lower[0], inplace=True)
    except: print("KeyError: {} not found in axis".format(lower[0]))

    print("New Shape --> ", df.shape)
    
    
#Modelling
df_backup = df.copy()

#Encoding categorical columns
for i in list(df.columns):
    print(i, ":",df[i].value_counts().shape[0])
"""
age:       41
sex:        2
cp:         4
trtbps:    42
chol:     149
fbs:        2
restecg:    3
thalachh:  89
exng:       2
oldpeak:   40
slp:        3
caa:        5
thall:      4
output:     2
"""

categorical_features =["cp","restecg","slp","caa","thall"]
df = pd.get_dummies(data = df,columns=categorical_features)
df.head(6)
"""
   age  sex  trtbps  chol  fbs  ...  caa_4  thall_0  thall_1  thall_2  thall_3
0   63    1     145   233    1  ...      0        0        1        0        0
1   37    1     130   250    0  ...      0        0        0        1        0
2   41    0     130   204    0  ...      0        0        0        1        0
3   56    1     120   236    0  ...      0        0        0        1        0
4   57    0     120   354    0  ...      0        0        0        1        0
5   57    1     140   192    0  ...      0        0        1        0        0
"""
df.columns

x=df.drop(["output"],axis=1)
y=df[["output"]]

#scaling
x[numerical_features] = scaler.fit_transform(x[numerical_features])
x.head(6)
"""
        age  sex    trtbps      chol  ...  thall_0  thall_1  thall_2  thall_3
0  0.957206    1  1.049817 -0.231231  ...        0        1        0        0
1 -1.871129    1  0.027128  0.095197  ...        0        0        1        0
2 -1.436001    0  0.027128 -0.788078  ...        0        0        1        0
3  0.195731    1 -0.654665 -0.173626  ...        0        0        1        0
4  0.304513    0 -0.654665  2.092165  ...        0        0        1        0
5  0.304513    1  0.708921 -1.018497  ...        0        1        0        0
"""

#train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=100,shuffle=True,test_size=0.25)

#logistic regression
lg = LogisticRegression()
lg.fit(x_train,y_train)

y_pred_proba = lg.predict_proba(x_test)
y_pred=np.argmax(y_pred_proba,axis=1)
"Test Accuracy : ", accuracy_score(y_pred,y_test)                   
"""
0.7746478873239436
"""

#logistic regression hyperparamater tuning
penalty = ["l1","l2"]
parameters = {"penalty":penalty}

lg_gridsearch = GridSearchCV(lg,parameters)
lg_gridsearch.fit(x_train,y_train)
"""
GridSearchCV(estimator=LogisticRegression(),
             param_grid={'penalty': ['l1', 'l2']})
"""

"Best paramater --> " , lg_gridsearch.best_params_
"""
'Best paramater --> ', {'penalty': 'l2'}
"""

y_predict = lg_gridsearch.predict(x_test)

"Test Accuracy : ", accuracy_score(y_predict,y_test)
"""
'Test Accuracy : ', 0.774647887323943
"""