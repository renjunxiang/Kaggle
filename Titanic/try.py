import pandas as pd
import numpy as np
import os
from sklearn_supervised import sklearn_supervised
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# PassengerId => 乘客ID（无意义）
# Pclass => 乘客等级(1/2/3等舱位)  （分类）
# Name => 乘客姓名  （分类）（无意义）
# Sex => 性别  （分类）
# Age => 年龄  （连续）
# SibSp => 堂兄弟/妹个数 （连续）
# Parch => 父母与小孩个数 （连续）
# Ticket => 船票信息   （分类）（无意义）
# Fare => 票价 （连续）
# Cabin => 客舱  （分类）
# Embarked => 登船港口  （分类）

# 剔除姓名、船票信息
train_x = train_data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
text_x = test_data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]
# 先合并训练和测试，用于转亚变量，避免测试集类别缺失
train_test_x = pd.concat([train_x, text_x], axis=0)
# 替换缺失值
replace=train_test_x['Age'].mean()

train_test_x['Age'] = np.where(train_test_x['Age'].isna(), 999, train_test_x['Age'])
train_test_x['Fare'] = np.where(train_test_x['Fare'].isna(), 999, train_test_x['Fare'])
# 舱位转文本
train_test_x['Pclass'] = train_x['Pclass'].astype(str)
# 转亚变量
train_test_x_dummies = pd.get_dummies(train_test_x)
# 拆分训练集、测试集
train_x_dummies = train_test_x_dummies.iloc[0:len(train_x), :]
text_x_dummies = train_test_x_dummies.iloc[len(train_x):, :]
# 训练集标签
train_y = train_data.iloc[:, 1:2]

# train_test_x_dummies.to_csv('./data/try.csv',index=False)

# train_data_dummies=pd.concat([train_y,train_x_dummies])
# train_data_dummies.corr(method='pearson')

for model_name in ['SVM', 'Logistic', 'KNN']:
    model = sklearn_supervised(data=train_x_dummies, label=train_y, model_name=model_name)
    print('score:%f' % (np.sum(model.predict(train_x_dummies) == train_y.iloc[:, 0]) / len(train_y)))
    test_y = model.predict(text_x_dummies)
    result = test_data.iloc[:, 0:1]
    result['Survived'] = test_y
    result.to_csv('./result/%s.csv'%model_name,index=False)

# model2 = sklearn_supervised(data=train_x_dummies, label=train_y, model_name='KNN')
# print('score:%f' %(np.sum(model2.predict(train_x_dummies) == train_y.iloc[:, 0]) / len(train_y)))
#
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X=train_x_dummies,y=train_y)
print('score:%f' % (np.sum(model.predict(train_x_dummies) == train_y.iloc[:, 0]) / len(train_y)))
test_y = model.predict(text_x_dummies)
result = test_data.iloc[:, 0:1]
result['Survived'] = test_y
result.to_csv('./result/%s.csv' % 'KNN', index=False)
