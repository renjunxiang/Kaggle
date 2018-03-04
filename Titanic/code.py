import pandas as pd
import numpy as np
import os
from sklearn_supervised import sklearn_supervised
from sklearn.model_selection import train_test_split
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
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

# Cabin变量过多分类和缺失值，按照有无缺失处理成二值变量
train_data['Cabin'] = np.where(train_data['Cabin'].isna(), 0, 1)
test_data['Cabin'] = np.where(test_data['Cabin'].isna(), 0, 1)

train_data_1 = train_data.iloc[:, :]

for i in train_data_1.columns:
    try:
        # 连续变量用中位数替换缺失值
        train_data_1[i][train_data_1[i].isna()] = train_data_1[i].median()
    except:
        # 分类变量用-999替换
        train_data_1[i][train_data_1[i].isna()] = -999

# 判断对存活有影响的离散变量,方差分析
discrete = []
for i in ['Pclass', 'Sex', 'SibSp', 'Cabin', 'Embarked']:
    stat_text = 'Survived~%s' % i
    discrete.append(anova_lm(ols(stat_text, data=train_data_1).fit()).iloc[0:1, :])
discrete = pd.concat(discrete, axis=0)
var_discrete = discrete['PR(>F)'][discrete['PR(>F)'] < 0.05].index

# 判断对存活有影响的连续变量,相关性分析
continuous = train_data_1.loc[:, ['Pclass', 'Age', 'Parch', 'Fare', 'Cabin', 'Survived']].corr()['Survived']

# 得到有意义的变量为'Pclass', 'Sex', 'Embarked','Fare'
train_x = train_data.loc[:, ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Fare']]
text_x = test_data.loc[:, ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Fare']]

# 先合并训练和测试，用于转亚变量，避免测试集类别缺失
train_test_x = pd.concat([train_x, text_x], axis=0)

# 连续变量用中位数替换缺失值，分类变量用-999替换
for i in train_test_x.columns:
    try:
        # 连续变量用中位数替换缺失值
        train_test_x[i][train_test_x[i].isna()] = train_test_x[i].median()
    except:
        # 分类变量用-999替换
        train_test_x[i][train_test_x[i].isna()] = -999

# 舱位转文本
train_test_x['Pclass'] = train_test_x['Pclass'].astype(str)

# 转亚变量
train_test_x_dummies = pd.get_dummies(train_test_x)

# 拆分训练集、测试集
train_x_dummies = train_test_x_dummies.iloc[0:len(train_x), :]
text_x_dummies = train_test_x_dummies.iloc[len(train_x):, :]

# 训练集标签
train_y = train_data.iloc[:, 1:2]

# 拆分训练集测试集
train_train_x, train_test_x, train_train_y, train_test_y = train_test_split(train_x_dummies, train_y, test_size=0.3)
result_all_test = test_data.iloc[:, 0:1]

# 单个模型的准确率都只在0.75左右，采用5个模型投票的方式可能会好一点(达到0.77)
result_all_train_train = pd.DataFrame()
result_all_train_test = pd.DataFrame()
f = open('./log.txt', mode='w')
for model_name in ['SVM', 'Logistic', 'KNN', 'DecisionTree', 'Naivebayes']:
    model = sklearn_supervised(data=train_train_x, label=train_train_y, model_name=model_name)
    print('%s train score: %f' % (
        model_name, np.sum(model.predict(train_train_x) == train_train_y.iloc[:, 0]) / len(train_train_y)))
    f.write('%s train score: %f\n\n' % (
        model_name, np.sum(model.predict(train_train_x) == train_train_y.iloc[:, 0]) / len(train_train_y)))
    print('%s test score: %f' % (
        model_name, np.sum(model.predict(train_test_x) == train_test_y.iloc[:, 0]) / len(train_test_y)))
    f.write('%s test score: %f\n\n' % (
        model_name, np.sum(model.predict(train_test_x) == train_test_y.iloc[:, 0]) / len(train_test_y)))
    result_all_train_train[model_name] = model.predict(train_train_x)
    result_all_train_test[model_name] = model.predict(train_test_x)
    test_y = model.predict(text_x_dummies)
    result = test_data.iloc[:, 0:1]
    result_all_test[model_name] = test_y
    result['Survived'] = test_y
    result.to_csv('./result/%s.csv' % model_name, index=False)

score_train_train = result_all_train_train.agg(lambda x: sum(list(x)), axis=1)
score_train_train[score_train_train < 3] = 0
score_train_train[score_train_train >= 3] = 1
print('score_train_train:%f' % (np.sum(np.array(score_train_train) == train_train_y.iloc[:, 0]) / len(train_train_y)))
f.write(
    'score_train_train:%f\n\n' % (np.sum(np.array(score_train_train) == train_train_y.iloc[:, 0]) / len(train_train_y)))

score_train_test = result_all_train_test.agg(lambda x: sum(list(x)), axis=1)
score_train_test[score_train_test < 3] = 0
score_train_test[score_train_test >= 3] = 1
print('score_train_test:%f' % (np.sum(np.array(score_train_test) == train_test_y.iloc[:, 0]) / len(train_test_y)))
f.write('score_train_test:%f\n\n' % (np.sum(np.array(score_train_test) == train_test_y.iloc[:, 0]) / len(train_test_y)))
f.close()
# 过半数模型投票1才判定为1
score = result_all_test.agg(lambda x: sum(list(x)[1:]), axis=1)
score[score < 3] = 0
score[score >= 3] = 1
result_5_models = test_data.iloc[:, 0:1]
result_5_models['Survived'] = score
result_5_models.to_csv('./result/result_5_models.csv', index=False)

