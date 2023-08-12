# ### Library Import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, learning_curve,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# import scipy.stats as ss
# import os


# ### Data Load


train_data = pd.read_csv('Datasets/aug_train.csv')
test_data = pd.read_csv('Datasets/aug_test.csv')


# train_data.head()
print(train_data.head())


# train_data.info()
print(train_data.info())


# train_data.describe()
print(train_data.describe())


# ### EDA - Exploratory Data Analysis
#


# Number of features contains missing values


# train_data.isnull().sum()
print(train_data.isnull().sum())


# Number of rows contains at least one missing value


x = train_data.dropna()
total = train_data.shape[0]
missing = x.shape[0]
print('Number of rows:', missing)
print('Percentage of rows that contains missing values:', (missing) * 100 / total)


# The percentage of rows that contains missing values is very high so we
# can not drop these rows. The next table will display the missing value
# in each column and their percentage.
#


null = pd.DataFrame({'No of null values': train_data.isnull().sum(
), 'Percentage of null values': train_data.isnull().sum() * 100 / train_data.shape[0]})
# null
print(null)


# Columns that contain missing values


# null[null['No of null values'] > 0]
print(null[null['No of null values'] > 0])


# enrollee_id does not have duplicate ids, so it does not carry any semantic load, except for the candidate index, which should have been used as
# the index of the dataset, but I will not do it now


# train_data['enrollee_id'].nunique() == train_data['enrollee_id'].count()
print(train_data['enrollee_id'].nunique() == train_data['enrollee_id'].count())


# We are not interested in enrollee_id so we will drop it


train_data.drop(['enrollee_id'], axis=1, inplace=True)
# train_data.shape
print(train_data.shape)


# Divide categorical columns and numerical columns


cat_col = []
num_col = []
for col in train_data.columns:
    if train_data[col].dtype == object:
        print('category coloumn : ', col)
        cat_col.append(col)
    else:
        print('numarical column : ', col)
        num_col.append(col)


# Check categorical Columns
# cat_col
print(cat_col)


# Check numerical Columns
# num_col
print(num_col)


# Check data in 'categorical columns'
# train_data[cat_col]
print(train_data[cat_col])


# We can see that our dataset contains 19158 entries and 14 columns. We have 4 numerical variables and 10 categorical variables
#


print(train_data.shape)
# print(train_data['city'].unique())
print(train_data['gender'].unique())
print(train_data['relevent_experience'].unique())
print(train_data['enrolled_university'].unique())
print(train_data['education_level'].unique())
print(train_data['major_discipline'].unique())
print(train_data['experience'].unique())
print(train_data['company_size'].unique())
print(train_data['company_type'].unique())
print(train_data['last_new_job'].unique())


# Distribution of Variables


# There is a strong imbalance of the target attribute 1:4

plt.figure(figsize=(15, 10))
sns.countplot(data=train_data, x='target')


count = train_data.target.value_counts()
print(count)
plt.figure(figsize=(6, 6))
sns.countplot(train_data.target, color=sns.color_palette()[0])
plt.title('distribution of target')
plt.xlabel('target values')
plt.ylabel('count')
for i in range(count.shape[0]):
    plt.text(i,
             count[i] + 500,
             str(round(100 * count[i] / train_data.target.count())) + '%',
             ha='center',
             va='top')


round(train_data['target'].value_counts(normalize=True) * 100)
print(round(train_data['target'].value_counts(normalize=True) * 100))


# target = 1 is more common in city_development_index < 0.65

plt.figure(figsize=(15, 10))
sns.histplot(
    data=train_data,
    x='city_development_index',
    hue='target',
    multiple="stack")


train_data[['city_development_index']].boxplot()


# Less than 200 candidates studied for more than 110 hours
# Due to the target imbalance, it can be assumed that the target attribute
# is distributed equally and training_hours does not have a strong
# influence

plt.figure(figsize=(15, 10))
sns.histplot(
    data=train_data,
    x='training_hours',
    hue='target',
    multiple="stack",
    kde=True)


train_data[['training_hours']].boxplot()


check_city = train_data['city'].value_counts().head(10)
# check_city
print(check_city)


# Most of the candidates are from 7 major cities

plt.figure(figsize=(15, 10))
sns.lineplot(data=check_city)


train_data['city'] = train_data['city'].apply(
    lambda x: int(x.replace('city_', '')))


plt.figure(figsize=(15, 10))
sns.histplot(x=train_data.city, hue=train_data.target, multiple="stack")


# Male objects are observed much more than the rest

plt.figure(figsize=(15, 10))
sns.countplot(data=train_data, x='gender', hue='target')


# train_data['relevent_experience'].value_counts()
print(train_data['relevent_experience'].value_counts())


# People with the relevent experience are more likely to be looking for a job

plt.figure(figsize=(15, 10))
sns.countplot(x=train_data.relevent_experience, hue=train_data.target)


change = train_data[train_data.target == 1].relevent_experience.value_counts()
no_change = train_data[train_data.target ==
                       0].relevent_experience.value_counts()
plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.pie(change, autopct='%1.2f%%', labels=change.index)
plt.legend()
plt.title('Change')
plt.subplot(1, 2, 2)
plt.pie(no_change, autopct='%1.2f%%', labels=no_change.index)
plt.legend()
plt.title('No Change')
plt.show()


# train_data['enrolled_university'].unique()
print(train_data['enrolled_university'].unique())


# Candidates who have completed full time course are more likely to look
# for a new job

plt.figure(figsize=(15, 10))
sns.countplot(data=train_data, x='enrolled_university', hue='target')


# Candidates who have no relevent experience but have completed full-time course are more likely to look for a job than candidates who have relevent
# experience and have completed full-time course

sns.catplot(
    data=train_data,
    x='enrolled_university',
    hue='relevent_experience',
    col='target',
    kind='count',
    height=5)


# train_data['education_level'].unique()
print(train_data['education_level'].unique())


# A natural situation, Graduate are most often in search of work

plt.figure(figsize=(15, 10))
sns.countplot(data=train_data, x='education_level', hue='target')


change = train_data[train_data.target == 1].education_level.value_counts()
no_change = train_data[train_data.target == 0].education_level.value_counts()
plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.pie(change, autopct='%1.2f%%', labels=change.index)
plt.legend()
plt.title('Change')
plt.subplot(1, 2, 2)
plt.pie(no_change, autopct='%1.2f%%', labels=no_change.index)
plt.legend()
plt.title('No Change')
plt.show()


# Most of the candidates in the sample with the STEM

# train_data['major_discipline'].value_counts()
print(train_data['major_discipline'].value_counts())


plt.figure(figsize=(15, 10))
sns.countplot(data=train_data, x='major_discipline', hue='target')


change = train_data[train_data.target == 1].major_discipline.value_counts()
no_change = train_data[train_data.target == 0].major_discipline.value_counts()
plt.figure(figsize=(20, 20))
plt.subplot(1, 2, 1)
plt.pie(change, autopct='%1.2f%%', labels=change.index)
plt.legend()
plt.title('Change')
plt.subplot(1, 2, 2)
plt.pie(no_change, autopct='%1.2f%%', labels=no_change.index)
plt.legend()
plt.title('No Change')
plt.show()


print(
    'unique values:',
    * train_data['experience'].unique(),
    '\n\n count unique values:',
    train_data['experience'].nunique())


# train_data['experience'].value_counts()
print(train_data['experience'].value_counts())


# Candidates who have worked for more than 20 years are most often found in the sample
# Candidates with experience = (24) are most often looking for a new job

plt.figure(figsize=(15, 10))
sns.countplot(data=train_data, x='experience', hue='target')


# At first I thought that having experience, we don't need relevant_experience, since they repeat each other, but no. Apparently, experience implies
# the GENERAL experience of the candidate, and not the experience in the
# field of search

plt.figure(figsize=(15, 10))
sns.countplot(data=train_data, x='experience', hue='relevent_experience')


# train_data['company_size'].value_counts()
print(train_data['company_size'].value_counts())


# The size of the company doesn't seem to have much effect on the target
# variable

plt.figure(figsize=(15, 10))
sns.countplot(data=train_data, x='company_size', hue='target')


# train_data['company_type'].value_counts()
print(train_data['company_type'].value_counts())


# The type of company doesn't seem to have much effect on the target variable

plt.figure(figsize=(15, 10))
sns.countplot(data=train_data, x='company_type', hue='target')


# train_data['last_new_job'].value_counts()
print(train_data['last_new_job'].value_counts())


# Most often, people are looking for a job with little or no work
# experience at all

plt.figure(figsize=(15, 10))
sns.countplot(data=train_data, x='last_new_job', hue='target')


# Transformaciones e inputaciones de NaN


# train_data.isnull().sum()
print(train_data.isnull().sum())


train_data = train_data.dropna()

train_data.drop(['city_development_index'], axis=1, inplace=True)

train_data['company_size'] = train_data['company_size'].replace(
    '10/49', '10-49')
train_data['company_size'].value_counts()

# train_data.gender = train_data.gender.fillna('Unidentified')

train_data.enrolled_university = train_data.enrolled_university.fillna(
    'No Data')

train_data.education_level = train_data.education_level.fillna('No Data')

train_data.major_discipline = train_data.major_discipline.fillna('No Data')

train_data.experience = train_data.experience.fillna('<1')

train_data['experience'].replace({'1': '1-5',
                                  '2': '1-5',
                                  '3': '1-5',
                                  '4': '1-5',
                                  '5': '1-5',
                                  '6': '6-10',
                                  '7': '6-10',
                                  '8': '6-10',
                                  '9': '6-10',
                                  '10': '6-10',
                                  '11': '11-15',
                                  '12': '11-15',
                                  '13': '11-15',
                                  '14': '11-15',
                                  '15': '11-15',
                                  '16': '16-20',
                                  '17': '16-20',
                                  '18': '16-20',
                                  '19': '16-20',
                                  '20': '16-20'}, inplace=True)

train_data.company_size = train_data.company_size.fillna('No Data')

train_data.company_type = train_data.company_type.fillna('No Data')

train_data.last_new_job = train_data.last_new_job.fillna('No Data')

train_data = train_data.dropna()

# train_data.isnull().sum()
print(train_data.isnull().sum())


# divide categorical columns and numerical columns
cat_col = []
num_col = []
for col in train_data.columns:
    if train_data[col].dtype == object:
        print('category coloumn : ', col)
        cat_col.append(col)
    else:
        print('numarical column : ', col)
        num_col.append(col)


# train_data
print(train_data)


to_LabelEncode = train_data[cat_col]

le = LabelEncoder()
df_temp = to_LabelEncode.astype("str").apply(le.fit_transform)
df_temp = df_temp.where(~to_LabelEncode.isna(), to_LabelEncode)

df_encod = df_temp.join(train_data[num_col])

# df_encod.shape
print(df_encod.shape)


# df_encod
print(df_encod)


# train_data_OHE = pd.get_dummies(data=train_data, columns=cat_col, drop_first=False)
# train_data_OHE


# train_data[num_col]
print(train_data[num_col])


plt.subplots(figsize=(20, 15))
sns.heatmap(df_encod.corr(), annot_kws={"size": 7}, annot=True)


# df_encod.corr()['target']
print(df_encod.corr()['target'])


# plt.plot(train_data_OHE.corr()['target'])


# sns.pairplot(train_data_OHE, hue='target')


# ### Split Train and Validation Set


y = df_encod['target']
X = df_encod.drop('target', axis=1)


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, shuffle=True, random_state=1, test_size=0.3)


# ### Decision Tree baseline


def roc_curve_plot(y_test, y_pred):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label="auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()


tree_clf = DecisionTreeClassifier(random_state=17)
tree_clf.fit(X_train, y_train)


tree_pred = tree_clf.predict(X_valid)
tree_pred_proba = tree_clf.predict_proba(X_valid)[:, 1]


print(classification_report(y_valid, tree_pred))


ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(
        y_valid, tree_pred)).plot()


roc_curve_plot(y_valid, tree_pred_proba)


print(
    'Train:', tree_clf.score(
        X_train, y_train), '\nTest: ', tree_clf.score(
            X_valid, y_valid))


# We see that the model is very much retrained


# Learning curve


def plot_learning_curve(estimator, X, y):
    train_sizes, train_score, test_score = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.01, 1.0, 50),
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        random_state=17)

    mean_train = np.mean(train_score, axis=1)
    mean_test = np.mean(test_score, axis=1)

    plt.plot(train_sizes, mean_train, '--', color="b", label="Training score")
    plt.plot(train_sizes, mean_test, color="g", label="Cross-validation score")

    plt.title('learning curve')
    plt.xlabel("size"),
    plt.ylabel("score"),
    plt.legend(loc="best")
    plt.show()


plot_learning_curve(tree_clf, X_train, y_train)


# We see a big variation on training and cross-qualification


# o begin with, let's try to tuning the hyperparameters of our model


# ### Hyperparameter Tuning DecisionTree


tree_param = {'max_depth': range(2, 20, 2),
              'min_samples_split': range(2, 52, 10),
              'min_samples_leaf': range(2, 20, 2)
              }

sf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)


rs = RandomizedSearchCV(
    tree_clf,
    tree_param,
    cv=sf,
    random_state=17,
    n_jobs=-1,
    verbose=1)
rs.fit(X_train, y_train)


rs.best_params_
print(rs.best_params_)


best_tree_param = {'min_samples_split': range(15, 28, 2),
                   'min_samples_leaf': range(12, 21),
                   'max_depth': range(3, 7)
                   }


gs = GridSearchCV(tree_clf, best_tree_param, cv=sf, n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)


best_tree = gs.best_estimator_
# gs.best_params_
print(gs.best_params_)


best_tree_pred = best_tree.predict(X_valid)
best_tree_pred_proba = best_tree.predict_proba(X_valid)[:, 1]


plt.figure(figsize=(17, 10))
tree.plot_tree(
    best_tree,
    filled=True,
    class_names=[
        '1',
        '0'],
    feature_names=X_train.columns)


tree_feature = pd.DataFrame(
    best_tree.feature_importances_,
    index=X_train.columns,
    columns=['result'])
# tree_feature
print(tree_feature)


print(
    'Train:', best_tree.score(
        X_train, y_train), '\nTest: ', best_tree.score(
            X_valid, y_valid))


plot_learning_curve(best_tree, X_train, y_train)


# overfitting is not visible


print(classification_report(y_valid, best_tree_pred))


ConfusionMatrixDisplay(confusion_matrix(y_valid, best_tree_pred)).plot()


# But we still see a preponderance of class 0, compared to 1


roc_curve_plot(y_valid, best_tree_pred_proba)


# Let's try to achieve a balance of classes for a better assessment


# ## Resampling Unbalanced class


plt.figure(figsize=(15, 10))
sns.countplot(x=y_train)


sm = SMOTE(random_state=17)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
ad = ADASYN(random_state=17)
X_train_ad, y_train_ad = ad.fit_resample(X_train, y_train)


X_train.shape, X_train_sm.shape, X_train_ad.shape
print(X_train.shape, X_train_sm.shape, X_train_ad.shape)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(x=y_train_sm, ax=ax[0])
sns.countplot(x=y_train_ad, ax=ax[1])


# Now having the same number of classes, let's try to train the model
# anew, with a new search for the best parameters


# SMOTE


sm_gs = GridSearchCV(tree_clf, best_tree_param, cv=sf, n_jobs=-1, verbose=1)
sm_gs.fit(X_train_sm, y_train_sm)


sm_best_tree = sm_gs.best_estimator_
# sm_gs.best_params_
print(sm_gs.best_params_)


sm_best_tree_pred = sm_best_tree.predict(X_valid)
sm_best_tree_pred_proba = sm_best_tree.predict_proba(X_valid)[:, 1]


plot_learning_curve(sm_best_tree, X_train_sm, y_train_sm)


print(classification_report(y_valid, best_tree_pred))


print(classification_report(y_valid, sm_best_tree_pred))


# As you can see, recall for class 1 has become higher


roc_curve_plot(y_valid, sm_best_tree_pred_proba)


# ADASYN


ad_gs = GridSearchCV(tree_clf, best_tree_param, cv=sf, n_jobs=-1, verbose=1)
ad_gs.fit(X_train_ad, y_train_ad)


ad_best_tree = ad_gs.best_estimator_
# ad_gs.best_params_
print(ad_gs.best_params_)


ad_best_tree_pred = ad_best_tree.predict(X_valid)
ad_best_tree_pred_proba = ad_best_tree.predict_proba(X_valid)[:, 1]


plot_learning_curve(ad_best_tree, X_train_ad, y_train_ad)


print(classification_report(y_valid, best_tree_pred))


print(classification_report(y_valid, sm_best_tree_pred))


print(classification_report(y_valid, ad_best_tree_pred))


# ADASAN shows even higher recall results for class 1, but also decreases
# for class 0


roc_curve_plot(y_valid, ad_best_tree_pred_proba)


# Now let's try to use Random forest as the main model


# ### Random forest


rs = RandomForestClassifier(random_state=17)


# I will not use baseline, but will immediately perform hypertuning of the
# parameters


rs_param = tree_param.copy()
rs_param['n_estimators'] = range(100, 2001, 100)
# rs_param
print(rs_param)


rs_rs = RandomizedSearchCV(
    rs,
    rs_param,
    cv=sf,
    n_jobs=-1,
    verbose=1,
    random_state=17)
rs_rs.fit(X_train, y_train)


# rs_rs.best_params_
print(rs_rs.best_params_)


best_rs = rs_rs.best_estimator_


# random forest on GridSearchCV will take a very long time to learn, so I
# will not run it, we will leave these parameters as the best


rs_pred = best_rs.predict(X_valid)
rs_pred_proba = best_rs.predict_proba(X_valid)[:, 1]


rs_feature = pd.DataFrame(
    best_rs.feature_importances_,
    index=X_train.columns,
    columns=['result']).sort_values(
        'result',
    ascending=False)
# rs_feature
print(rs_feature)


# Like Decision Trees, random forest considers city_development_index to
# be the main variable, but also added weight to the rest of the variables


print(
    'Train:', best_rs.score(
        X_train, y_train), '\nTest: ', best_rs.score(
            X_valid, y_valid))


plot_learning_curve(best_rs, X_train, y_train)


# We see a big gap between train store and cross score, and we also see
# that cross continues to grow as the size of the dataset increases


print(classification_report(y_valid, rs_pred))


ConfusionMatrixDisplay(confusion_matrix(y_valid, rs_pred)).plot()


roc_curve_plot(y_valid, rs_pred_proba)


# Now let's try to use Resample data on random forest


# ### Random forest with resample methods


sm_rs = best_rs.fit(X_train_sm, y_train_sm)
ad_rs = best_rs.fit(X_train_sm, y_train_sm)


sm_rs_pred = sm_rs.predict(X_valid)
sm_rs_pred_proba = sm_rs.predict_proba(X_valid)[:, 1]

ad_rs_pred = ad_rs.predict(X_valid)
ad_rs_pred_proba = ad_rs.predict_proba(X_valid)[:, 1]


plot_learning_curve(best_rs, X_train_sm, y_train_sm)


plot_learning_curve(best_rs, X_train_ad, y_train_ad)


print(classification_report(y_valid, sm_rs_pred))


print(classification_report(y_valid, ad_rs_pred))


roc_curve_plot(y_valid, sm_rs_pred_proba)


roc_curve_plot(y_valid, ad_rs_pred_proba)


final_result = pd.DataFrame(
    [
        [
            'Decision Tree', roc_auc_score(
                y_valid, tree_pred_proba)], [
                    'Decision Tree tuning', roc_auc_score(
                        y_valid, best_tree_pred_proba)], [
                            'Decision Tree tuning with SMOTE', roc_auc_score(
                                y_valid, sm_best_tree_pred_proba)], [
                                    'Decision Tree tuning with ADASYN', roc_auc_score(
                                        y_valid, ad_best_tree_pred_proba)], [
                                            'Random forest', roc_auc_score(
                                                y_valid, rs_pred_proba)], [
                                                    'Random forest tuning with SMOTE', roc_auc_score(
                                                        y_valid, sm_rs_pred_proba)], [
                                                            'Random forest tuning with ADASYN', roc_auc_score(
                                                                y_valid, ad_rs_pred_proba)]], columns=[
                                                                    'method', 'result']).sort_values(
                                                                        'result', ascending=False)


# # Final results


# final_result
print(final_result)
