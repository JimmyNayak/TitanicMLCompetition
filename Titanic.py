# In this challenge, we ask you to build a predictive model that answers the question:
# “what sorts of people were more likely to survive?” using passenger data
# (ie name, age, gender, socio-economic class, etc).


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics as sl

titanic_train_data = pd.read_csv('data/train.csv')
titanic_train_data.head()

titanic_test_data = pd.read_csv('data/test.csv')
titanic_test_data.head()

# titanic_train_data.info()
# titanic_test_data.info()
#
# print('Train columns with null values: {} \n'.format(titanic_train_data.isnull().sum()))
#
# print('Test columns with null values: {}'.format(titanic_test_data.isnull().sum()))
#
# print(titanic_train_data.describe())
# print(titanic_test_data.describe())
#
# # Train data clearing
#
# titanic_train_data['Age'].fillna(titanic_train_data['Age'].median(), inplace=True)
# titanic_train_data['Embarked'].fillna(titanic_train_data['Embarked'].mode()[0], inplace=True)
# titanic_train_data['Fare'].fillna(titanic_train_data['Fare'].median(), inplace=True)
#
# # Test data clearing
#
# titanic_test_data['Age'].fillna(titanic_test_data['Age'].median(), inplace=True)
# titanic_test_data['Embarked'].fillna(titanic_test_data['Embarked'].mode()[0], inplace=True)
# titanic_test_data['Fare'].fillna(titanic_test_data['Fare'].median(), inplace=True)
#
# print('Train columns with null values: {} \n'.format(titanic_train_data.isnull().sum()))
#
# print('Test columns with null values: {}'.format(titanic_test_data.isnull().sum()))
#
# # Dropping the following ['PassengerId','Cabin', 'Ticket'] COLUMNS
# drop_column = ['PassengerId', 'Cabin', 'Ticket']
# titanic_train_data.drop(drop_column, axis=1, inplace=True)
# titanic_test_data.drop(drop_column, axis=1, inplace=True)

women = titanic_train_data.loc[titanic_train_data.Sex == 'female']["Survived"]
rate_women = sum(women) / len(women)

men = titanic_train_data.loc[titanic_train_data.Sex == 'male']["Survived"]
rate_men = sum(men) / len(men)

y = titanic_train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(titanic_train_data[features])
X_test = pd.get_dummies(titanic_test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

# print(sl.accuracy_score(titanic_test_data, predictions))

output = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
