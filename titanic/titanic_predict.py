import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib as plt

#Predict using all the data
df = pd.read_csv('titanic_with_ages.csv')

def title(name):
    splitter = name.split(" ")
    common = ["Mr.", "Miss.", "Master.", "Mrs."]
    royalty = ["Jonkeer.", "Don.", "Sir.", "Countess.", "Dona.", "Lady."]
    crew = ["Capt.", "Col.", "Major.", "Dr.", "Rev."]
    for split in splitter:
        if split in common:
            return common.index(split)
        elif split in royalty:
            return 10
        elif split in crew:
            return 20
    return 99

def testAge(x):
    if x < 10:
        return 0 #0 is young
    if x > 50:
        return 1 #1 is old
    else:
        return 2 #adult

def cabin(no):
    no = str(no)
    numbers = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    if no[0] in numbers:
        return numbers.index(no[0])
    return 99

df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df.Sex.cat.codes
df['Embarked'] = df['Embarked'].astype('category')
df['Embarked'] = df.Embarked.cat.codes
df['Title'] = df['Name'].apply(title)
df['Family Size'] = df['SibSp']+df['Parch']
df['Cabin'] = df['Cabin'].apply(cabin)

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age', 'Title', 'Family Size', 'Cabin']


Xtrain = df[features]
Ytrain = df['Survived'].values

#kfold = model_selection.KFold(n_splits=10) #Provides train/test indices to split df in train/test sets. Split dfset into k consecutive folds (without shuffling by default).
model = RandomForestClassifier()
model.fit(Xtrain, Ytrain) #Model is now trained on the training data!

test = pd.read_csv('titanic_test.csv')
test['Sex'] = test['Sex'].astype('category')
test['Sex'] = test.Sex.cat.codes
test['Embarked'] = test['Embarked'].astype('category')
test['Embarked'] = test.Embarked.cat.codes
test['Title'] = test['Name'].apply(title)
test['Family Size'] = test['SibSp']+test['Parch']
test['Cabin'] = test['Cabin'].apply(cabin)

#There is a Fare entry with nan, need to replace with the average
test['Fare'].fillna( test.Fare.mean(), inplace=True )
test['Age'].fillna( test.Age.mean(), inplace=True )
predictions = model.predict(test[features])

#test.loc[:,'Survived'] = predictions #Insert back in
test.insert(1, 'Survived', predictions)
output = pd.concat([test['PassengerId'], test['Survived']], axis=1, keys=['PassengerId', 'Survived'])

print(output.head())
output.to_csv("titanic_predicted3.csv", mode = 'w', index=False)