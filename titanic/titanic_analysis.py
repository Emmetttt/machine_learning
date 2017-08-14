import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib as plt

df = []
#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

df = pd.read_csv('titanic_with_ages.csv')
#print(df.info())
#print(df['Name'].head())

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

def cabin(no):
    no = str(no)
    numbers = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    if no[0] in numbers:
        return numbers.index(no[0])
    return 99

print(title("Braund, Mr. Owen Harris"))
#df.loc[df['Sex'] == 'male', 'Sex'] = 1      #Make the gender binary
df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df.Sex.cat.codes
df['Embarked'] = df['Embarked'].astype('category')
df['Embarked'] = df.Embarked.cat.codes
df['Title'] = df['Name'].apply(title)
df['Family Size'] = df['SibSp']+df['Parch']
df['Cabin'] = df['Cabin'].apply(cabin)

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age', 'Title', 'Family Size', 'Cabin']

'''Next we shall split our df up into two subsidiaries, train and test.
https://chrisalbon.com/machine-learning/random_forest_classifier_example_scikit.html
We put a random 75% into train, and the other 25% into test'''
df['train?'] = np.random.uniform(0, 1, len(df)) <= 0.70 #Creates new column, inputs True if rand <= 0.75

##Split up the data
train, test = df[df['train?'] == True], df[df['train?'] == False]

Xtrain = train[features]
Ytrain = train['Survived'].values


##FIT MODEL##
#kfold = model_selection.KFold(n_splits=10) #Provides train/test indices to split df in train/test sets. Split dfset into k consecutive folds (without shuffling by default).
model = RandomForestClassifier()

model.fit(Xtrain, Ytrain) #Model is now trained on the training data!

print(model.score(test[features], test['Survived'].values)) #How accurate is the model???
#print(model.predict(test[features])) #Predictions on the test features
#print(model.predict_proba(test[features])[:10]) #Probabilities on the tests
'''
##Confusion Matrix
#Look at the predictions
predictions = model.predict(test[features])

#Look at the reality
reality = test['Survived'].values

#Create a confusion matrix, anything not on the main diagnal is a poor estimate
conf_matrix = pd.crosstab(reality, predictions, rownames=['Actual state'], colnames=['Predicted State'])
print(conf_matrix)


##Feature Importance
feat_importance = list(zip(train[features], model.feature_importances_))
print(feat_importance)

'''