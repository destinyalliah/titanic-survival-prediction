import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')

#finding out how many rows and columns
print(train.shape)

#print the first 5 rows
print(train.head())

#print the column types and missing values
print(train.info())

#print the basic statistics
print(train.describe())

#check for missing values
print(train.isnull().sum())

# Survival rate by sex
print(train.groupby('Sex')['Survived'].mean())

# Survival rate by passenger class
print(train.groupby('Pclass')['Survived'].mean())

# Age distribution
train['Age'].hist(bins=30)
plt.title('Age distribution')
plt.show()

# Survival by sex and class together
sns.barplot(data=train, x='Pclass', y='Survived', hue='Sex')
plt.title('Survival rate by class and sex')
plt.show()

def clean(df):
    df = df.copy()

    #Fill missing values
    #use median 
    df['Age'] = df['Age'].fillna(df['Age'].median())
    #median fare
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    #embarked column tells you which port each passenger boarded the Titanic from
    df['Embarked'] = df['Embarked'].fillna('S')


    #convert text columns to number (ML models need numbers)
    #female = 0, male = 1
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

    df = df.drop(['Cabin', 'Name', 'Ticket'], axis = 1)

    return df

train = clean(train)
test = clean(test)

print(train.isnull().sum())

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = train[features]
y = train['Survived']
X_test = test[features]

#Train the model
model = RandomForestClassifier(
    n_estimators = 100,
    max_depth = 5,
    random_state = 42
)

model.fit(X, y)

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV accuracy: {scores.mean():.3f} ± {scores.std():.3f} ")

predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

submission.to_csv('data/submission.csv', index=False)
print(f"Saved {len(submission)} predictions!")
