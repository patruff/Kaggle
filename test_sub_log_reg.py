# Import the linear regression class
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold
import numpy as np
import pandas
from sklearn import cross_validation

# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the titanic_test variable.
titanic_test = pandas.read_csv("test.csv")
titanic = pandas.read_csv("titanic_train.csv")

# Print the first 5 rows of the dataframe.
print(titanic_test.head(5))
print(titanic_test.describe())


#fill in missing values of age with the median age
titanic_test["Age"]=titanic["Age"].fillna(titanic["Age"].median())


# Find all the unique genders -- the column appears to contain only male and female.
print(titanic_test["Sex"].unique())

# Replace all the occurences of male with the number 0 and female with 1
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

#fill the null values of Fare in the test set with the median value
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())


# Find all the unique values for "Embarked".
print(titanic_test["Embarked"].unique())

#fill the null values in Embarked with the most common value (in this case 'S')
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"]=="S", "Embarked"]=0
titanic_test.loc[titanic_test["Embarked"]=="C", "Embarked"]=1
titanic_test.loc[titanic_test["Embarked"]=="Q", "Embarked"]=2


#fill in missing values of age with the median age
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())


# Find all the unique genders -- the column appears to contain only male and female.
print(titanic["Sex"].unique())

# Replace all the occurences of male with the number 0 and female with 1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

#fill the null values of Fare in the test set with the median value
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())


# Find all the unique values for "Embarked".
print(titanic["Embarked"].unique())

#fill the null values in Embarked with the most common value (in this case 'S')
titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"]=="S", "Embarked"]=0
titanic.loc[titanic["Embarked"]=="C", "Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q", "Embarked"]=2


# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


# Initialize the algorithm class
alg = LogisticRegression(random_state=1)

# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("kaggle.csv",index=False)

'''
# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic_test dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic_test.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic_test[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic_test["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic_test[predictors].iloc[test,:])
    predictions.append(test_predictions)
    
 
# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == titanic_test["Survived"]]) / len(predictions)

# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic_test[predictors], titanic_test["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())'''
