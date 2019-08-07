import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
from os import system


# Creates Pandas dataframe from csv file
df = pd.read_csv("MockGraduationData.csv")


# Converts completed field from yes/no into 1/0
df['Completed Program'] = (df['Completed Program'] == "Yes").astype(int)

# Converts "1st Gen" field from yes/no into 1/0
df['1st Gen'] = (df['1st Gen'] == "yes").astype(int)

# Converts "Working" field into numerical values
# "Working" field Key: 0 = Not Working, 1 = PartTime, 2 = Fulltime
df.loc[df['Working'] == "Fulltime", 'Working'] = 2
df.loc[df['Working'] == "Parttime", 'Working'] = 1
df.loc[df['Working'] == "no", 'Working'] = 0

# Removes student id from decision tree
df = df.drop("ID", axis=1)

# Seperates classification column and attributes columns
X = df.drop('Completed Program', axis=1)
y = df['Completed Program']

# Splits data into 80% training set and 20% Testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# Creates and trains decision tree classifier
classifier = DecisionTreeClassifier(random_state = 0, max_depth = 10)
classifier.fit(X_train, y_train)

print(classifier.score(X_test, y_test))


# Creates .Dot file that can be used to visualize tree at "http://www.webgraphviz.com/"
from sklearn.tree import export_graphviz
export_graphviz(classifier, out_file='tree.dot',
                rounded = True, proportion = False,
                precision = 2, filled = True)


# Collects input fields for build student attributes for classification
GPA = input("What is the students GPA? (no decimal plz)")
GPA = int(float(GPA))
Prior_Degree = input("Does the student have a prior degree? (y/n)")
if Prior_Degree.lower == "y" or Prior_Degree.lower == "yes":
    Prior_Degree = 1
else:
    Prior_Degree = 0
Test_Score = input("What is the students Entry Exam Score (1-10)?")
Test_Score = int(Test_Score)
Income =  input("What is the students household income?")
Income = int(Income)
Working = input("is the student working (Fulltime/Parttime/no)")
if Working.lower == "parttime" :
    Working = 1
elif Working.lower == "fulltime" :
    Working = 2
else:
    Working = 0
First_Gen = input("is this a first gen student? (y/n)")
if First_Gen.lower == "y" or First_Gen.lower == "yes":
    First_Gen = 1
else:
    First_Gen = 0

print(df.head())
Student_Attributes = [GPA, Prior_Degree, Test_Score, Income, Working, First_Gen]

# Runs prediction with input values and reports results
y_predict = classifier.predict([Student_Attributes])
if y_predict > 0:
    print("This model predicts this student will complete the program")
if y_predict == 0:
    print("This model predicts this student will not complete the program")
