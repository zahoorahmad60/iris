import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the Iris dataset
iris_data = pd.read_csv('iris.csv')
iris_data.drop('Id',axis=1,inplace=True)
# print(iris_data.head(5))

# Split the data into features (X) and labels (y)
X = iris_data.drop('Species', axis=1)
y = iris_data['Species']
print(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


pickle.dump(model, open("model.pkl", "wb"))
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
