# todo:PRACTICAL-08

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predictions)
accuracy_percentage = accuracy * 100
print(f"Gaussian Naive Bayes model Accuracy(in %): {accuracy_percentage:.2f}%")
