from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier()
model.fit(X, y)

sample = [[5.1, 3.5, 1.4, 0.2]]
predicted = model.predict(sample)
print(predicted)  # 0=setosa, 1=versicolor, 2=virginica
