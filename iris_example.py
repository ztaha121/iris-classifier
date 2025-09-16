from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create and train model
model = DecisionTreeClassifier()
model.fit(X, y)

# New samples: multiple flowers
samples = [
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 3.0, 4.8, 1.8],
    [5.9, 3.0, 5.1, 1.8]
]

# Predict
predicted = model.predict(samples)

# Map numbers to species names
species_names = [iris.target_names[i] for i in predicted]

# Print results
for i, flower in enumerate(samples):
    print(f"Flower {i+1} with measurements {flower} is predicted as {species_names[i]}")
