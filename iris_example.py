from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()

# Create a DataFrame (like a spreadsheet)
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the species column
df['species'] = [iris.target_names[i] for i in iris.target]

# Display the complete dataset
print(df)
