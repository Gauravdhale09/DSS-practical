# download Iris.csv from kaggle platform
import pandas as pd
iris_df = pd.read_csv('Iris.csv')
setosa_data = iris_df[iris_df['Species'] == 'Iris-setosa']
versicolor_data = iris_df[iris_df['Species'] == 'Iris-versicolor']
verginica_data = iris_df[iris_df['Species'] == 'Iris-verginica']
print("Iris-setosa statistics:")
print(setosa_data.describe())

print("\n Iris-versicolor statistics:")
print(versicolor_data.describe())

print("\n Iris-verginica statistics:")
print(verginica_data.describe())
