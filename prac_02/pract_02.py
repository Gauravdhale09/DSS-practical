# todo:PRACTICAL:02

import pandas as pd

df = pd.read_csv('Customers.csv')

print("Original Dataset:")
print(df.head())

grouped_stats = df.groupby('Age')['Annual Income ($)'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()

print("\nSummary Statistics Grouped by Age Group:")
print(grouped_stats)
