import pandas as pd
import numpy as np

# Sample data
data = {
    'Color': ['Red', 'Green', 'Blue', 'Green', 'Blue', 'Red'],
    'Size': ['S', 'M', 'L', 'S', 'M', 'L']
}

# Create DataFrame
df = pd.DataFrame(data)

# 1. Ordinal Encoding
def ordinal_encode(df, column, categories):
    """Performs ordinal encoding of a categorical column based on provided categories"""
    mapping = {category: index for index, category in enumerate(categories)}
    df[column] = df[column].map(mapping)
    return df

# Ordinal encoding for 'Size' (order: S -> 0, M -> 1, L -> 2)
df_ordinal = ordinal_encode(df.copy(), 'Size', ['S', 'M', 'L'])
print("Ordinal Encoding:\n", df_ordinal)

# 2. One-Hot Encoding
def one_hot_encode(df, column):
    """Performs one-hot encoding of a categorical column"""
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df.drop(column, axis=1), one_hot], axis=1)
    return df

# One-hot encoding for 'Color'
df_one_hot = one_hot_encode(df.copy(), 'Color')
print("\nOne-Hot Encoding:\n", df_one_hot)