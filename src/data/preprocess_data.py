# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

def process_row(row):
    """
    Process a row of the dataset to determine the order of the 'toxic' and 'neutral' columns.
    
    The row will be re-ordered so that the more toxic text is labeled as 'toxic'
    and the less toxic text is labeled as 'neutral'.
    
    Parameters:
        row (Series): A row of the dataframe.
    
    Returns:
        Series: A pandas Series with ordered fields.
    """
    # Compare toxicity scores of the reference and the translation
    if row['ref_tox'] > row['trn_tox']:
        # If reference is more toxic, return reference first
        return pd.Series([row['reference'], row['translation'], row['ref_tox'], row['trn_tox']])
    else:
        # If translation is more toxic, return translation first
        return pd.Series([row['translation'], row['reference'], row['trn_tox'], row['ref_tox']])

# Load the raw data from a TSV file.
data = pd.read_csv(r'../../data/raw/filtered.tsv', sep='\t')

# Apply the `process_row` function to each row of the dataframe.
toxic_neutral_data = data.apply(process_row, axis=1)

# Rename the columns of the resulting dataframe.
toxic_neutral_data.columns = ['toxic', 'neutral', 'toxicity score', 'toxicity of neutral score']

# Save the processed data to a new CSV file for further use.
toxic_neutral_data.to_csv('../../data/interim/toxic_neutral_data.csv', index=False)

# Reload the processed data.
toxic_neutral_data = pd.read_csv("../../data/interim/toxic_neutral_data.csv")

# Split the data into a training set and a testing set with a test size of 20%.
train_df, test_df = train_test_split(toxic_neutral_data, test_size=0.2, random_state=42)

# Save the split datasets to their respective CSV files.
train_df.to_csv(r"../../data/interim/training.csv", index=False)
test_df.to_csv(r"../../data/interim/testing.csv", index=False)