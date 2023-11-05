import pandas as pd
from sklearn.model_selection import train_test_split


def process_row(row):
    if row['ref_tox'] > row['trn_tox']:
        return pd.Series([row['reference'], row['translation'], row['ref_tox'], row['trn_tox']])
    else:
        return pd.Series([row['translation'], row['reference'], row['trn_tox'], row['ref_tox']])


data = pd.read_csv(r'../../data/raw/filtered.tsv', sep='\t')

toxic_neutral_data = data.apply(process_row, axis=1)

toxic_neutral_data.columns = ['toxic', 'neutral', 'toxicity score', 'toxicity of neutral score']
toxic_neutral_data.to_csv('../../data/interim/toxic_neutral_data.csv', index=False)


toxic_neutral_data = pd.read_csv("../../data/interim/toxic_neutral_data.csv")


train_df, test_df = train_test_split(toxic_neutral_data, test_size=0.2, random_state=42)

train_df.to_csv(r"../../data/interim/training.csv", index=False)
test_df.to_csv(r"../../data/interim/testing.csv", index=False)
