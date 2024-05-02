## Processing the embedding, as follow:
# - make a list of patients
# - for each patient, get the list of embedding.
# - sum the embedding of the patients.
# - save it back to the final embedding df

import pandas as pd
import numpy as np

# Function to convert embedding string to numpy array
def convert_embedding(s):
    # Remove the leading and trailing characters and replace the newline with spaces
    cleaned = s.strip("[]'").replace('\n', ' ')
    # Convert string to numpy array
    return np.fromstring(cleaned, sep=' ')


emb        = pd.read_csv("../Data/emb.csv")
NoteEvents = pd.read_csv("../Data/MIMIC_resources/NOTEEVENTS.csv")

df1 = emb
df2 = NoteEvents

df1 = pd.merge(df1, df2[['ROW_ID', 'SUBJECT_ID', 'HADM_ID']], left_on='Clinical Note', right_on='ROW_ID', how='left')

df1.drop('Clinical Note', axis=1, inplace=True)
df1.drop('ROW_ID', axis=1, inplace=True)

print('Working...\n')

df = df1
# Apply the conversion function to the 'Embedding' column
df['Embedding'] = df['Embedding'].apply(convert_embedding)

# Sum embeddings by 'SUBJECT_ID'
# This groups the DataFrame by 'SUBJECT_ID', sums each group, and resets the index
df2 = df.groupby('SUBJECT_ID')['Embedding'].apply(lambda x: np.sum(np.vstack(x), axis=0)).reset_index()

df2.to_csv('../Data/grouped_emb_all.csv')
print('mission accomplished!')