import pandas as pd
import spacy
import scispacy
import concurrent.futures
import numpy as np


# ---------------------- FUNCTIONS----------------------------
# Function to modify the note

def modify_note(note):
    doc = nlp(note)
    modified_text = note
    for ent in doc.ents:
        if ent.label_ == 'DISEASE':
            # Replace the text in the original note with 'X' characters using the entity's span
            modified_text = modified_text[:ent.start_char] + 'X' * (ent.end_char - ent.start_char) + modified_text[ent.end_char:]
    return modified_text

# --------------------------------------------------------------
# Function to process a chunk of the DataFrame

def process_chunk(chunk):
    chunk['NEW_TEXT'] = chunk['TEXT'].apply(modify_note)
    return chunk

# --------------------------------------------------------------
# Function to parallelize processing

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        df = pd.concat(executor.map(func, df_split))
    return df

# --------------------------------------------------------------



# Load the spaCy model
nlp = spacy.load("en_ner_bc5cdr_md")

# Load the clinical notes DataFrame
clinical_notes = pd.read_csv("../../Data/MIMIC_resources/NOTEEVENTS.csv")
print(clinical_notes.shape)

# --------------------------------------------------------------
# NUM_REC = 1000
# test_notes = clinical_notes.head(NUM_REC).copy()  


# --------------------------------------------------------------
NUM_REC = clinical_notes.shape[0]
test_notes = clinical_notes.head(NUM_REC).copy()

# --------------------------------------------------------------
print(f'Here we go....processing {NUM_REC}')


# --------------------------------------------------------------
# Parallelize the `tokenize_text` function across the dataframe
df_processed = parallelize_dataframe(test_notes, process_chunk, n_cores=32)  # Adjust `n_cores` as needed

# Apply the modification function to the entire clinical_notes dataframe
test_notes['TEXT'] = test_notes['TEXT'].apply(modify_note)


# Save the DataFrame with the modified notes to a new CSV file
test_notes.to_csv(f'../../Data/MIMIC_resources/MODIFIED_NOTEEVENTS_{NUM_REC}.csv', index=False)

print('Mission accomplished!')