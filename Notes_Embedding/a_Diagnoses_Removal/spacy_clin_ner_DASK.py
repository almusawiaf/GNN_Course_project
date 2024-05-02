import dask.dataframe as dd
import pandas as pd
import spacy
import scispacy

# Ensure spaCy utilizes the GPU if available
spacy.prefer_gpu()

# Load the spaCy model
nlp = spacy.load("en_ner_bc5cdr_md")

# Define the function that modifies text within documents
def modify_text_in_docs(docs):
    modified_texts = []
    for doc in nlp.pipe(docs, batch_size=20):
        modified_text = doc.text
        for ent in doc.ents:
            if ent.label_ == 'DISEASE':
                modified_text = modified_text[:ent.start_char] + 'X' * (ent.end_char - ent.start_char) + modified_text[ent.end_char:]
        modified_texts.append(modified_text)
    return modified_texts

# Function to be applied to each partition of the Dask DataFrame
def process_partition(partition):
    partition['modified_text'] = modify_text_in_docs(partition['TEXT'].values)
    return partition

# Load the clinical notes into a Dask DataFrame
dd_clinical_notes = dd.read_csv("../../Data/MIMIC_resources/NOTEEVENTS.csv").head(100)

# Depending on your dataset's size, you might need to adjust the number of partitions
dd_clinical_notes = dd_clinical_notes.repartition(npartitions=10)

print('We start working....')

# Apply the processing function to each partition of the DataFrame
# This operation is lazy and won't compute until explicitly told to do so
dd_modified_notes = dd_clinical_notes.map_partitions(process_partition, meta=dd_clinical_notes)

# Compute the result, converting back to a pandas DataFrame
# This step might take some time depending on the size of your dataset and the capabilities of your system
modified_notes = dd_modified_notes.compute()

# Save the modified notes to a new CSV file
modified_notes.to_csv("../../Data/MIMIC_resources/MODIFIED_NOTEEVENTS.csv", index=False, single_file=True)

print('Mission accomplished!')
