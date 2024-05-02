import pandas as pd
import spacy
from tqdm.auto import tqdm  # Optional, for progress bar

# Ensure spaCy uses GPU
spacy.require_gpu()
nlp = spacy.load("en_ner_bc5cdr_md")

# Load the clinical notes
clinical_notes = pd.read_csv("../Data/MIMIC_resources/NOTEEVENTS.csv")

# Function to process and modify a batch of notes
def process_and_modify_batch(notes_batch):
    modified_notes = []
    for doc in nlp.pipe(notes_batch, batch_size=20):  # Adjust batch_size based on your GPU's memory capacity
        modified_text = doc.text
        for ent in doc.ents:
            if ent.label_ == 'DISEASE':
                start = ent.start_char
                end = ent.end_char
                modified_text = modified_text[:start] + 'X' * (end - start) + modified_text[end:]
        modified_notes.append(modified_text)
    return modified_notes

# Function to apply modifications in batches and utilize GPU acceleration
def modify_notes_gpu(clinical_notes):
    notes = clinical_notes['TEXT'].tolist()
    modified_notes = process_and_modify_batch(notes)  # Assuming you can fit your dataset or process it in chunks
    return modified_notes

# Modify clinical notes and update DataFrame
clinical_notes['Modified_Note'] = modify_notes_gpu(clinical_notes)

# Save the modified clinical notes to a new CSV file
clinical_notes.to_csv('../Data/MIMIC_resources/NOTEEVENTS_NER.csv', index=False)
