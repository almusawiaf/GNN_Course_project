import pandas as pd
import spacy
import scispacy

# Load the spaCy model
nlp = spacy.load("en_ner_bc5cdr_md")

# Load the clinical notes
clinical_notes = pd.read_csv("../Data/MIMIC_resources/NOTEEVENTS.csv")


# Function to modify the note
def modify_note(note):
    doc = nlp(note)
    modified_text = note
    for ent in doc.ents:
        if ent.label_ == 'DISEASE':
            # Replace the text in the original note with 'X' characters using the entity's span
            modified_text = modified_text[:ent.start_char] + 'X' * (ent.end_char - ent.start_char) + modified_text[ent.end_char:]
    return modified_text

# Apply the modification function to the entire clinical_notes dataframe
clinical_notes['Modified_Note'] = clinical_notes['TEXT'].apply(modify_note)

clinical_notes.to_csv('../Data/MIMIC_resources/NOTEEVENTS_NER.csv"', index=False)