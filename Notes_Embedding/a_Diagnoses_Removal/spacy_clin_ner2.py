import pandas as pd
import spacy
import scispacy

spacy.prefer_gpu()
# Load the spaCy model
nlp = spacy.load("en_ner_bc5cdr_md")

# Define the function that will be applied to each batch of documents
def modify_text_in_docs(docs):
    modified_texts = []
    for doc in docs:
        modified_text = doc.text
        for ent in doc.ents:
            if ent.label_ == 'DISEASE':
                modified_text = modified_text[:ent.start_char] + 'X' * (ent.end_char - ent.start_char) + modified_text[ent.end_char:]
        modified_texts.append(modified_text)
    return modified_texts

# Load the clinical notes DataFrame
clinical_notes = pd.read_csv("../../Data/MIMIC_resources/NOTEEVENTS.csv")

# Processing a subset for demonstration; adjust as needed
test_notes = clinical_notes.head(1000).copy()  # Create an explicit copy to avoid SettingWithCopyWarning
print('We start working....')

# Process the text data in batches using spaCy's nlp.pipe
modified_texts = []
for doc in nlp.pipe(test_notes['TEXT'], batch_size=20):
    modified_text = doc.text
    for ent in doc.ents:
        if ent.label_ == 'DISEASE':
            # Replace detected disease entities with 'X' characters
            modified_text = modified_text[:ent.start_char] + 'X' * (ent.end_char - ent.start_char) + modified_text[ent.end_char:]
    modified_texts.append(modified_text)

# Safely assign the modified texts to a new column in the DataFrame
test_notes['modified_text'] = modified_texts  # This is now safe as we're working with a copy

# Save the DataFrame with the modified notes to a new CSV file
test_notes.to_csv("../../Data/MIMIC_resources/MODIFIED_NOTEEVENTS.csv", index=False)

print('Mission accomplished!')