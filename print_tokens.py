# Get start and end indexes of tokens
import spacy


# Load a blank English model
nlp = spacy.blank("en")

# List of example sentences
example_sentences = [
    "Schedule a meeting Marketing Campaign Review for next Thursday at 10:00 am",
    "Organize a seminar Sales Strategy Discussion with partners@company.com on August 10 at 2:45 pm",
    "Add a conference call Project Status Update with team@company.com on May 20 at 1:15pm",
    "Set up a board meeting Quarterly Review for October 2 at 4:30 pm",

]


# Function to print tokens and their indices for each sentence
def print_token_indices(nlp, text):
    doc = nlp(text)
    for token in doc:
        start_index = token.idx
        end_index = start_index + len(token)
        print(
            f"Token: {token.text}, Start Index: {start_index}, End Index: {end_index}"
        )


# Iterate over each example sentence
for sentence in example_sentences:
    print(f"Sentence: {sentence}")
    print_token_indices(nlp, sentence)
    print()