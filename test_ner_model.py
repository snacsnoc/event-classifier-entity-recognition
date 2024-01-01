# Test the model
# Load the trained model
import spacy


nlp = spacy.load("models/parser")

print("################################################################")
# Test data - sentences that the model has not seen before
test_data = [
    "Plan a webinar Product Launch Announcement for next Friday at 11:30am",
    "Organize a workshop Employee Training Session on October 5th at 9:00am",
    "Add an event called Board Meeting for November 12th at 3:30pm",
    "Create an event Team Lunch on July 8th at 12:30pm",
    "Schedule a team meeting Project Status Update with john@example.com butt@google.com on September 15 at 10am",
    "Add a presentation Product Launch for November 5th at 2:00 pm",
]

# Evaluate the model on new data
for text in test_data:
    doc = nlp(text)
    print(f"Text: {text}")
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print()