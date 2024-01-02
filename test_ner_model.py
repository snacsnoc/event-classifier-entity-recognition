# Test the model
# Load the trained model
import spacy
import time

nlp = spacy.load("models/parser")

# Test data - sentences that the model has not seen before
test_data = [
    "Plan a webinar Product Launch Announcement for next Friday at 11:30am",
    "Organize a workshop Employee Training Session on October 5th at 9:00am",
    "Add an event called Board Meeting for November 12th at 3:30pm",
    "Create an event Team Lunch on July 8th at 12:30pm",
    "Schedule a team meeting Project Status Update with john@example.com butt@google.com on September 15 at 10am",
    "Add a presentation Product Launch for November 5th at 2:00 pm",
    "Schedule a conference Technology Trends in 2024 on Friday from 2pm to 4pm",
    "Organize a client meet-up at Downtown Cafe on August 21st at 1pm",
    "Book a brainstorming session for the marketing strategy with team@company.com on April 17 at 3:15 pm",
    "Set an appointment for Annual Health Check with dr.smith@healthcare.com on March 22nd at 10am",
    "Arrange an office party for New Year's Eve at The Grand Hall on December 31st, starting at 8pm",
    "Plan a seminar titled 'Innovations in Renewable Energy' with experts jane@energy.com, bob@innovate.com on June 30th at 9am",
    "Add a training session Social Media Marketing Techniques on next Wednesday at 11am, ending at 1pm",
    "Create a team outing to the National Park on May 5th, starting at 9am and concluding by 5pm",

]

# Evaluate the model on new data
for text in test_data:
    start_time = time.time()
    doc = nlp(text)
    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Text: {text}")
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print(f"Processing Time: {processing_time:.4f} seconds")
    print()