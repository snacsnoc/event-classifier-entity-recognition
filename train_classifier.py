import pickle
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import os

# Ensure the tokenizer is available
nltk.download("punkt")
training_data = [
    ("schedule a meeting", "SCHEDULE"),
    ("set up an event", "SCHEDULE"),
    (
        "create a meeting product development recap with bill.g@microsoft.com for friday at 9am",
        "SCHEDULE",
    ),
    ("set up an event birthday party for 7pm tomorrow", "SCHEDULE"),
    ("organize a team call next Monday", "SCHEDULE"),
    ("book a room for a conference on Wednesday", "SCHEDULE"),
    ("arrange a webinar session", "SCHEDULE"),
    ("cancel the appointment", "CANCEL"),
    ("abort the scheduled meeting", "CANCEL"),
    ("call off the event planned for next week", "CANCEL"),
    ("remove the dinner reservation", "CANCEL"),
    ("postpone the scheduled training", "CANCEL"),
    ("schedule a meeting for next Wednesday", "SCHEDULE"),
    ("set up a team call tomorrow at 10 AM", "SCHEDULE"),
    ("create meeting Product Dev Recap next friday at 1pm", "SCHEDULE"),
    (
        "add an event Team Meeting with XXXXXXXXXXXXXXXX for next Monday at 10am",
        "SCHEDULE",
    ),
    ("add a calendar event for the project deadline on July 5th", "SCHEDULE"),
    ("book a conference room for the sales meeting next Friday", "SCHEDULE"),
    ("organize a lunch meeting with the client on Monday", "SCHEDULE"),
    ("create a meeting for team building activities next month", "SCHEDULE"),
    ("arrange a video call with the marketing team next Thursday", "SCHEDULE"),
    ("plan a workshop session on the 15th", "SCHEDULE"),
    ("remind me to discuss the budget in tomorrow's meeting", "SCHEDULE"),
    ("set a reminder for the performance review next week", "SCHEDULE"),
    ("cancel my 3 PM appointment today", "CANCEL"),
    ("I need to call off the meeting scheduled for this evening", "CANCEL"),
    ("postpone the webinar that was planned for tomorrow", "CANCEL"),
    ("remove the lunch plan from the calendar", "CANCEL"),
    ("I won't make it to the team dinner, please cancel it", "CANCEL"),
    ("abort the planning session we had for next Monday", "CANCEL"),
    ("delete the event set for my project presentation", "CANCEL"),
    ("Can we talk about the budget", "UNKNOWN"),
    ("Just checking in on the project status", "UNKNOWN"),
    ("Don't forget the team outing next month", "UNKNOWN"),
    ("Any updates on the client feedback", "UNKNOWN"),
    ("Let me know if you need anything", "UNKNOWN"),
    ("What time is it", "UNKNOWN"),
    ("Just a reminder to submit the reports", "UNKNOWN"),
    ("Who is attending the seminar next week", "UNKNOWN"),
    ("I'll be out of the office tomorrow", "UNKNOWN"),
    ("Need assistance with the new software update", "UNKNOWN"),
    ("What's the deadline for the marketing proposal", "UNKNOWN"),
    ("Is the team lunch still happening next week", "UNKNOWN"),
    ("Can you please send me the report", "UNKNOWN"),
    ("hey", "UNKNOWN"),
    ("help me", "UNKNOWN"),
    ("what do i do", "UNKNOWN"),
    ("how do I use this thing", "UNKNOWN"),
    ("what is this", "UNKNOWN"),
    ("hey hows it going, how is your day", "UNKNOWN"),
    ("convert 3pm EST to PST", "TZCONVERT"),
    ("what's 10 am in London time in New York", "TZCONVERT"),
    ("change 2pm Central Time to GMT", "TZCONVERT"),
    ("show me 5pm Tokyo time in EST", "TZCONVERT"),
    ("convert 7am CST to IST", "TZCONVERT"),
    ("what is 9pm Paris time in CST", "TZCONVERT"),
    ("change 11am in Berlin to Eastern Time", "TZCONVERT"),
    ("convert 1pm Pacific Time to Australian Eastern Time", "TZCONVERT"),
    ("what will be the time in Dubai when it's 8am in New York", "TZCONVERT"),
    ("change 4pm in Singapore to London time", "TZCONVERT"),
    ("convert my 6pm to Tokyo time", "TZCONVERT"),
    ("what's 12pm in Moscow in Eastern Standard Time", "TZCONVERT"),
    ("change 3pm in Madrid to Mountain Time", "TZCONVERT"),
    ("convert 8am GMT to India Standard Time", "TZCONVERT"),
    ("what's 9pm in Rome in Central Time", "TZCONVERT"),
]


def word_feats(words):
    return dict([(word, True) for word in words])


all_feats = [
    (word_feats(word_tokenize(text)), intent) for (text, intent) in training_data
]
classifier = NaiveBayesClassifier.train(all_feats)

# Ensure the directory exists before saving the file
if not os.path.exists("models"):
    os.makedirs("models")

# Save the trained classifier to a file in the specified directory
with open("models/classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)
