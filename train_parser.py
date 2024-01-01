import spacy
from spacy.training.example import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
import random

TRAIN_DATA = [
    (
        "add an event Pick Up Dog Poop in the yard for next friday at 9am",
        {
            "entities": [
                (13, 29, "EVENT_NAME"),  # Pick Up Dog Poop
                (46, 57, "DATE"),  # next friday
                (61, 64, "TIME"),  # 9am
            ]
        },
    ),
    (
        "add an event does stuff for next friday at 9am",
        {
            "entities": [
                (13, 23, "EVENT_NAME"),  # does stuff
                (28, 39, "DATE"),  # next friday
                (43, 46, "TIME"),  # 9am
            ]
        },
    ),
    (
        "add a meeting Product Dev Recap with bill@microsoft.com for June 23 at 1pm",
        {
            "entities": [
                (14, 31, "EVENT_NAME"),  # Product Dev Recap
                (37, 55, "ATTENDEE"),  # bill@microsoft.com
                (60, 67, "DATE"),  # June 23
                (71, 74, "TIME"),  # 1pm
            ]
        },
    ),
    (
        "schedule a call Web Development Update with jane@example.com on April 5 at 2pm",
        {
            "entities": [
                (16, 38, "EVENT_NAME"),  # Web Development Update
                (44, 60, "ATTENDEE"),  # jane@example.com
                (64, 71, "DATE"),  # April 5
                (75, 78, "TIME"),  # 2pm
            ]
        },
    ),
    (
        "organize an event Team Building Activity next monday at 11am",
        {
            "entities": [
                (18, 40, "EVENT_NAME"),  # Team Building Activity
                (41, 52, "DATE"),  # next monday
                (56, 60, "TIME"),  # 11am
            ]
        },
    ),
    (
        "set up a meeting Product Review with alex@company.com on March 12 at 3pm",
        {
            "entities": [
                (17, 31, "EVENT_NAME"),  # Product Review
                (37, 53, "ATTENDEE"),  # alex@company.com
                (57, 65, "DATE"),  # March 12
                (69, 72, "TIME"),  # 3pm
            ]
        },
    ),
    (
        "Set up a team meeting for Project Launch Discussion on July 10th at 10am",
        {
            "entities": [
                (26, 51, "EVENT_NAME"),  #  Project Launch Discussion
                (55, 64, "DATE"),  # July 10th
                (68, 72, "TIME"),  # 10am
            ]
        },
    ),
    (
        "Create an event for Office Team Building on June 5th at 4pm",
        {
            "entities": [
                (20, 40, "EVENT_NAME"),
                (44, 52, "DATE"),
                (56, 59, "TIME"),
            ]
        },
    ),
    (
        "Schedule a doctor's appointment for Annual Health Check on April 22nd at 2pm",
        {
            "entities": [
                (36, 55, "EVENT_NAME"),
                (59, 69, "DATE"),
                (73, 76, "TIME"),
            ]
        },
    ),
    (
        "Plan a coffee catch-up with emma@workplace.com next Friday at 11am",
        {
            "entities": [
                (7, 19, "EVENT_NAME"),
                (28, 46, "ATTENDEE"),
                (47, 58, "DATE"),
                (62, 66, "TIME"),
            ]
        },
    ),
    (
        "Add an event called Talk To Sales team with billy@saascompanyco.com for next wednesday at 10:30 am",
        {
            "entities": [
                (20, 38, "EVENT_NAME"),
                (44, 67, "ATTENDEE"),
                (72, 86, "DATE"),
                (90, 98, "TIME"),
            ]
        },
    ),
    (
        "Create a meeting Team Discussion for December 24th at 9pm",
        {
            "entities": [
                (17, 32, "EVENT_NAME"),
                (37, 50, "DATE"),
                (54, 57, "TIME"),
            ]
        },
    ),
    (
        "Add a meeting Discuss 2024 Budget next friday at 9am with user@companyco.com",
        {
            "entities": [
                (14, 33, "EVENT_NAME"),
                (34, 45, "DATE"),
                (49, 52, "TIME"),
                (58, 76, "ATTENDEE"),
            ]
        },
    ),
    (
        "Create a workshop Customer Training Session on September 15th at 3:30pm",
        {
            "entities": [
                (18, 43, "EVENT_NAME"),  # Customer Training Session
                (47, 61, "DATE"),  # September 15th
                (65, 71, "TIME"),  # 3:30pm
            ]
        },
    ),
    (
        "Organize a seminar Sales Strategy Discussion with partners@company.com on August 10 at 2:45 pm",
        {
            "entities": [
                (19, 44, "EVENT_NAME"),  # Sales Strategy Discussion
                (50, 70, "ATTENDEE"),  # partners@company.com
                (74, 83, "DATE"),  # August 10
                (87, 94, "TIME"),  # 2:45 pm
            ]
        },
    ),
    (
        "Schedule a meeting Marketing Campaign Review for next Thursday at 10:00 am",
        {
            "entities": [
                (19, 44, "EVENT_NAME"),  # Marketing Campaign Review
                (49, 62, "DATE"),  # next Thursday
                (66, 74, "TIME"),  # 10:00 am
            ]
        },
    ),
    (
        "Add a conference call Project Status Update with team@company.com on May 20 at 1:15pm",
        {
            "entities": [
                (22, 43, "EVENT_NAME"),  # Project Status Update
                (49, 65, "ATTENDEE"),  # team@company.com
                (69, 75, "DATE"),  # May 20
                (79, 85, "TIME"),  # 1:15pm
            ]
        },
    ),
    (
        "Set up a board meeting Quarterly Review for October 2 at 4:30 pm",
        {
            "entities": [
                (23, 39, "EVENT_NAME"),  # Quarterly Review
                (44, 53, "DATE"),  # October 2
                (57, 64, "TIME"),  # 4:30 pm
            ]
        },
    ),
    (
        "Plan a team-building event Outdoor Adventure on July 17 at 11:00am",
        {
            "entities": [
                (27, 44, "EVENT_NAME"),  # Outdoor Adventure
                (48, 55, "DATE"),  # July 17
                (59, 66, "TIME"),  # 11:00am
            ]
        },
    ),
    (
        "Create a conference Marketing Summit with marketingteam@company.com for June 8 at 9:30am",
        {
            "entities": [
                (20, 36, "EVENT_NAME"),  # Marketing Summit
                (42, 67, "ATTENDEE"),  # marketingteam@company.com
                (72, 78, "DATE"),  # June 8
                (82, 88, "TIME"),  # 9:30am
            ]
        },
    ),
    (
        "Add a presentation Product Launch for November 5th at 2:00 pm",
        {
            "entities": [
                (19, 33, "EVENT_NAME"),  # Product Launch
                (38, 50, "DATE"),  # November 5th
                (54, 61, "TIME"),  # 2:00 pm
            ]
        },
    ),
    (
        "Schedule a meeting Client Demo with prospects@company.com on April 30 at 3pm",
        {
            "entities": [
                (19, 30, "EVENT_NAME"),  # Client Demo
                (36, 57, "ATTENDEE"),  # prospects@company.com
                (61, 69, "DATE"),  # April 30
                (73, 76, "TIME"),  # 3pm
            ]
        },
    ),
    (
        "Plan an off-site retreat Company Strategy Planning on August 25th at 10:30am",
        {
            "entities": [
                (25, 50, "EVENT_NAME"),  # Company Strategy Planning
                (54, 65, "DATE"),  # August 25th
                (69, 76, "TIME"),  # 10:30am
            ]
        },
    ),
    (
        "Organize a webinar Marketing Trends 2024 on July 5 at 1:00pm",
        {
            "entities": [
                (19, 40, "EVENT_NAME"),  # Marketing Trends 2024
                (44, 50, "DATE"),  # July 5
                (54, 60, "TIME"),  # 1:00pm
            ]
        },
    ),
]


# Load a blank English model
nlp = spacy.blank("en")

# for text, annotations in TRAIN_DATA:
#     doc = nlp.make_doc(text)
#     biluo_tags = offsets_to_biluo_tags(doc, annotations.get("entities"))
#     print("Text:", text)
#     print("BILUO Tags:", biluo_tags)

# Create the NER pipeline with the updated labels
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add labels to the NER pipeline
ner.add_label("EVENT_NAME")
ner.add_label("DATE")
ner.add_label("TIME")
ner.add_label("ATTENDEE")

with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != "ner"]):
    optimizer = nlp.begin_training()
    for itn in range(20):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses)
        print(f"Iteration {itn}, Losses: {losses}")

# Test the trained model
for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

# Save the model
nlp.to_disk("models/parser")



