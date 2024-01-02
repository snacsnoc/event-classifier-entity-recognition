import spacy
from spacy.training.example import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
import random

TRAIN_DATA = [
    (
        "Create an event for Office Team Building on June 5th at 4pm",
        {
            "entities": [
                (20, 40, "EVENT_NAME"),  # Office Team Building
                (44, 52, "DATE"),  # June 5th
                (56, 59, "TIME"),  # 4pm
            ]
        },
    ),
    (
        "Add a meeting Discuss 2024 Budget next friday at 9am with user@companyco.com",
        {
            "entities": [
                (14, 45, "EVENT_NAME"),  # Discuss 2024 Budget next friday
                (49, 52, "TIME"),  # 9am
                (58, 76, "ATTENDEE"),  # user@companyco.com
            ]
        },
    ),
    (
        "Plan a coffee catch-up with emma@workplace.com next Friday at 11am",
        {
            "entities": [
                (7, 46, "EVENT_NAME"),  # coffee catch-up with emma@workplace.com
                (47, 58, "DATE"),  # next Friday
                (62, 66, "TIME"),  # 11am
            ]
        },
    ),
    (
        "Set up a team meeting for Project Launch Discussion on July 10th at 10am",
        {
            "entities": [
                (26, 51, "EVENT_NAME"),  # Project Launch Discussion
                (55, 64, "DATE"),  # July 10th
                (68, 72, "TIME"),  # 10am
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
        "Add an event called Talk To Sales team with billy@saascompanyco.com for next wednesday at 10:30 am",
        {
            "entities": [
                (20, 38, "EVENT_NAME"),  # Talk To Sales team
                (44, 67, "ATTENDEE"),  # billy@saascompanyco.com
                (72, 86, "DATE"),  # next wednesday
                (90, 98, "TIME"),  # 10:30 am
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
        "Plan a meeting Marketing Campaign Review with marketingteam@company.com on September 5 at 3:30 pm",
        {
            "entities": [
                (15, 40, "EVENT_NAME"),  # Marketing Campaign Review
                (46, 71, "ATTENDEE"),  # marketingteam@company.com
                (75, 86, "DATE"),  # September 5
                (90, 97, "TIME"),  # 3:30 pm
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
        "Schedule a doctor's appointment for Annual Health Check on April 22nd at 2pm",
        {
            "entities": [
                (36, 55, "EVENT_NAME"),  # Annual Health Check
                (59, 69, "DATE"),  # April 22nd
                (73, 76, "TIME"),  # 2pm
            ]
        },
    ),
    (
        "Plan an event Team Building Retreat on July 15th at 3pm",
        {
            "entities": [
                (14, 35, "EVENT_NAME"),  # Team Building Retreat
                (39, 48, "DATE"),  # July 15th
                (52, 55, "TIME"),  # 3pm
            ]
        },
    ),
    (
        "Schedule a team meeting Project Kickoff on July 1st at 10:15am",
        {
            "entities": [
                (24, 39, "EVENT_NAME"),  # Project Kickoff
                (43, 51, "DATE"),  # July 1st
                (55, 62, "TIME"),  # 10:15am
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
        "Add an event Team Building Workshop on May 15th at 2:30 pm",
        {
            "entities": [
                (13, 35, "EVENT_NAME"),  # Team Building Workshop
                (39, 47, "DATE"),  # May 15th
                (51, 58, "TIME"),  # 2:30 pm
            ]
        },
    ),
    (
        "Create a meeting Team Discussion for December 24th at 9pm",
        {
            "entities": [
                (17, 32, "EVENT_NAME"),  # Team Discussion
                (37, 50, "DATE"),  # December 24th
                (54, 57, "TIME"),  # 9pm
            ]
        },
    ),
    (
        "Schedule a lunch meeting with the marketing team at Jack's Cafe on February 18th at 1pm.",
        {
            "entities": [
                (11, 63, "EVENT_NAME"),  # lunch meeting with the marketing team at Jack's Cafe
                (67, 80, "DATE"),  # February 18th
                (84, 87, "TIME"),  # 1pm
            ]
        },
    ),
    (
        "Organize a webinar titled 'Future of AI in Business' with guest speakers john@tech.com and sara@innovate.com for March 3rd at 11am.",
        {
            "entities": [
                (11, 51, "EVENT_NAME"),  # webinar titled 'Future of AI in Business'
                (73, 86, "ATTENDEE"),  # john@tech.com
                (91, 108, "ATTENDEE"),  # sara@innovate.com
                (113, 122, "DATE"),  # March 3rd
                (126, 130, "TIME"),  # 11am
            ]
        },
    ),
    (
        "Set up a birthday party for Emma at The Green Park on April 10th at 2pm, invitees include emma@family.com, jack@friends.com, lisa@friends.com.",
        {
            "entities": [
                (9, 50, "EVENT_NAME"),  # birthday party for Emma at The Green Park
                (90, 105, "ATTENDEE"),  # emma@family.com
                (107, 123, "ATTENDEE"),  # jack@friends.com
                (125, 141, "ATTENDEE"),  # lisa@friends.com
                (54, 64, "DATE"),  # April 10th
                (68, 71, "TIME"),  # 2pm
            ]
        },
    ),
    (
        "Create a project debrief session for the XYZ project with project-team@ourcompany.com on January 22nd at 4pm.",
        {
            "entities": [
                (9, 52, "EVENT_NAME"),  # project debrief session for the XYZ project
                (58, 85, "ATTENDEE"),  # project-team@ourcompany.com
                (89, 101, "DATE"),  # January 22nd
                (105, 108, "TIME"),  # 4pm
            ]
        },
    ),

    (
        "Plan a team building exercise at the downtown gym for all department heads on May 5th at 3:30pm.",
        {
            "entities": [
                (7, 74, "EVENT_NAME"),  # team building exercise at the downtown gym for all department heads
                (78, 85, "DATE"),  # May 5th
                (89, 95, "TIME"),  # 3:30pm
            ]
        },
    ),
    (
        "Arrange a networking dinner with potential clients at the Harbor Restaurant on June 20th at 7pm, attendees include client1@business.com, client2@enterprise.com.",
        {
            "entities": [
                (10, 75, "EVENT_NAME"),  # networking dinner with potential clients at the Harbor Restaurant
                (115, 135, "ATTENDEE"),  # client1@business.com
                (137, 159, "ATTENDEE"),  # client2@enterprise.com
                (79, 88, "DATE"),  # June 20th
                (92, 95, "TIME"),  # 7pm
            ]
        },
    ),
    (
        "Book a conference room for the Annual Sales Meeting on September 15th from 9am to 5pm.",
        {
            "entities": [
                (7, 51, "EVENT_NAME"),  # conference room for the Annual Sales Meeting
                (55, 69, "DATE"),  # September 15th
                (70, 78, "TIME"),  # from 9am to 5pm
            ]
        },
    ),
    (
        "Schedule a follow-up appointment with Dr. Smith at the Wellness Center on July 23rd at 10am.",
        {
            "entities": [
                (11, 70, "EVENT_NAME"),  # follow-up appointment with Dr. Smith at the Wellness Center
                (74, 83, "DATE"),  # July 23rd
                (87, 91, "TIME"),  # 10am
            ]
        },
    ),
    (
        "Organize a virtual training session on new software tools with trainer@techco.com on August 1st at 4pm.",
        {
            "entities": [
                (11, 57, "EVENT_NAME"),  # virtual training session on new software tools
                (63, 81, "ATTENDEE"),  # trainer@techco.com
                (85, 95, "DATE"),  # August 1st
                (99, 102, "TIME"),  # 4pm
            ]
        },
    ),
    (
        "Plan a coffee meet-up with old colleagues at Central Café on October 12th at 11am, invitees are mark@oldjob.com, anna@pastwork.com.",
        {
            "entities": [
                (7, 57, "EVENT_NAME"),  # coffee meet-up with old colleagues at Central Café
                (96, 111, "ATTENDEE"),  # mark@oldjob.com
                (113, 130, "ATTENDEE"),  # anna@pastwork.com
                (61, 73, "DATE"),  # October 12th
                (77, 81, "TIME"),  # 11am
            ]
        },
    ),
    (
        "Set up a review meeting for the budget report with finance-team@company.com on December 6th at 2pm.",
        {
            "entities": [
                (9, 45, "EVENT_NAME"),  # review meeting for the budget report
                (51, 75, "ATTENDEE"),  # finance-team@company.com
                (79, 91, "DATE"),  # December 6th
                (95, 98, "TIME"),  # 2pm
            ]
        },
    ),
    (
        "Arrange a workshop titled 'Leadership Skills in the 21st Century' with guest speaker expert@leadership.com on November 19th at 1pm.",
        {
            "entities": [
                (10, 64, "EVENT_NAME"),  # workshop titled 'Leadership Skills in the 21st Century'
                (85, 106, "ATTENDEE"),  # expert@leadership.com
                (110, 123, "DATE"),  # November 19th
                (127, 130, "TIME"),  # 1pm
            ]
        },
    ),
    (
        "Book a team outing to the Tech Museum for the engineering department on March 29th at 10am.",
        {
            "entities": [
                (7, 68, "EVENT_NAME"),  # team outing to the Tech Museum for the engineering department
                (72, 82, "DATE"),  # March 29th
                (86, 90, "TIME"),  # 10am
            ]
        },
    ),
    (
        "Organize a graduation party for John at Lakeside Hall on June 22nd at 6pm, guests include family@johnsfamily.com, friends@johnsfriends.com.",
        {
            "entities": [
                (11, 53, "EVENT_NAME"),  # graduation party for John at Lakeside Hall
                (90, 112, "ATTENDEE"),  # family@johnsfamily.com
                (114, 138, "ATTENDEE"),  # friends@johnsfriends.com
                (57, 66, "DATE"),  # June 22nd
                (70, 73, "TIME"),  # 6pm
            ]
        },
    ),
    (
        "Create a meeting to discuss the client feedback report with management@ourfirm.com on April 8th at 3pm.",
        {
            "entities": [
                (9, 54, "EVENT_NAME"),  # meeting to discuss the client feedback report
                (60, 82, "ATTENDEE"),  # management@ourfirm.com
                (86, 95, "DATE"),  # April 8th
                (99, 102, "TIME"),  # 3pm
            ]
        },
    ),
    (
        "Schedule a volunteer day for the community service team at the City Park on May 16th at 9am.",
        {
            "entities": [
                (11, 72, "EVENT_NAME"),  # volunteer day for the community service team at the City Park
                (76, 84, "DATE"),  # May 16th
                (88, 91, "TIME"),  # 9am
            ]
        },
    ),
    (
        "Plan a farewell lunch for retiring employees at The Heritage Restaurant on July 31st at 12pm.",
        {
            "entities": [
                (7, 71, "EVENT_NAME"),  # farewell lunch for retiring employees at The Heritage Restaurant
                (75, 84, "DATE"),  # July 31st
                (88, 92, "TIME"),  # 12pm
            ]
        },
    ),
    (
        "Set up a brainstorming session for the new advertising campaign with creative@adagency.com on August 14th at 2:30pm.",
        {
            "entities": [
                (9, 63, "EVENT_NAME"),  # brainstorming session for the new advertising campaign
                (69, 90, "ATTENDEE"),  # creative@adagency.com
                (94, 105, "DATE"),  # August 14th
                (109, 115, "TIME"),  # 2:30pm
            ]
        },
    ),
    (
        "Book a seminar on digital marketing trends with experts from industry@digitalmarketing.com and insights@techworld.com on October 25th at 10am.",
        {
            "entities": [
                (7, 42, "EVENT_NAME"),  # seminar on digital marketing trends
                (61, 90, "ATTENDEE"),  # industry@digitalmarketing.com
                (95, 117, "ATTENDEE"),  # insights@techworld.com
                (121, 133, "DATE"),  # October 25th
                (137, 141, "TIME"),  # 10am
            ]
        },
    ),
    (
        "Organize an online discussion on global health with panelists doctor@health.org, researcher@science.net on September 21st at 8pm.",
        {
            "entities": [
                (12, 46, "EVENT_NAME"),  # online discussion on global health
                (62, 79, "ATTENDEE"),  # doctor@health.org
                (81, 103, "ATTENDEE"),  # researcher@science.net
                (107, 121, "DATE"),  # September 21st
                (125, 128, "TIME"),  # 8pm
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
    for itn in range(100):
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



