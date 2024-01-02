# Event Scheduler Classifier and NER Model

This repository contains code for an Event Scheduler and Named Entity Recognition (NER) model built using Python's NLTK and spaCy libraries. The project consists of two main components:

1. **Event Scheduler**: A simple event scheduling and classification system that allows users to schedule events, meetings, and appointments using natural language commands. The NLTK library is used to train a Naive Bayes Classifier to classify user input into different scheduling commands.

2. **NER Model**: A Named Entity Recognition model created using spaCy. This model identifies and extracts structured information from text, such as event names, dates, times, and attendees. The NER model is trained on a custom dataset.

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- NLTK library 
- spaCy library 

### Installation

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage
__Event Scheduler__

The event scheduler uses a trained Naive Bayes Classifier to classify user input into scheduling commands. 
To use it, run the `train_classifier.py` script. 

The model saves into `model/classifier.pkl`

__NER Model__
The NER model is trained to extract entities such as event names, dates, times, and attendees from text. To use it, follow the instructions below:

1. Train the NER model:

The NER model is trained on a custom dataset. You can find the training data and code in the `train_parser.py` file.

2. Save the trained NER model:

After training, save the NER model using `nlp.to_disk("myapp/nlp_models/parser")`.

3. Test the model:

You can test the trained NER model using the `test_ner_model.py` script. Provide new text inputs to see how the model extracts entities.