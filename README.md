# Message Classifer for Disaster Events

End-2-end Machine Learning [system for message classification](https://github.com/leanguardia/disaster-message-classifier) and help in the organization of aid teams in
real time. This system classifies text message with none, one or more classes like `request`,
`offer`, `medical_help`, `water`, `food`, `fire`, and others.

### Dependencies
- Python 3
- Pandas
- Numpy
- Scikit-learn
- Nltk
- Flask
- SQLite

### Setup
Run the following commands in the project's root directory to set up your database and model.

#### 1. ETL Pipeline
It cleans data and stores in database

`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

#### 2. ML Pipeline
It trains the classifier and saves it in a pickle file

`python models/train_classifier.py data/DisasterResponse.db models/message-cls.pkl`

### Web app
Once the previous tasts were successfully executed, run the following command to start the server

`python app/run.py`

Then go to http://0.0.0.0:3001/ in your browser.

#### Pages

- **Home:** allows to enter text messages for classification. It also renders three plots to
  visualize some characteristics of the dataset used for the classifier's training.

  ![Message Category Distribution]('app/message_distribution.png')
  
- **Query:** after a message has been submited in the home page, this view shows the
  predicted classes the text belongs to.

### Files

- `app/run.py` definition of the web endpoints and web application setup
- `data/process_data.py` implementation of the ETL pipeline
- `models/train_classifier.py` implementation of the Machine Learning pipeline.

### Acknowledgements
The "Data Science" Nanodegree at Udacity, the Scikit-learn community and StackOverflow.

### Licence
MIT License | Copyright (c) 2020 D. Leandro Guardia V.