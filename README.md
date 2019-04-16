# nd-proj-DisasterResponsePipeline
My Implementation of the Udacity Project Disaster Response Pipeline as part of the Nanodegree "Data Scientist"

## General

<!-- first line needs to stay here, otherwise the table is not rendered! -->
|  |  | 
| ------------- | ------------- |
| **Description** | The data set contains messages which have been send during disaster events. Each message was examined in 36 categories, multiple selection allowed. A Machine Learning Pipeline was created to analyse the events. For easier usage, a web app is provided where new messages can be analysed and visualisations are provided.|
| **Anaconda environment** | env_udacity |
| **Data Set** | messages which were send during disaster events and a categorisation into 36 categories for each message, multi selection allowed. Data can be found in folder *data*. |

## Instructions for the local machine
1. Run the following commands in the project's root directory (*src*) to set up your database and model.

    - To run the ETL pipeline that cleans data and stores in database

        `python data/process_data.py ../data/disaster_messages.csv ../data/disaster_categories.csv ../data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves

        `python models/train_classifier.py ../data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory (*src/app*) to run your web app.

    - `python run.py`

3. Open the app
    - Go to http://0.0.0.0:3001/



## Folder Structure
### data
The data needed for this project, separate into two csv-files and the data base wjere the cleaned data is stored.
- *disaster_categories.csv*: Contains the message Id and the categories
- *disaster_messages.csv*: Contains the message Id, the message translated to english, the original message and the genre
- *DisasterResponse.db*: Data base which contains a table "messages" with the cleaned data.

### notebooks
For analysing the data and testing posssible pipelines, notebooks were used.
- *1 - ETL Pipeline Preparation.ipynb*: Create a pipeline to clean the data and store it 
- *2 - ML Pipeline Preparation.ipynb*: Load the cleaned data, perform text processing, build a ml pipeline and optimize it with GridSearchCV

### src
The code ot the notebooks has been refactored into Python-Scripts.
#### src/app
Scripts for the visualation of the web app.

#### src/data
Processing and cleaning of the data set.
Storage Lcoation of the data base

#### src/models
Training of the classifier


## If the project should be put everything back to Udacity
[Project Workspace IDE](https://classroom.udacity.com/nanodegrees/nd025/parts/3f1cdf90-8133-4822-ba56-934933b6b4bb/modules/b46b8867-d211-4be9-88f9-2365a35874be/lessons/7a929d2c-6da9-49d4-9849-e725b8c6e7a2/concepts/94f3a9bf-52af-4c12-82e2-b6065716fa1f)

If the project should be returned to the udacity workspace, the following filestructure has to be rebuild:
- workspace
    - app
        - templates
            - go.html
            - master.html
        - run.py
    - data
        - DisasterResponse.db
        - disaster_categories.csv
        - disaster_messages.csv
        - process_data.py
    - models
        - classifier.pkl
        - train_classifier.py
    - README.md

In addition, all path specifications must be checked and adjusted.

**Terminal commands:**

(project's root directory: *workspace*)
- `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

(app's directory: *workspace/app*)
- `python run.py`

(open the app)
- Execute `env | grep WORK`  in the included terminal to extract Workspace Id and Workspace Domain
- Go to `http://WORKSPACEID-3001.WORKSPACEDOMAIN`
