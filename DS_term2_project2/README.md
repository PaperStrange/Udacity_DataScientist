### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Before running this code, make sure by using Python versions 3.*, python packages including numpy, pandas, sklearn, matplotlib and seaborn are installed properly.

<!--Meanwhile, to use model [LightBGM](https://github.com/Microsoft/LightGBM), use pip to install it as follows.-->
<!--```-->
<!--pip install lightgbm-->
<!--```-->

After above installation, the code should run with no issues.

## Project Motivation<a name="motivation"></a>

For this project, to better understand the below several questions:

1. (tag: Tech) How to deal with different expressions which share the same meaning to compress datasets? Word embedding? N-gram model?

2. (tag: Tech) How to visualize the comprehension of model?

3. (tag: Tech) How to measure the ability of model generalization?

4. (tag: Business) This model is suitable for Figure Eight, show me the reasons :)

5. (tag: Business) If applied, what is needed to make sure the  normal operation of this model? How to evaluate the cost of potential necessary changes occured in company like staff structure, financial?

6. (tag: Business) Draw a data transportation map(like from client to host, from host to company staff), find out the most time-consuming transportation line and try to optimize

## File Descriptions <a name="files"></a>

In total, the structure of this project files is shown below:

Name | Description |
------------ | -------------
app/template/master.html | main page of web app
app/template/go.html | classification result page of web app
app/run.py | Flask file that runs app
-- | --
data/disaster_categories.csv | data to process
data/disaster_messages.csv | --
data/process_data.py | data to process
data/InsertDatabaseName.db | database to save clean data to
-- | --
models/train_classifier.py | --
models/classifier.pkl | saved model
-- | --
.gitignore | --
LICENSE | --
README.md | --


In detail, there is . Meanwhile, data are stored in "data" folder.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results<a name="results"></a>

<!--The main findings of the code can be found at the post available [here](https://PaperStrange.github.io/).-->

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

This repository is distributed under the GNU license.

Must give credit to kaggle for the data. Anyway, feel free to use the code here as you would like!

