### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions for use](#instructions)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Before running this code, make sure by using Python versions 3.*, python packages including numpy, pandas, sklearn, matplotlib and seaborn are installed properly.

Then, to apply model to a website, use pip to install flask as follows.
```
pip install flask
```

As for the online visualization, plotly is recommended here: (make sure keras is installed first)
```
pip install plotly
```

Meanwhile, to use text characterization method from nltk, use pip to install it as follows:
```
pip install nltk
```

After above installation, the code should run with no issues.

## Project Motivation<a name="motivation"></a>

For this project, the below several questions are expected to be dived into:

##### [Q1 (tag: Tech)](#Q1)
How to deal with the imbalanced data for some labels? How that affects training the model, and your thoughts about emphasizing precision or recall for the various categories.

##### [Q2 (tag: Tech)](#Q2)
How to improve the efficiency of the code in the ETL and ML pipeline?

##### [Q3 (tag: Tech)](#Q3)
How to measure the ability of model generalization?

##### [Q4 (tag: Business)](#Q4)
If applied, what is needed to make sure the normal operation of this model? How to evaluate the cost of potential necessary changes occurred in company like staff structure, financial?

##### [Q5 (tag: Business)](#Q5)
Based on the categories that the ML algorithm classifies text into, how to advise some organizations to connect to?

##### [Q6 (tag: Business)](#Q6)
Assume all deploy is done, when tracing the data transportation map(like from client to host, from host to company staff), how to find out the most time-consuming transportation line and optimize?

## File Descriptions <a name="files"></a>

In total, the structure of this project files is shown below:

Name | Description |
------------ | -------------
app/template/master.html | main page of web app
app/template/go.html | classification result page of web app
app/run.py | Flask file that runs app
-- | --
data/process_data.py | script to execute data processing codes
data/InsertDatabaseName.db | database to save clean data to
data/weights.csv | csv data to save all weights for each data
-- | --
graph | all graphs in README file are stored here
-- | --
models/train_classifier.py | generate classifier model
models/train_classifier.ipynb | jupyter notebook containing under-testing classifier codes using LightGBM model (Fail to combine with sklearn Pipeline, `see issue #4`)
models/classifier.pkl | saved model
-- | --
.gitignore | --
LICENSE | --
README.md | --

Data are stored in "data" folder. Thoungh data will not be provided here while you can find more in the figure-eight website [here](https://www.figure-eight.com/)

## Instructions for use<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/, enjoy it!

## Results<a name="results"></a>

<!--<img src="https://raw.githubusercontent.com/PaperStrange/Udacity_records/DS_term2/DS_term2_project2/graphs/WebDisplay.png" width = "300" align=center />

**<center>Fig1: Web Display</center>**-->

<!--<img src="https://raw.githubusercontent.com/PaperStrange/Udacity_records/DS_term2/DS_term2_project2/graphs/model_acc.png" width = "300" align=center />

**<center>Fig2: Accuracy</center>**-->

The Fig1 is a screenshot of my web app when inputing a message, the Fig2 is the accuracy of my model on each label. Main findings of the code are listed to reply the above six questions respectively and will be posted in my blog [here](https://PaperStrange.github.io/) later.

##### A1<a name="Q1"></a>

<!--<img src="https://raw.githubusercontent.com/PaperStrange/Udacity_records/DS_term2/DS_term2_project2/graphs/A1_1.png" width = "300" align=center />

**<center>Fig3: amount distribution of label "offer"</center>**-->

<!--<img src="https://raw.githubusercontent.com/PaperStrange/Udacity_records/DS_term2/DS_term2_project2/graphs/A1_2.png" width = "300" align=center />

**<center>Fig4: several scores of label "offer"</center>**-->

From the Fig3, it easy to see the data distribution of "offer" label. This label have little amount of data in class "1" while large amount of data in class "0". These imbalance distributions truly bring precision nearly 100% but lead to an obvious differences in recall score shown in Fig4 (The recall score in one class will be much less than the other class). Thus, the trained model has a great chance to develop a strong dependent on one class and ignorance on the other class and generalize badly in the future.

The three well-know methods to deal with the imbalanced includes under-sampling, over-sampling, and generating synthetic data. Because the relative small amount of data(26216), undersampling may lead to information leak and oversampling may lead to too much noise. Generating synthetic data is not not so suitable because of the data origin: twitter message, which means synthetic data is not so easy to be generated and evaluated. As a result, for this case, i think, expanding features by adding statistical standards for the matrix of token counts (using class named as "StatisticalAnalysis" in "train_classifier.py") is preferable to balance data to some extent.

##### A2<a name="Q2"></a>
Cause not so familiar with sklearn Pipeline, i refer to the help of google. One interesting solution is a blog introducing a new pipeline imported from dasklearn. The accelerate result as ten times faster is really considerable. However, the author said it is a trade-off: less computation is needed while exposing a parallel map are no longer sufficient for the training. More details are listed in his blog [here](http://blaze.pydata.org/blog/2015/10/19/dask-learn/).

##### A3<a name="Q3"></a>
In my opinion, for each label, the less difference between recall scores of classes, the less bias will be induced to the model prediction of this label, which means better generalization ability(PS: if less sensitive to complexity of evaluation, confusion matrix  or self-designed loss function may be more suitable). Another  possible metrics, i think, is using auc score to measure the separability between training dataset and testing dataset. More details could refer a blog [here](https://towardsdatascience.com/understan).

##### A4<a name="Q4"></a>
To assure convincible deploy, for technology, logging and test codes are supposed to be maintained regularly; for other non-technical staff, a instruction list including rules of model input data type, model output data type, issues and relative solutions etc is preferable(inspired by [*Work Rules! Insights from Inside Google That Will Transform How You Live and Lead*](https://www.amazon.com/Work-Rules-Insights-Inside-Transform-ebook/dp/B00MEMMVB8)). As for the cost evaluation, i think, could be derived from each cost analysis provided by associated department by sum or apply weight to highlight some costs.

##### A5<a name="Q5"></a>
Excepy for disaster agency of each government, there are several civil self-organization 
agencies more than the well-known Read Cross. More descriptions could refer to this blog [here](https://www.thoughtco.com/top-disaster-relief-organizations-701272). 

##### A6<a name="Q6"></a>
Data transportation could be divided by three nexus nodes: client/customer(use mobile, laptop etc.), host(cloud or real database part, model part, message part etc.) and developer(laptop). Easy to find that the most large loads are more likely to happen when too often instant messaging occurs between millions clients and host. As a result, a strong database, i think, is in larger demand and importance than a robust model.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

This repository is distributed under the GNU license.

Must give credit to kaggle for the data. Anyway, feel free to use the code here as you would like!

