### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Before running this code, make sure by using Python versions 3.*, python packages including numpy, pandas, sklearn, matplotlib and seaborn are installed properly.

Meanwhile, to use model [LightBGM](https://github.com/Microsoft/LightGBM), use pip to install it as follows.
```
pip install lightgbm
```

After above installation, the code should run with no issues.

## Project Motivation<a name="motivation"></a>

For this project, I was interested in using the data oriented from a kaggle competition [Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation) to better understand the below several questions:

1. What scale and what kind of data are necessary for customer loyalty prediction model?

2. In which way the application of customer loyalty predicition model would impact different business units?

3. How to evaluate the influence of deploying customer loyalty prediction model? Is modeling target really convincible?

4. How to balance model prediction and customer review?

5. What could customers expect to be benefited from this customer loyalty prediction model?

## File Descriptions <a name="files"></a>

There is 1 notebook available here to showcase work related to the above questions which is exploratory in following procedures of `CRISP-DM process` to the questions showcased by this notebook title. Markdown cells were used to assist in walking through the thought process for individual steps.

Data are stored in "data" folder while some data files such as merchant data, transaction data are too large to upload. However all data are available for freely download in the competition page. Except for training and testing data stored respectively in "train" and "test" folder, a `.xlsx` file is provided for storing authority descriptions of all data files.

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://PaperStrange.github.io/).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

This repository is distributed under the GNU license.

Must give credit to kaggle for the data. Anyway, feel free to use the code here as you would like!

