# Introduction

This project is origined from
a nearby kaggle competition [Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation)

# Business & Data Understanding

## Questions 1

**1.** Provide the number of rows and columns in this datasets, find each data type of this datasets columns

**2.** Provide the missing value distribution of this datasets, find out which columns have the most missing values and provide a set of column names that have more than 75% if their values missing if any

**3.** Check duplication of this datasets along with "card_id" column feat

**4.** Collect description of each column of this datasets if provided

**5.** Provide column feat "target" value distribution, find out if any outlier exist

**6.** Check out the relationship between column feat "target" and rest column feats of training dataset

## Answers 1

**1.** Seeing the result of assignment to function "check_basically" in the notebook, training dataset has 201917 rows and 6 columns, testing dataset has 123623 rows and 5 columns. Furthermore, among training dataset, and testing dataset, apart from two categorical column feat "first_active_month", "card_id", the rest column feats are all stored in numberical type.

**2.** Seeing the result of assignment to function "check_completion" in the notebook, training dataset has no vacancy while testing dataset has less than 0.1% vacancy at column feat "first_active_month". Thus there is no vacancy with its proportion larger than 75%.

**3.** Seeing the result of assignment to function "chech_repeatablity" in the notebook, there is no repeatablity in two datasets.

**4.** Seeing the result of assignment to function "collect_datadescription" in the notebook.

**5.** Seeing the result of assignment to function "check_target_distribution" in the notebook, meanwhile there is 2207 outliers found

**6.** Seeing the result in "linear relationship checking" part in the notebook, it seems that no linear relationship exists. 

## Questions 2

**1.** According to importance distribution, which column feat contributes most to the model prediction? which column feat contribute least?

**2.** According to importance distribution, by ranging importance of column feat extracted from date data, which extraction wins and which one lose?

**3.** By comparing training dataset and testing dataset, do the column feats found in question2.1 and question 2.2 show different distribution?

**4.** In the next step, to add more feats or to tune the model or change the model? Why?

**5.** If you choose to change the model, why and why not?

## Answers 2

**1.** "first_active_month_elapsed_time_today" contributes most, while "first_active_month_monthsession_January" contributes least. 

**2.** Among the feat extracted from date data, "first_active_month_elapsed_time_today" gains the most importance while the "first_active_month_monthsession_January" gain the least importance.

**3.** Seeing the "analysis between training data and testing data" part in the notebook, "first_active_month_elapsed_time_today", "first_active_month_elapsed_time_specific" shows measurable different distribution between training data and testing data, among which the first one is the feature picked by the above two questions. On the onther hand, the left feats couldn't provide considerable difference on data distribution between two datasets. 

**4.** In the next step, adding more feats is more preferable. Th reason is that up to now this notebook only use the little obscure feats provided by two datasets. This competition also provides other dataset like transaction records which contains more than 100w data. Meanwhile, now we just use raw data with little data engineering work to obtain a baseline score for later comparison.

**5.** By checking the Lightgbm github [blog](https://github.com/Microsoft/LightGBM), it is learned that this model is truly suitable for this prediction task mainly because of its optimization in speed and optimal split for categorical features which is really helpful for dealing with dataset with more useful information containing in categorical features.

# License

This repository is distributed under the GNU license.
