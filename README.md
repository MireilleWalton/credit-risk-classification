# Module 12 Report

## Overview of the Analysis

The purpose of this project is to build and train a supervised machine learning model and then evaluate its affectiveness to analyse lending activity, assess loan risk and identify the creditworthiness indiviual borrowers.

A historial dataset has been provided which the below data fields: 
- loan size - dollar value of the amount loaned to an individual
- interest rate - the rate of interest applied to the loaned amount
- borrower income - the earning capacity of the individual taking out the loan
- debt to income ratio - (presumably a calculation of "loan_size" value and "borrower_income"
- number of accounts - the total number of bank accounts an individual has
- derogatory marks - (unknown - potentially based on credit rating agency records)
- total debt - the total value of debt attributable to the individual (presumably including credit card, personal loans, store cards etc)
- loan status - a binary value indicating whether or not a loan is in default

There are 77,536 data rows within the data set. 

The first step in building the model was to read the csv data file into a pandas dataframe using jupyter notebook. 

Next is to separate the data into labels and features: 'labels' being the data that the model is being developed to predict, and 'features' being the data upon which the model will base its predictions. To do this, two variables are are created ('y' for lables and 'X' for features), and the relevant fields/columns from the dataframe are assigned to each variable.  

The 'y' labels variable comprises the "loan_status" column; and the 'X' features variable comprises the remaining columns within the dataset, ie the 'loan-status'field is removed from to ensure the model does not incorporate the 'labels' data into the analysis, this is done using the pandas '.drop()' function.   

Then the SKLearn 'train_test_learn' module is used to split the data into two sets: one set is used to train the model (58,152 rows of data were seleted by the module for training); and the second set is used to test its effectiveness (19,384 rows were selected by the model for testing).

The LogisticRegression module from teh SKLearn library is then used to classify the data based on relationships or correlations that exist between the data points within the dataset (AWS, data unkown).  The 'model.score()' method provides an indication of how well the model was able to classify the data by assigning a score between 0 and 1.  In this case the training data received a score of 0.9915 and the testing data a score of 0.9924, indicating that the model was very effective at classifying the data with only a marginal difference between training and testing.  The LinearRegression model there for can be considered to be an effective model to use to assess loan risk and creditworthiness of individuals (Varoquaux, G, data unknown).   

To further evaluate the model's performance a confusion matrix is generated, as well as a classification report.  


## Results

#### Confusion Matrix:
  
  * True Positive = 18,679 - the number of loans correctly identified as "healthy loans". 
  * True Negative = 80 - the number of loans incorrectly identified as "healthy loans". 
  * False Positives = 67 - the number of loans incorrectly identified as "high-risk loans"
  * False Negatives = 558 - the number of loans correctly identified as "high risk"


#### Classificaiton Report:
  * SKLearn's classificaiton report provides a snapshot of how well the applied machine learning model has performed on the test data by providing percentage values for correct positive predictions for Precision and Recall, and a weighted F1 Score of between 0 and 1, where 1 indicates better performance (Zach 2022)
  * 
    ![Screenshot 2023-10-15 at 12 48 57 pm](https://github.com/VioletRogue12/credit-risk-classification/assets/130148039/cf89d3b9-0616-494b-a7eb-b1548902cad3)

An explanation of the different categories within the above credit risk classification report is provided below:

##### "Precision: percentage of correct positive predictions relative to total positive predictions", (Zach, 2022)
The report shows that the linear regression model correctly predicted 100% of total Healthy Loans, whereas it correctly predicted 87% of total High-risk loans.   

##### "Recall: percentage of correct positive predictions relative to total actual positives" (ie true and false positives), (Zach, 2022)
The report shows that the linear regression model correctly predicted 100% of total true/false positive Healthy Loans, whereas it correctly predicted 89% of total true/false positive High-risk loans. 

##### "F1 Score: a weighted harmonic mean of precision and recall. The closer to 1, the better the model", (Zach, 2022)
The linear regression model received a ratinig of 1 for Healthy Loans 0.88 for High-risk loans. 

## Summary

Overall the linear regression model has performed very well in predicting the serviceability of loans.  The relevant organisation can feel confident in using the results to target individuals it may wish to offer additional benefits / incentives to garner loyalty from those customers; or to implement early intervention programs for individuals identified as being at high risk of defaulting on their loans.  



## Options for machine learning models
Choosing which machine learning model to use can be complex.  A data analyst may run one or more models against a data set in order to determine which provides the best results.  

Roger Huang (FreeCodeCamp,2019) sets out three variables to consider when selecting a machine learning model to use on a given project: These include:

- Unsupervised learning - requiring no initial processisng of the original data - vs supervised learning where some initial processing / cleaning of the data may be required to get it into a state where is it is ready to be used.  This may include re-labelling of columns, removing null values etc; 

- Space and time considerations - the processing time for the various machine learning algorythyms varies and may impact on performance and lead to a delay in output of results;

- The output - this is the form of the desired results.  Examples provided by R Haung inlcude categorised data or predicting future data points.  

Linear Regression models are useful for classification and prediction where the data source is well structured or has already been distinctly labelled - ie supervised machine learning (AWS, date unknown).    

For unsupervised machine learning where pre-defined categories and labels are not referenced, the K-means clustering algorhythm can be used to group and cluster data by measuring the distance between each point (LEDU, 2018). 

Other models to consider include:
- decision trees which show how data has been split when applying the classification process
- deep learning for unstructured and large scale data sets
- linear support vector which can be used for test classification. 


## References

Tommy Dang, 13 Oct 2022, updated 2 June 2023, Mage AI, "Guide to accuracy, precision, and recall", accessed 13 October 2023. 

AWS, Amazon Web Services, date unknown, "What is logistic regression?", https://aws.amazon.com/what-is/logistic-regression/#:~:text=Logistic%20regression%20is%20a%20data,outcomes%2C%20like%20yes%20or%20no, accessed 15 October 2023. 

Gael Varoquaux, date unknown, Scipy Lecture Notes, "3.6 scikit-learn: machine learning in Python", http://scipy-lectures.org/packages/scikit-learn/index.html, accessed 15 October 2023.

Roger Huang, 6 February 2019, FreeCodeCamp, "When to use different machine learning algorithms: a simple guid", https://www.freecodecamp.org/news/when-to-use-different-machine-learning-algorithms-a-simple-guide-ba615b19fb3b/, 15 October 2023. 

Education Ecosystem (LEDU), 13 September 2018, "Understanding K-means Clustering in Machine Learning", https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1, accessed 15 October 2023.  

Zach, 9 May 2022, Statology, "How to interpret the Classification Report in sklearn (with example)", https://www.statology.org/sklearn-classification-report/, accessed 15 October 2023.

