# Diabetes_readmission_prediction
## This is a simple ML classification problem. The objective is to use historical patient data to predict the likelihood of readmission. The doctor intends to use this as an assessment tool before he discharges his patient. The data analyst has already did the analysis and develop with a predictive model.
## Deploy this -
###
### 1. Batch mode - 
#### a. Apply the model on a daily csv file
#### b. Get the readmission prob score and 
#### c. load the dataset (daily csv file + Prob Score) into a database table in the hospital operational data store (ODS).
#### d. Frontend applications will be interface with the ODS to display the predicted result.
### 2. Interactive mode - 
#### a. Create an api that can accept the inputs for the prediction model
#### b. Imagine that there is a frontend web application that will take in these parameters and upon submission, the readmission probability score will be returned to a result placeholder.       
