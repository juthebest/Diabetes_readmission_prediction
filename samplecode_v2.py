# -*- coding: utf-8 -*-
"""
Description/background - 
This is a simple ML classification problem.
The objective is to use historical patient data to predict the likelihood of readmission.
The doctor intends to use this as an assessment tool before he discharges his patient.
The data analyst has already did the analysis and develop with a predictive model.

The next step is to deploy this so that the model can be utilised by the doctor.

After understanding the unlying mechanisms that drive the model from the data analyst and the deployment 
strategy from the doctors, there are 2 possible ways to deploy this -
1. Batch mode - a. Apply the model on a daily csv file
                b. Get the readmission prob score and 
                c. load the dataset (daily csv file + Prob Score) into a database table in the hospital 
                   operational data store (ODS).
                d. Frontend applications will be interface with the ODS to display the predicted result.
2. Interactive mode - a. Create an api that can accept the inputs for the prediction model
                      b. Imagine that there is a frontend web application that will take in these parameters and 
                         upon submission, the readmission probability score will be returned to a result placeholder.       

Tasks - 
1. You are to come up with a schematic diagram to illustrate end to end data pipeline.
   Labelled the key components and state their functions. 
   Suggest suitable table/s model and the use of them. Explain the reasons for the proposed design.
Output - 2 to 3 slides   
2. Use a simple API framework (i.e. Python/Flask), code a simple function that will wrap the developed model and return
   return the readmission probability score. The inputs to this will be the attributes used in the model.  
Output - Python code or any other programming language you think can perform this task. 

What are given - 
1. A sample script (samplecode_v2.py) that contains a mock up model development process and the 2 proposed deployment strategies
2. 2 datasets - 1 being used for model development (Diabetes_Dev.csv) and 
   the other a sample of daily file received to perform prediction (Diabetes_Prod.csv)
3. The saved logreg model (readm_finalized_pred_model_v1.sav)
"""

# CREATE TABLE READMISSION_DATA (
# ROWID INT,RACE VARCHAR(1024),GENDER VARCHAR(1024),AGE VARCHAR(1024),WEIGHT VARCHAR(1024),
# ADMISSION_TYPE_ID VARCHAR(1024),DISCHARGE_DISPOSITION_ID VARCHAR(1024),
# ADMISSION_SOURCE_ID VARCHAR(1024),TIME_IN_HOSPITAL INT,PAYER_CODE VARCHAR(1024),
# MEDICAL_SPECIALTY VARCHAR(1024),NUM_LAB_PROCEDURES INT,NUM_PROCEDURES INT,NUM_MEDICATIONS INT,
# NUMBER_OUTPATIENT INT,NUMBER_EMERGENCY INT,NUMBER_INPATIENT INT,DIAG_1 INT,DIAG_2 FLOAT,
# DIAG_3 INT,NUMBER_DIAGNOSES INT,MAX_GLU_SERUM VARCHAR(1024),A1CRESULT VARCHAR(1024),
# METFORMIN VARCHAR(1024),REPAGLINIDE VARCHAR(1024),NATEGLINIDE VARCHAR(1024),
# CHLORPROPAMIDE VARCHAR(1024),GLIMEPIRIDE VARCHAR(1024),ACETOHEXAMIDE VARCHAR(1024),
# GLIPIZIDE VARCHAR(1024),GLYBURIDE VARCHAR(1024),TOLBUTAMIDE VARCHAR(1024),PIOGLITAZONE VARCHAR(1024),
# ROSIGLITAZONE VARCHAR(1024),ACARBOSE VARCHAR(1024),MIGLITOL VARCHAR(1024),TROGLITAZONE VARCHAR(1024),
# TOLAZAMIDE VARCHAR(1024),EXAMIDE VARCHAR(1024),CITOGLIPTON VARCHAR(1024),INSULIN VARCHAR(1024),
# GLYBURIDE_METFORMIN VARCHAR(1024),GLIPIZIDE_METFORMIN VARCHAR(1024),GLIMEPIRIDE_PIOGLITAZONE VARCHAR(1024),
# METFORMIN_ROSIGLITAZONE VARCHAR(1024),METFORMIN_PIOGLITAZONE VARCHAR(1024),CHANGE_1 VARCHAR(1024),
# DIABETESMED INT,READMITTED TINYINT,DIAG_1_DESC VARCHAR(1024),DIAG_2_DESC VARCHAR(1024),
# DIAG_3_DESC VARCHAR(1024),PRED_PROB FLOAT);


# Specify the path you put all the files
path = "/Users/jujutyr/Desktop/singhealth/"

# Python Packages Used
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# importing in csv file
df = pd.read_csv(path + "data/Diabetes_Dev.csv")

# Understanding the imported dataset
df.info()
df.head()

# Transforming some of the existing columns
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x is True else 0)
df['diabetesMed'] = df['diabetesMed'].apply(lambda x: 1 if x == 'Yes' else 0)

# Example of some pre modeling analysis
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)

# Selecting features and label
feature_names = ['time_in_hospital','num_lab_procedures','num_procedures','number_outpatient','number_emergency','number_inpatient','number_diagnoses','num_medications','diabetesMed'] 
X = df[feature_names]
y = df['readmitted']

# Splitting the dataset to training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Applying normalisation technique
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Training a Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Pring the performance metrics
pred = logreg.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Example of Predicted Probability
pred_prob = logreg.predict_proba(X_test)
print(pred_prob[:,1])

# Assuming the model is selected and next step is to save the model to disk
modelname = path + "model/readm_finalized_pred_model_v1.sav"
pickle.dump(logreg, open(modelname, 'wb'))

########################### Development Script End #####################################

########################### Deployment Strategies ######################################
########################### 1. Batch Mode Sample #######################################

# Daily file send for Readmission Prediction 
dfp = pd.read_csv(path + "data/Diabetes_Prod.csv")

# Load developed model
prod_modelname = path + "model/readm_finalized_pred_model_v1.sav"
loaded_model = pickle.load(open(prod_modelname, 'rb'))

# Prepared the data for scoring
feature_names = ['time_in_hospital','num_lab_procedures','num_procedures','number_outpatient','number_emergency','number_inpatient','number_diagnoses','num_medications','diabetesMed'] 
#dfp['readmitted'] = dfp['readmitted'].apply(lambda x: 1 if x is True else 0)
dfp['diabetesMed'] = dfp['diabetesMed'].apply(lambda x: 1 if x == 'Yes' else 0)
X = dfp[feature_names]
#y_test = dfp['readmitted']
X = scaler.fit_transform(X)

# Scoring & merge score to the last column of the dataset
result = loaded_model.predict_proba(X)
print(result[:,1])

dfp['pred_prob'] = result[:,1]

print("judith")
print(len(dfp))
# Export the data into a table in ODS

"""// Code to export dataset into a database using ODBC //"""
'''
Create a mapping of df dtypes to mysql data types (not perfect, but close enough)
'''
def dtype_mapping():
    return {'object' : 'VARCHAR(1024)',
        'int64' : 'INT',
        'float64' : 'FLOAT',
        'datetime64' : 'DATETIME',
        'bool' : 'TINYINT',
        'category' : 'VARCHAR(1024)',
        'timedelta[ns]' : 'VARCHAR(1024)'}
'''
Create a sqlalchemy engine
'''
import pandas as pd
from sqlalchemy import create_engine

infile = r'path/to/file.csv'
db = 'a001_db'
db_tbl_name = 'a001_rd004_db004'
def mysql_engine(user = 'root', password = 'abc', host = '127.0.0.1', port = '3306', database = 'a001_db'):
    engine = create_engine("mysql://{0}:{1}@{2}:{3}/{4}?charset=utf8".format(user, password, host, port, database))
    return engine

'''
Create a mysql connection from sqlalchemy engine
'''
def mysql_conn(engine):
    conn = engine.raw_connection()
    return conn
'''
Create sql input for table names and types
'''
def gen_tbl_cols_sql(df):
    dmap = dtype_mapping()
    sql = "pi_db_uid INT AUTO_INCREMENT PRIMARY KEY"
    df1 = df.rename(columns = {"" : "nocolname"})
    hdrs = df1.dtypes.index
    hdrs_list = [(hdr, str(df1[hdr].dtype)) for hdr in hdrs]
    for i, hl in enumerate(hdrs_list):

        sql += ",{0} {1}".format(str(hl[0]).replace('.', '_').upper(), dmap[hl[1]])
    return sql


# INSERT
# INTO
# TestDB.dbo.READMISSION_DATA([BusinessEntityID], [FirstName], [LastName])
# values(?, ?, ?)", row['BusinessEntityID'], row['FirstName'] , row['LastName'])


def gen_insert_col(df):
    dmap = dtype_mapping()
    sql = "INSERT INTO TestDB.dbo.READMISSION_DATA("
    df1 = df.rename(columns = {"" : "nocolname"})
    hdrs = df1.dtypes.index
    hdrs_list = [(hdr, str(df1[hdr].dtype)) for hdr in hdrs]
    for i, hl in enumerate(hdrs_list):

        sql += "[{0}],".format(str(hl[0]).replace('.', '_').upper())

    for i, hl in enumerate(hdrs_list):
        print(len(hdrs_list))
        a = "values(?"
        for z in range(len(hdrs_list)-1):

            a+=",?"


        sql_values = a
    sql_values +=")\""
    sql_2=""
    for i, hl in enumerate(hdrs_list):

        sql_2 += "row['{0}'],".format(str(hl[0]).replace('.', '_').upper())
    return sql+") "+sql_values+sql_2+")"
'''
Create a mysql table from a df
'''
def create_mysql_tbl_schema(df, conn, db, tbl_name):
    tbl_cols_sql = gen_tbl_cols_sql(df)
    print()
    sql = "USE {0}; CREATE TABLE {1} ({2})".format(db, tbl_name, tbl_cols_sql)
    print("glenn")
    print(sql)
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.commit()

'''
Write df data to newly create mysql table
'''


def df_to_mysql(df, engine, tbl_name):
    df.to_sql(tbl_name, engine, if_exists='replace')

# df = dfp
# create_mysql_tbl_schema(df, mysql_conn(mysql_engine()), db, db_tbl_name)
# df_to_mysql(df, mysql_engine(), db_tbl_name)


def create_mysql_tbl_schema(dfp):
    tbl_cols_sql = gen_tbl_cols_sql(dfp)
    print("glenn")
    print(tbl_cols_sql)
    sql = "USE {0}; CREATE TABLE {1} ({2})"


create_mysql_tbl_schema(dfp)


## From DataFrame Pandas to SQL

import pyodbc


import pyodbc
# the DSN value should be the name of the entry in odbc.ini, not freetds.conf
conn = pyodbc.connect('DSN=TestDB;UID=SA;PWD=<YourStrong@Passw0rd>')
crsr = conn.cursor()
rows = crsr.execute("select * from TestDB.dbo.READMISSION_DATA ").fetchall()
print(rows)

print(gen_insert_col(dfp))
dfp.columns = map(str.upper, dfp.columns)
dfp.columns = dfp.columns.str.replace(".", "_")
dfp = dfp.fillna(value=0)
print(len(dfp))
for index,row in dfp.iterrows():
    crsr.execute("INSERT INTO TestDB.dbo.READMISSION_DATA([ROWID],[RACE],[GENDER],[AGE],[WEIGHT],[ADMISSION_TYPE_ID],[DISCHARGE_DISPOSITION_ID],[ADMISSION_SOURCE_ID],[TIME_IN_HOSPITAL],[PAYER_CODE],[MEDICAL_SPECIALTY],[NUM_LAB_PROCEDURES],[NUM_PROCEDURES],[NUM_MEDICATIONS],[NUMBER_OUTPATIENT],[NUMBER_EMERGENCY],[NUMBER_INPATIENT],[DIAG_1],[DIAG_2],[DIAG_3],[NUMBER_DIAGNOSES],[MAX_GLU_SERUM],[A1CRESULT],[METFORMIN],[REPAGLINIDE],[NATEGLINIDE],[CHLORPROPAMIDE],[GLIMEPIRIDE],[ACETOHEXAMIDE],[GLIPIZIDE],[GLYBURIDE],[TOLBUTAMIDE],[PIOGLITAZONE],[ROSIGLITAZONE],[ACARBOSE],[MIGLITOL],[TROGLITAZONE],[TOLAZAMIDE],[EXAMIDE],[CITOGLIPTON],[INSULIN],[GLYBURIDE_METFORMIN],[GLIPIZIDE_METFORMIN],[GLIMEPIRIDE_PIOGLITAZONE],[METFORMIN_ROSIGLITAZONE],[METFORMIN_PIOGLITAZONE],[CHANGE],[DIABETESMED],[READMITTED],[DIAG_1_DESC],[DIAG_2_DESC],[DIAG_3_DESC],[PRED_PROB]) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",row['ROWID'],row['RACE'],row['GENDER'],row['AGE'],row['WEIGHT'],row['ADMISSION_TYPE_ID'],row['DISCHARGE_DISPOSITION_ID'],row['ADMISSION_SOURCE_ID'],row['TIME_IN_HOSPITAL'],row['PAYER_CODE'],row['MEDICAL_SPECIALTY'],row['NUM_LAB_PROCEDURES'],row['NUM_PROCEDURES'],row['NUM_MEDICATIONS'],row['NUMBER_OUTPATIENT'],row['NUMBER_EMERGENCY'],row['NUMBER_INPATIENT'],row['DIAG_1'],row['DIAG_2'],row['DIAG_3'],row['NUMBER_DIAGNOSES'],row['MAX_GLU_SERUM'],row['A1CRESULT'],row['METFORMIN'],row['REPAGLINIDE'],row['NATEGLINIDE'],row['CHLORPROPAMIDE'],row['GLIMEPIRIDE'],row['ACETOHEXAMIDE'],row['GLIPIZIDE'],row['GLYBURIDE'],row['TOLBUTAMIDE'],row['PIOGLITAZONE'],row['ROSIGLITAZONE'],row['ACARBOSE'],row['MIGLITOL'],row['TROGLITAZONE'],row['TOLAZAMIDE'],row['EXAMIDE'],row['CITOGLIPTON'],row['INSULIN'],row['GLYBURIDE_METFORMIN'],row['GLIPIZIDE_METFORMIN'],row['GLIMEPIRIDE_PIOGLITAZONE'],row['METFORMIN_ROSIGLITAZONE'],row['METFORMIN_PIOGLITAZONE'],row['CHANGE'],row['DIABETESMED'],row['READMITTED'],row['DIAG_1_DESC'],row['DIAG_2_DESC'],row['DIAG_3_DESC'],row['PRED_PROB'])
    conn.commit()
crsr.close()
conn.close()

########################### 2. Interactive Mode Sample #######################################

"""// Code a function to take the user input and return the result in JSON //"""

# 'time_in_hospital','num_lab_procedures','num_procedures','number_outpatient','number_emergency','number_inpatient','number_diagnoses','num_medications','diabetesMed'
# Simulating user inputs (submission from web application)
user_in =[[4,29,0,0,0,0,8,10,1]] 
# Prepare the raw data from scoring
user_in = scaler.transform(user_in)
# Load developed model
int_modelname = path + "model/readm_finalized_pred_model_v1.sav"
loaded_int_model = pickle.load(open(int_modelname, 'rb'))

result_int = loaded_int_model.predict_proba(user_in)
print(result_int[:,1])
