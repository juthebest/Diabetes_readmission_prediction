
# Python Packages Used
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from flask import Flask
from werkzeug.routing import BaseConverter
# Specify the path you put all the files
path = "/Users/jujutyr/Desktop/singhealth/"
class IntListConverter(BaseConverter):
    """Match ints separated with ';'."""

    # at least one int, separated by ;, with optional trailing ;
    regex = r'\d+(?:;\d+)*;?'

    # this is used to parse the url and pass the list to the view function
    def to_python(self, value):
        return [int(x) for x in value.split(';')]

    # this is used when building a url with url_for
    def to_url(self, value):
        return ';'.join(str(x) for x in value)

# register the converter when creating the app
app = Flask(__name__)
app.url_map.converters['int_list'] = IntListConverter

# use the converter in the route
@app.route('/user/<int_list:user_in>')
def process_readmission(user_in):
    # Prepare the raw data from scoring
    scaler = MinMaxScaler()
    user_in = scaler.fit_transform([user_in])
    # Load developed model
    int_modelname = path + "model/readm_finalized_pred_model_v1.sav"
    loaded_int_model = pickle.load(open(int_modelname, 'rb'))

    result_int = loaded_int_model.predict_proba(user_in)
    print(result_int[:, 1])
    return(str(result_int[:, 1]))


app.run()