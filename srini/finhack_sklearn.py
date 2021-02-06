#!/usr/bin/env python
# coding: utf-8
# Import modules
import sys
import os
import pandas as pd
import numpy as np
# Sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Functions

def handle_non_numerical_data(df):
    """
    This function to handle for non-numerical data. First, cycle through the
    columns in the Pandas dataframe. For columns that are not numbers,
    find their unique elements. This can be done by simply take a set of the
    column values. From here, the index within that set can be the new
    "numerical" value or "id" of the text data.
    """
    data_dict = {}
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))
            #print(text_digit_vals)
            data_dict[column] = text_digit_vals

    return df, data_dict

## Check modules version
print('Notebook running from:',sys.executable)
print('Pandas:',pd.__version__)

## List data-raw
path_pwd = os.getcwd()
path_data_train = path_pwd+'/'+'data_raw/train/'
os.listdir(path_data_train)

## Set trainig data files
file_customer_train = path_data_train+'train_Data.xlsx'
print('Customer file:', file_customer_train)

## Load customer trainig data
df_customer = pd.read_excel(file_customer_train, engine='openpyxl')
df_customer.head()

# Shape as data loaded
df_customer = df_customer.dropna() # Drop NaN and missing data rows
print(df_customer.shape)

data_drop_list  = ['BranchID', 'Area', 'DisbursalDate', 'MaturityDAte','AuthDate',
                'AssetID', 'ManufacturerID', 'SupplierID',
                'City', 'State', 'ZiPCODE']

df_customer = df_customer.drop(data_drop_list, axis=1)

# Convert non-numeric to unique numerical values
df_customer, column_mapped_dict = handle_non_numerical_data(df_customer)
print(column_mapped_dict)

# Divide data set into train and test
data, data_unseen, data_target, data_unseen_target =train_test_split(
    df_customer.drop('Top-up Month', axis=1),
    df_customer['Top-up Month'] ,test_size = 0.1, random_state=13)

# Training

## Logistic Regression
clf = LogisticRegression().fit(data, data_target)
predictions = clf.predict(data_unseen)

print(classification_report(data_unseen_target, predictions))

# Testing
path_data_test = path_pwd+'/'+'data_raw/test/'
os.listdir(path_data_test)

## Set testing data
file_customer_test = path_data_test+'test_Data.xlsx'
print('Customer file:', file_customer_test)


## Load customer trainig data
df_customer_test = pd.read_excel(file_customer_test, engine='openpyxl')
#df_customer_test.head()

# Shape as data loaded
print(df_customer_test.shape)
df_customer_test = df_customer_test.dropna() # Drop NaN and missing data rows
print(df_customer_test.shape)

df_customer_test = df_customer_test.drop(data_drop_list, axis=1)

# Convert non-numeric to unique numerical values
for key in list(column_mapped_dict.keys()):
    print(key)
    if key in list(df_customer_test.keys()):
        print(column_mapped_dict[key].keys())
        print(column_mapped_dict[key].values())
        df_customer_test[key] = df_customer_test[key].replace(list(column_mapped_dict[key].keys()),list(column_mapped_dict[key].values()))

predictions_test = clf.predict(df_customer_test)

"""

# Organize data
data = df_customer.sample(frac=0.90, random_state=786)
data_unseen = df_customer.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# ***

# ## Setup

# Sub-set of data to test
nsubset = 96413
data = data.iloc[:nsubset,:]
data.shape

# list of attributes to consider
list_attributes = ['Frequency', 'InstlmentMode', 'LoanStatus', 'PaymentMode', 'Tenure', 'AssetCost', 'AmountFinance', 'DisbursalAmount', 'EMI', 'DisbursalDate', 'MaturityDAte', 'AuthDate', 'LTV', 'SEX', 'AGE', 'MonthlyIncome','Top-up Month']

# Divide data into data_features and data_target
#data = data[list_attributes]

data.shape

## Setup
mclf = setup(data = data, target = 'Top-up Month', session_id=123, experiment_name='FinHack1')

# Available models for multilevel classification
models()

# Identify the best model - Run overnight
#best_model = compare_models()

# ## Build Model

# Create model
model = create_model('ada', fold=4)

# hyper-param tuning
tuned_model = tune_model(model, fold=4)

plot_model(tuned_model, plot='auc')


# Finalize the model
final_model = finalize_model(tuned_model)
print(final_model)

# ## Save Model

# model_name (INPUT)
model_name = 'ada'

# model_save_name
model_save_name  = model_name+'_'+str(nsubset)

# SAve model
save_model(final_model, 'models_saved/'+model_save_name)

# ## Re-load Model

# Load saved model
saved_model = load_model('models_saved/'+model_save_name)

unseen_predictions = predict_model(saved_model, data=data_unseen)
unseen_predictions.head()

unseen_predictions['Label'].value_counts()

from pycaret.utils import check_metric
check_metric(data_unseen['Top-up Month'], unseen_predictions['Label'], metric = 'Accuracy')


### Training data from Hackthon
file_customer_test = 'data_raw/test_Data.xlsx'


df_customer_test = pd.read_excel(file_customer_test, engine='openpyxl')
df_customer_test.head()

list_attributes2 = ['ID','Frequency', 'InstlmentMode', 'LoanStatus', 'PaymentMode', 'Tenure', 'AssetCost', 'AmountFinance', 'DisbursalAmount', 'EMI', 'DisbursalDate', 'MaturityDAte', 'AuthDate', 'LTV', 'SEX', 'AGE', 'MonthlyIncome']


#data_unseen2 = df_customer_test[list_attributes2]
data_unseen2 = df_customer_test.copy()
data_unseen2.shape


unseen_predictions2 = predict_model(final_model, data=data_unseen2)
unseen_predictions2.head()

unseen_predictions2['Label'].value_counts()

save_submission = unseen_predictions2[['ID','Label']]
save_submission.rename(columns={'Label':'Top-up Month'}, inplace=True)
save_submission.head()

save_submission.to_csv(model_save_name+'.csv', index=False)
"""
