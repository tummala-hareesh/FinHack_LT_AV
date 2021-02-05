# FINHACK - a LTFS & AV hackathon
In this hackathon challenge, we are going to develop a Machine Learning model for **Top-up loan UpSell Prediction**. 

## Problem
L&T Financial Services (LTFS) wants to understand the most suitable time to offer a loan top-up to its customers. A detailed description about the project can be found [here](https://datahack.analyticsvidhya.com/contest/ltfs-data-science-finhack-3/#ProblemStatement).

## Data
LTFS has provided following data of its customers for this hackathon.

- **Customer's Demographics**: demographic information related to frequency of the loan, tenure of the loan, disbursal amount for a loan & LTV etc., 
	
- **Bureau data**: behavioural and transacational attributes of the customers like current balance, Loan Amount, Overdue etc.,

## Dependencies
- Create working environment for the project 
```sh
python3 -m venv FinHack_LT_AV/py3-finhack
```

- Activate working environment
```sh
source FinHack_LT_AV/py3-finhack/bin/activate
```

- To set env. python for jupyter-notebook
```sh
python -m ipykernel install --user --name py3-finhack --display-name "py3-finhack-d"
```

- Install python modules after activating the environemnt
```sh
pip install -r requirements.txt
```

- Deactive working environment
```sh
deactive
```

## Phases for completing the project
### 1. Exploratory phase
In this phase, we are going to quickly analyze (perform data cleaning, if needed, EDA) the data we have and build an _upsell prediction_ model rightaway using *PyCaret* library.

### 2. Development phase
After exploring the possibilities of creating an ML model, in the development stage, we are going to custom ML model using *Sklearn* library. 

### 3. Expansion phase
Look into the possiblities of using TensorFlow or PyTorch to solve the same problem more (computationally) effeciently on GPUs. We may not reach this phase of the project. But, it is a point to keep in mind and try out if time permits. 


## Solution - **Code**
Problem solution code can be found here.


## Solution - **Approach**
The approach that we followed to solve the problem is:
1. Brief description about the approach 
2. What data-preprocessing/feature engineering ideas worked? How did we discover them?
3. What does the final model look like? How did you reach it?