import pandas as pd
import numpy as np
import re
import pickle

CLASSES = ['lived', 'cardio', 'gf', 'cancer', 'inf']
yr_1_labels = ['lived_1_yr', 'cardio_1_yr', 'gf_1_yr', 'cancer_1_yr', 'inf_1_yr']
yr_5_labels = ['lived_5_yr', 'cardio_5_yr', 'gf_5_yr', 'cancer_5_yr', 'inf_5_yr']

df = pd.read_csv('./ottr_standard.csv')
print(df.shape)

X = df.drop(columns=['Unnamed: 0', 'Patient ID', 'Death from infection', 'Death from graft failure', 'Death from cancer',
                         'Death from cardiac', 'Date of Death'])


Y_1 = X[yr_1_labels].idxmax(1)
Y_5 = X[yr_5_labels].idxmax(1)
X= X.drop(columns=(yr_1_labels+yr_5_labels))

yr_1_model = None
with open('./tree_1_yr.pickle', 'rb') as file:
  yr_1_model = pickle.load(file)

yr_5_model = None
with open('./tree_5_yr.pickle', 'rb') as file:
  yr_5_model = pickle.load(file)

y_1_pred = yr_1_model.predict_proba(X)
y_5_pred = yr_5_model.predict_proba(X)

# Combine the predicted probabilities from yr_1_model and yr_5_model
pred_probs = pd.DataFrame(np.concatenate([y_1_pred, y_5_pred], axis=1), columns=yr_1_labels + yr_5_labels)
pred_probs['Patient ID'] = df['Patient ID']
pred_probs['Days since Transplant'] = df['Days since Transplant']

# Find the rows where both models made correct predictions
df['correct'] = (Y_1 == pred_probs[yr_1_labels].idxmax(1)) & (Y_5 == pred_probs[yr_5_labels].idxmax(1))

# Calculate the difference between the predicted probability of the "lived_1_yr" class and the "lived_5_yr" class
df['diff'] = pred_probs['lived_1_yr'] - pred_probs['lived_5_yr']

df = df[(df['correct']) & (df['diff'] > 0.1)]

to_save = pd.merge(df[['Patient ID', 'Days since Transplant'] + yr_1_labels + yr_5_labels], pred_probs, on=['Patient ID', 'Days since Transplant'], how='inner')
to_save.to_csv('./good_predictions.csv')

print(df[['Patient ID', 'Days since Transplant'] + yr_1_labels + yr_5_labels])



# df = df[(df['Days since Transplant'] > 365*5) & (df['Death from infection'] == 1)]
#
# df['Transplant Date'] = pd.to_datetime(df['Transplant Date'], format='mixed')
# df['Date of Death'] = pd.to_datetime(df['Date of Death'], format='mixed')
#
# df['time_to_death'] = (df['Date of Death'] - df['Transplant Date']).dt.days - df['Days since Transplant']
#
# # df = df[(df['time_to_death'] > 400) & (df['time_to_death'] < 1000)]
#
# groups = df.groupby('Patient ID').size().reset_index(name='counts')
# print(groups)
#
# print(df['Patient ID'].unique())