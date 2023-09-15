import pandas as pd
import json
import numpy as np
from decimal import Decimal
import re

OTTR_PATH = '/home/peter/WebstormProjects/Transplant_Time_Series/OTTR_Data'
PATIENT_ID = 2158653
BEFORE_DATE = '2014-09-15'

if __name__ == '__main__':
  ottr_base = pd.read_csv(f"{OTTR_PATH}/OTTR_PAT_TRANSPLANT.csv", low_memory=False)
  ottr_labs = pd.read_csv(f"{OTTR_PATH}/OTTR_Labs.csv", low_memory=False)
  ottr_meds = pd.read_csv(f"{OTTR_PATH}/OTTR_MEDICATION.csv", low_memory=False)
  ottr_follow_up = pd.read_csv(f"{OTTR_PATH}/OTTR_FOLLOW_UP_DIAGNOSIS.csv", low_memory=False)

  # Int columns are parsed as float, convert back to int
  for col in ottr_labs.select_dtypes(include='float'):
    ottr_labs[col] = np.floor(pd.to_numeric(ottr_labs[col], errors='coerce')).astype('Int64')

  # Filter down to test patient:
  ottr_labs = ottr_labs[ottr_labs['Patient ID'] == PATIENT_ID]
  ottr_labs = ottr_labs[ottr_labs['Date of Lab'] <= BEFORE_DATE]

  newest_date = ottr_labs['Date of Lab'].max()
  time_diff = pd.Timestamp('2022-05-01') - pd.Timestamp(newest_date)

  # define fuzz function
  def fuzz_value(value):
    if isinstance(value, str) and '-' in value:
      val = pd.Timestamp(value) + time_diff + pd.DateOffset(days=np.random.randint(-3, 3))  # adjust -14 and 14 to control the degree of fuzziness
      return pd.Timestamp(val).date()
    elif isinstance(value, str):
      if re.search(r"[^0-9.]+", value) is not None or value == '.':
        return pd.NA

      if "." in value:
        num_sig_digits = len(value.split('.')[1])
        offset = np.random.normal(0, float(value) * 0.05)  # adjust 0.05 to control the degree of fuzziness
        sig_digits = '1.' + '0' * num_sig_digits
        return float(Decimal(float(value) + offset).quantize(Decimal(sig_digits)))
      else:
        offset = np.random.normal(0, float(value) * 0.05)  # adjust 0.05 to control the degree of fuzziness
        return int(float(value) + offset)
    elif pd.isna(value):
      return value
    else:
      offset = np.random.normal(0, float(value) * 0.05)  # adjust 0.05 to control the degree of fuzziness
      return int(value + offset)


  int_cols = ottr_labs.select_dtypes(include=['Int64']).columns
  for col in ottr_labs.columns:
    if col != 'Patient ID':
      ottr_labs[col] = ottr_labs[col].apply(fuzz_value)
  ottr_labs[int_cols] = ottr_labs[int_cols].astype('Int64')

  df = ottr_labs

  # Create an empty list to hold our data in JSON format
  data = []

  # Loop through each row in the DataFrame and create a dictionary for that row's data
  for index, row in df.iterrows():
    data_dict = {}
    data_dict["Patient ID"] = row["Patient ID"]
    data_dict["Date of Lab"] = row["Date of Lab"]
    sub_data_dict = {}
    for col in df.columns:
      if col not in ["Patient ID", "Date of Lab"] and not pd.isnull(row[col]):
        sub_data_dict[col] = row[col]
    data_dict["Data"] = sub_data_dict
    data.append(data_dict)


  def np_encoder(object):
    if isinstance(object, np.generic):
      return object.item()
    else:
      return str(object)

  # Write the data to a JSON file
  with open(f'./test-patient-{PATIENT_ID}.json', 'w') as f:
    json.dump(data, f, indent=4, default=np_encoder)
