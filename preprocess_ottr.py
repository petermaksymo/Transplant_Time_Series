# All in one preprocessing script for OTTR data

from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import json

OTTR_PATH = '/home/peter/WebstormProjects/Transplant_Time_Series/OTTR_Data'

def has_keyword(df, search_cols, keyword, case_sensitive=False):
  # If substrings is a string, convert it to a list with a single element
  if isinstance(keyword, str):
    keyword = [keyword]

  # Check if the substring is present in either column and return 1 or 0
  # Initialize an empty boolean series to hold the substring checks
  keyword_checks = pd.Series([False] * len(df))

  # Loop over the substrings and check if they are present in either column
  for k in keyword:
    for search_col in search_cols:
      keyword_check = df[search_col].str.contains(k, regex=False) if case_sensitive else \
        df[search_col].str.lower().str.contains(k.lower(), regex=False)
      keyword_checks = keyword_checks | keyword_check

  # Convert the resulting boolean series to an integer series with 1's where the substring is present and 0's where it is not
  return keyword_checks.astype(int)

def preprocess_ottr(is_training=True):
  ottr_base = pd.read_csv(f"{OTTR_PATH}/OTTR_PAT_TRANSPLANT.csv", low_memory=False)
  ottr_labs = pd.read_csv(f"{OTTR_PATH}/OTTR_Labs.csv", low_memory=False)
  ottr_meds = pd.read_csv(f"{OTTR_PATH}/OTTR_MEDICATION.csv", low_memory=False)
  ottr_follow_up = pd.read_csv(f"{OTTR_PATH}/OTTR_FOLLOW_UP_DIAGNOSIS.csv", low_memory=False)

  ottr_lab_cols = ["Patient ID", "Date of Lab", "Serum Sodium (Na)", "Serum Total Bilirubin",
                   "Alkaline Phosphatase (ALP)", "SGOT (AST)", "SGPT (ALT)", "White Blood Cell (WBC)",
                   "Hemoglobin (Hgb)", "Platelets (PLT)", "International Normalized Ratio (INR)",
                   "Serum Creatinine (Cr)", "MELD", "MELD-Na", "Weight", "Height",
                   "Tacrolimus trough level", "Cyclosporine trough level", "Sirolimus trough level"
                   ]

  ottr_base = ottr_base.drop(columns=['Sex', 'Donor Sex', 'DOB'])

  ottr_base["Primary Diagnosis"] = ottr_base["Primary Diagnosis"].str.lower()
  ottr_base["Secondary Diagnosis"] = ottr_base["Secondary Diagnosis"].str.lower()

  # Convert to integer
  ottr_base['Donor type (deceased / living)'] = ottr_base['Donor type (deceased / living)'].replace(
    {'living': 1, 'deceased': 0})

  def has_diagnosis(df, diagnosis):
    return has_keyword(df, ['Primary Diagnosis', 'Secondary Diagnosis'], diagnosis)

  ottr_base['NASH Cirrhosis'] = has_diagnosis(ottr_base, 'NASH')
  ottr_base['HCC'] = has_diagnosis(ottr_base, 'HCC')
  ottr_base['Hepatitis C'] = has_diagnosis(ottr_base, 'Hepatitis C')
  ottr_base['Hepatitis B'] = has_diagnosis(ottr_base, 'Hepatitis B')
  ottr_base['Hepatitis B'] = has_diagnosis(ottr_base, 'Hepatitis B')
  ottr_base['Autoimmune hepatitis cirrhosis'] = has_diagnosis(ottr_base,
                                                              ['Autoimmune Hepatitis', 'Autoimmune Active Hepatitis',
                                                               'Cirrhosis -- Autoimmune'])
  ottr_base['Primary Sclerosing Cholangitis'] = has_diagnosis(ottr_base, 'Primary Sclerosing Cholangitis')
  ottr_base['Primary Biliary Cholangitis'] = has_diagnosis(ottr_base, 'Primary Biliary Cholangitis')
  ottr_base['Alcoholic Cirrhosis'] = has_diagnosis(ottr_base, 'Alcoholic Cirrhosis')
  ottr_base['Fulminant Hepatic Failure'] = has_diagnosis(ottr_base, 'Fulminant Hepatic Failure')
  ottr_base['Other Cirrhosis'] = has_diagnosis(ottr_base, 'Cirrhosis-Other')  # TODO: improve this one
  ottr_base = ottr_base.drop(columns=['Primary Diagnosis', 'Secondary Diagnosis'])

  ottr_base['GF due to HBV/HCV'] = has_keyword(ottr_base, ['Reason Graft Failed'], ['Hepatitis', 'HCV'])
  ottr_base['GF due to Thrombosis'] = has_keyword(ottr_base, ['Reason Graft Failed'], 'Thrombosis')
  ottr_base = ottr_base.drop(columns=['Reason Graft Failed'])

  def has_follow_up_diagnosis(df, diagnosis, case_sensitive=False):
    return has_keyword(df, ['Diagnosis'], diagnosis, case_sensitive)

  ottr_follow_up['Graft failure'] = has_follow_up_diagnosis(ottr_follow_up, 'Graft failure', True)
  ottr_follow_up['Cirrhosis'] = has_follow_up_diagnosis(ottr_follow_up, 'cirrhosis')
  ottr_follow_up['Drainage by PTC'] = has_follow_up_diagnosis(ottr_follow_up, 'PTC drain')
  ottr_follow_up['Esophageal Varices'] = has_follow_up_diagnosis(ottr_follow_up, 'Esophageal Varices')
  ottr_follow_up['Ascites'] = has_follow_up_diagnosis(ottr_follow_up, 'Ascites')
  ottr_follow_up['NAFLD'] = has_follow_up_diagnosis(ottr_follow_up, 'NAFLD')
  ottr_follow_up['Liver Fibrosis'] = has_follow_up_diagnosis(ottr_follow_up, ['Hepatic Fibrosis', 'Chr Hepatitis'])
  ottr_follow_up['Acute Kidney injury'] = has_follow_up_diagnosis(ottr_follow_up, 'Acute Kidney injury')
  ottr_follow_up['Diabetes Mellitis'] = has_follow_up_diagnosis(ottr_follow_up, 'Mellitis')
  ottr_follow_up['Systemic Hypertension'] = has_follow_up_diagnosis(ottr_follow_up, 'Hypertension')
  ottr_follow_up['Atrial Fibrillation'] = has_follow_up_diagnosis(ottr_follow_up, 'Atrial Fibrillation')
  ottr_follow_up['Pleural Effusion'] = has_follow_up_diagnosis(ottr_follow_up, 'Pleural Effusion')
  ottr_follow_up['CMV Viremia'] = has_follow_up_diagnosis(ottr_follow_up, 'CMV Viremia')  # potentially more options?
  ottr_follow_up['Pneumonia'] = has_follow_up_diagnosis(ottr_follow_up, 'Pneumonia')
  ottr_follow_up['Shingles'] = has_follow_up_diagnosis(ottr_follow_up, 'Shingles')
  ottr_follow_up['UTI'] = has_follow_up_diagnosis(ottr_follow_up, 'UTI', True)
  ottr_follow_up['Sepsis'] = has_follow_up_diagnosis(ottr_follow_up, 'Sepsis')
  ottr_follow_up['C.Difficile'] = has_follow_up_diagnosis(ottr_follow_up, 'difficile')
  ottr_follow_up['Osteoporosis'] = has_follow_up_diagnosis(ottr_follow_up, 'Osteoporosis')
  ottr_follow_up['Cholangitis'] = has_follow_up_diagnosis(ottr_follow_up, 'Cholangitis')
  ottr_follow_up['Recurrent HCC'] = has_follow_up_diagnosis(ottr_follow_up,
                                                            ['HCC Recurrence', 'Recurrent Hepatocellular Ca (HCC)'])
  ottr_follow_up['De novo malignancy'] = has_follow_up_diagnosis(ottr_follow_up, 'Malignancy - De Novo')

  # create a list of column names to exclude
  diagnosis_cols = ottr_follow_up.columns.tolist()
  diagnosis_cols.remove('Patient ID')
  diagnosis_cols.remove('Diagnosis')
  diagnosis_cols.remove('Diagnosis Date')

  # filter the dataframe to exclude rows where all encoded values are 0
  ottr_follow_up = ottr_follow_up.loc[ottr_follow_up[diagnosis_cols].sum(axis=1) > 0]

  print(ottr_base['Transplant Date'].unique())
  # convert the "Transplant Date" column to a datetime object and drop transplants newer than 5 years
  ottr_base['Transplant Date'] = pd.to_datetime(ottr_base['Transplant Date'])
  if is_training:
    ottr_base = ottr_base[ottr_base['Transplant Date'] <= ottr_base['Transplant Date'].max() - relativedelta(years=5)]

  ottr_base["Date"] = pd.to_datetime(ottr_base["Transplant Date"]).dt.normalize()
  ottr_follow_up["Date"] = pd.to_datetime(ottr_follow_up["Diagnosis Date"]).dt.normalize()
  ottr_follow_up = ottr_follow_up.drop(columns=['Diagnosis', 'Diagnosis Date'])
  ottr_full = pd.merge(ottr_base, ottr_follow_up, on=["Patient ID", "Date"], how="outer")

  ottr_labs = ottr_labs[ottr_lab_cols]
  lab_data_cols = ottr_lab_cols
  lab_data_cols.remove('Patient ID')
  lab_data_cols.remove('Date of Lab')
  ottr_labs.replace('', inplace=True)
  ottr_labs = ottr_labs.dropna(subset=lab_data_cols, how='all')
  ottr_labs.fillna('', inplace=True)

  ottr_labs["Date"] = pd.to_datetime(ottr_labs["Date of Lab"]).dt.normalize()
  ottr_labs = ottr_labs.drop(columns=['Date of Lab'])
  ottr_full = pd.merge(ottr_full, ottr_labs, on=["Patient ID", "Date"], how="outer")

  # Drop rows with no lab values:
  ottr_full = ottr_full[ottr_full['Patient ID'].isin(set(ottr_base['Patient ID']) & set(ottr_labs['Patient ID']))]

  # Forward fill base values
  cols_to_ffill = ottr_base.columns.tolist()
  cols_to_ffill.remove('Date')
  ottr_full[cols_to_ffill] = ottr_full.groupby('Patient ID')[cols_to_ffill].fillna(method='ffill')
  ottr_full.reset_index(drop=True)

  # calculate the time difference between "Date" and "Transplant Date" columns
  ottr_full['Days since Transplant'] = (ottr_full['Date'] - ottr_full['Transplant Date']).dt.days
  ottr_full = ottr_full.drop(columns=['Date', 'Transplant Date'])

  ottr_full = ottr_full.sort_values(['Patient ID', 'Days since Transplant'])

  if is_training:
    infection_keywords = [
      'Recurrent Hepatitis C', 'PNEUMONIA', 'SEPSIS', 'HCV', 'INFECTION', 'PTLD', 'Septic Shock', 'SEPTICEMIA',
      'COVID', 'Recurrent Hep C', 'SEPTICAEMIA', 'RECURRENT HEP B', 'ARDS', 'Primary Renal Disease',
      'ACUTE PANCREATITIS', 'Invasive Aspergillus',
    ]
    graft_failure_keywords = [
      'Liver Failure', 'REJECTION', 'Cirrhosis', 'Graft', 'LIVER COMPLICATIONS', 'Hepatorenal failure',
      'Chronic Allograft Dysfunction',
      'H.A.T.', 'HAT', 'H.A.T', 'HEPATIC ARTERY THROMBOSIS', 'HEPATIC ART THROMB'
    ]
    cancer_keywords = [
      'CANCER', 'Carcinoma', 'LYMPHOMA', 'Mesothelioma', 'Malignancy', 'ADENOCARCINOMA', 'Recurrent HCC', 'LEUKEMIA',
      'METASTATIC MELANOMA', 'MALIGNANT', 'Pancreatic Mass', 'TUMOR',
    ]
    cardiac_keywords = [
      'STROKE', 'Aorticstenosis', 'HEART', 'M.I.', 'CARDIAC', 'CEREBROVASCULAR ACCIDENT', 'HEPATIC ARTERY ANEURYSM',
      'Complications Following MI', 'MYOCARDIAL INFARCT', 'Cardiovascular', 'CORONARY ARTERY DIS',
      'MYOCARDIAL INFARCTION',
      'RUPTURED AORTIC ANEURYSM', 'MI.', 'Severe Aortic Stenosis', 'Massive CVA',
      'Technical Failure - Other, RV Clot Thrombosis, Cardiogenic shock',
    ]

    ottr_full['Cause of Death'] = ottr_full['Cause of Death'].fillna('')
    ottr_full_non_func = ottr_full.loc[~ottr_full['Cause of Death'].str.lower().str.contains('functioning')]
    ottr_full['Death from infection'] = has_keyword(ottr_full_non_func, ['Cause of Death'], infection_keywords)
    ottr_full['Death from graft failure'] = has_keyword(ottr_full_non_func, ['Cause of Death'], graft_failure_keywords)
    ottr_full['Death from cancer'] = has_keyword(ottr_full_non_func, ['Cause of Death'], cancer_keywords)
    ottr_full['Death from cardiac'] = has_keyword(ottr_full_non_func, ['Cause of Death'], cardiac_keywords)

    # Drop patients with uncategorized deaths
    ottr_full = ottr_full[(ottr_full['Death from infection'] == 1) |
                          (ottr_full['Death from graft failure'] == 1) |
                          (ottr_full['Death from cancer'] == 1) |
                          (ottr_full['Death from cardiac'] == 1) |
                          (ottr_full['Date of Death'].isnull())]

  ottr_full = ottr_full.drop(columns='Cause of Death')

  # reset the index
  ottr_full = ottr_full.reset_index(drop=True)

  grouped = ottr_full.groupby('Patient ID')
  print(f"# of deaths from infection: {len(ottr_full[ottr_full['Death from infection'] == 1].groupby('Patient ID'))}")
  print(
    f"# of deaths from graft failure: {len(ottr_full[ottr_full['Death from graft failure'] == 1].groupby('Patient ID'))}")
  print(f"# of deaths from cancer: {len(ottr_full[ottr_full['Death from cancer'] == 1].groupby('Patient ID'))}")
  print(f"# of deaths from cardiac: {len(ottr_full[ottr_full['Death from cardiac'] == 1].groupby('Patient ID'))}")
  print(f"# of survive: {len(ottr_full[ottr_full['Date of Death'].isnull()].groupby('Patient ID'))}")

if __name__ == '__main__':
  preprocess_ottr()