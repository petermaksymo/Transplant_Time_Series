import pandas as pd
import torch
import os

FEATURE_LIST = [
    'TX_ACUTE_REJ_EPISODE', 'REC_LIFE_SUPPORT_OTHER', 'CAN_AGE_AT_LISTING', 'CAN_GENDER',
    'DON_GENDER', 'CAN_LAST_SERUM_SODIUM', 'CAN_LAST_SERUM_CREAT', 'CAN_LAST_SRTR_LAB_MELD',
    'CAN_PREV_HL', 'CAN_PREV_HR', 'CAN_PREV_IN', 'CAN_PREV_KI', 'CAN_PREV_LU',
    'CAN_PREV_PA', 'CAN_PREV_TX', 'DON_AGE', 'REC_AGE_AT_TX',
    'REC_POSTX_LOS', 'REC_PREV_HL', 'REC_PREV_HR', 'REC_PREV_IN',
    'REC_PREV_KI', 'REC_PREV_KP', 'REC_PREV_LI', 'REC_PREV_LU',
    'REC_PREV_PA', 'REC_FAIL_BILIARY', 'CAN_EDUCATION', 'ANGINA',
    'CAN_CEREB_VASC', 'CAN_DRUG_TREAT_COPD', 'CAN_DRUG_TREAT_HYPERTEN', 'CAN_PERIPH_VASC',
    'CAN_PULM_EMBOL', 'CAN_MALIG', 'DON_TY', 'REC_DGN_4235',
    'REC_DGN_4240', 'REC_DGN_4241', 'REC_DGN_4242', 'REC_DGN_4245',
    'REC_DGN_4250', 'REC_DGN_4255', 'REC_DGN_4260', 'REC_DGN_4264',
    'REC_DGN_4265', 'REC_DGN_4270', 'REC_DGN_4271', 'REC_DGN_4272',
    'REC_DGN_4275', 'REC_DGN_4280', 'REC_DGN_4285', 'REC_DGN_4290',
    'REC_DGN_4300', 'REC_DGN_4301', 'REC_DGN_4302', 'REC_DGN_4303',
    'REC_DGN_4304', 'REC_DGN_4305', 'REC_DGN_4306', 'REC_DGN_4307',
    'REC_DGN_4308', 'REC_DGN_4315', 'REC_DGN_4400', 'REC_DGN_4401',
    'REC_DGN_4402', 'REC_DGN_4403', 'REC_DGN_4404', 'REC_DGN_4405',
    'REC_DGN_4410', 'REC_DGN_4420', 'REC_DGN_4430', 'REC_DGN_4450',
    'REC_IMMUNO_MAINT_MEDS', 'REC_FAIL_HEP_DENOVO', 'REC_FAIL_HEP_RECUR', 'REC_FAIL_INFECT',
    'REC_FAIL_PRIME_GRAFT_FAIL', 'REC_FAIL_RECUR_DISEASE', 'REC_FAIL_REJ_ACUTE', 'REC_FAIL_VASC_THROMB',
    'REC_PRIMARY_PAY', 'REC_MALIG', 'REC_HIV_STAT', 'REC_MED_COND_HOSP',
    'REC_MED_COND_ICU', 'TX_ACUTE_REJ_EPISODE_TREATED_ADDITIONAL', 'REC_DGN_4100', 'REC_DGN_4101',
    'REC_DGN_4102', 'REC_DGN_4104', 'REC_DGN_4105', 'REC_DGN_4106',
    'REC_DGN_4107', 'REC_DGN_4108', 'REC_DGN_4110', 'REC_DGN_4200',
    'REC_DGN_4201', 'REC_DGN_4202', 'REC_DGN_4204', 'REC_DGN_4205',
    'REC_DGN_4206', 'REC_DGN_4207', 'REC_DGN_4208', 'REC_DGN_4209',
    'REC_DGN_4210', 'REC_DGN_4212', 'REC_DGN_4213', 'REC_DGN_4214',
    'REC_DGN_4215', 'REC_DGN_4216', 'CAN_RACE_SRTR_BLACK', 'CAN_RACE_SRTR_NATIVE',
    'CAN_RACE_SRTR_PACIFIC', 'CAN_ENTHNICITY_SRTR_LATINO', 'REC_DGN_4451', 'REC_FUNCTN_STAT',
    'TFL_CAD', 'TFL_CREAT', 'TFL_INR', 'TFL_TOT_BILI',
    'TFL_SGPT', 'TFL_SGOT', 'TFL_REJ_EVENT_NUM', 'time_since_transplant',
    'TFL_DIAB_DURING_FOL', 'REC_DGN_4598', 'CAN_RACE_SRTR_WHITE', 'CAN_RACE_SRTR_ASIAN',
    'DON_HIST_DIAB', 'REC_DGN_4510', 'DON_HIST_HYPERTEN', 'CAN_HIST_DIAB',
    'TFL_INSULIN_DEPND', 'TFL_FAIL_BILIARY', 'TFL_GRAFT_STAT', 'TFL_HOSP',
    'TFL_IMMUNO_DISCONT', 'TFL_MALIG', 'TFL_MALIG_LYMPH', 'TFL_MALIG_RECUR_TUMOR',
    'TFL_MALIG_TUMOR', 'TFL_PX_NONCOMP', 'TFL_REJ_TREAT', 'REC_DGN_4220',
    'REC_DGN_4217', 'REC_DGN_4455', 'REC_DGN_4500', 'TFL_ACUTE_REJ_EPISODE'
]

# REC_FUNCTN_STAT gets one-hot encoded to 26 features -> 25 possiilities in DS
# REC_PRIMARY_PAY gets one-hot encoded to 14 features -> 15 possibilities in DS
# CAN_DIAB not included
# REC_DGN_4598 missing?
# CAN_ANGINA -> ANGINA ✓
# ACUTE_REJ_EPISODE -> from 2 to 3 featrues
# = 186 features

# CAN_LAST_SRTR_LAB_MELD may need to be encoded to something

'''
    processed data
        -> training
            -> class 0
            -> class 1
            ...
            -> class 4
        -> validation
        -> holdout
        
    all data saved in memory, subset sampled each epoch
'''

CLASSES = ['lived', 'cardio', 'gf', 'cancer', 'inf']

if __name__ == '__main__':
    print('started')
    path = 'processed_data/train_tensors/'

    for class_name in CLASSES:
        for files in sorted(os.listdir(path+class_name), key=lambda x: int(x.split('.')[0])):
            print(files)



    columns = list(df.columns)
    print(len(set(columns).intersection(FEATURE_LIST)))
    print(len(FEATURE_LIST))

    print(set(FEATURE_LIST) - set(columns))
    print(set(columns) - set(FEATURE_LIST))
