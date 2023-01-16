# exec(open("preprocess_data.py").read())
import pandas as pd
from pandas.api.types import is_numeric_dtype
import scipy.stats as stats
import numpy as np
import re
import os
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

DATASET = 'SRTR' # SRTR, BOTH -> both will only extract common features from SRTR found in UHN

SRTR_FEATURE_LIST = [
    'ACCUTE_REJ_EPISODE', 'REC_LIFE_SUPPORT_OTHER', 'CAN_AGE_AT_LISTING', 'CAN_GENDER',
    'DON_GENDER', 'CAN_LAST_SERUM_SODIUM', 'CAN_LAST_SERUM_CREAT', 'CAN_LAST_SRTR_LAB_MELD',
    'CAN_PREV_HL', 'CAN_PREV_HR', 'CAN_PREV_IN', 'CAN_PREV_KI', 'CAN_PREV_LU',
    'CAN_PREV_PA', 'CAN_PREV_TX', 'DON_AGE', 'REC_AGE_AT_TX',
    'REC_POSTX_LOS', 'REC_PREV_HL', 'REC_PREV_HR', 'REC_PREV_IN',
    'REC_PREV_KI', 'REC_PREV_KP', 'REC_PREV_LI', 'REC_PREV_LU',
    'REC_PREV_PA', 'REC_FAIL_BILIARY', 'CAN_EDUCATION', 'ANGINA',
    'CAN_CEREB_VASC', 'CAN_DRUG_TREAT_COPD', 'CAN_DRUG_TREAT_HYPERTEN', 'CAN_PERIPH_VASC',
    'CAN_PULM_OMBOL', 'CAN_MALIG', 'DON_TY', 'REC_DGN_4235',
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
    'REC_MED_COND_ICU', 'ACUTE_REJ_EPISODE_TREATED_ADDITIONAL', 'REC_DGN_4100', 'REC_DGN_4101',
    'REC_DGN_4102', 'REC_DGN_4104', 'REC_DGN_4105', 'REC_DGN_4106',
    'REC_DGN_4107', 'REC_DGN_4108', 'REC_DGN_4110', 'REC_DGN_4200',
    'REC_DGN_4201', 'REC_DGN_4202', 'REC_DGN_4204', 'REC_DGN_4205',
    'REC_DGN_4206', 'REC_DGN_4207', 'REC_DGN_4208', 'REC_DGN_4209',
    'REC_DGN_4210', 'REC_DGN_4212', 'REC_DGN_4213', 'REC_DGN_4214',
    'REC_DGN_4215', 'REC_DGN_4216', 'CAN_RACE_SRTR_BLACK', 'CAN_RACE_SRTR_NATIVE',
    'CAN_RACE_SRTR_PACIFIC', 'CAN_ETHNICITIY_SRTR_LATINO', 'REC_DGN_4451', 'REC_FUNCTN_STAT',
    'TFL_CAD', 'TFL_CREAT', 'TFL_INR', 'TFL_TOT_BILI',
    'TFL_SGPT', 'TFL_SGOT', 'TFL_REJ_EVENT_NUM', 'Time_since_transplant',
    'TFL_DIAB_DURING_FOL', 'REC_DGN_4598', 'CAN_RACE_SRTR_WHITE', 'CAN_RACE_SRTR_ASIAN',
    'DON_HIST_DIAB', 'REC_DGN_4510', 'DON_HIST_HYPERTEN', 'CAN_HIST_DIAB',
    'TFL_INSULIN_DEPND', 'TFL_FAIL_BILIARY', 'TFL_GRAFT_STAT', 'TFL_HOSP',
    'TFL_IMMUNO_DISCONT', 'TFL_MALIG', 'TFL_MALIG_LYMPH', 'TFL_MALIG_RECUR_TUMOR',
    'TFL_MALIG_TUMOR', 'TFL_PX_NONCOMP', 'TFL_REJ_TREAT', 'REC_DGN_4220',
    'REC_DGN_4217', 'REC_DGN_4455', 'REC_DGN_4500'
]

COMMON_FEATURES = [
    'time_since_transplant', 'CAN_LAST_SERUM_SODIUM', 'TFL_TOT_BILI', 'TFL_ALKPHOS',
    'TFL_SGOT', 'TFL_SGPT', 'TFL_INR', 'CAN_LAST_SRTR_LAB_MELD', 'TFL_BMI',
    'CAN_LAST_SERUM_CREAT', 'DON_AGE', 'REC_AGE_AT_TX', 'TFL_REJ_EVENT_NUM',
    'TFL_GRAFT_STAT', 'REC_FAIL_HEP_RECUR', 'REC_FAIL_VASC_THROMB', 'REC_DGN_4214',
    'REC_DGN_4212', 'REC_DGN_4204', 'REC_DGN_4202', 'REC_DGN_4215', 'REC_DGN_4208',
    'REC_DGN_4240', 'REC_DGN_4220', 'REC_DGN_4110', 'REC_DGN_4401',
    'sirolimus_prescription', 'tacrolimus_prescription', 'cyclosporine_prescription'
]

UHN_FEATURES = [
    'DONOR_AGE', 'R_AGE_TX', 
]

CLASSES = ['lived', 'cardio', 'gf', 'cancer', 'inf']

if __name__ == '__main__':
    prev_tx_li = pd.read_csv("./tx_li.csv", index_col=0, low_memory=False)
    prev_txf_li = pd.read_csv("./txf_li.csv", index_col=0, low_memory=False)
    fol_immuno = pd.read_csv("./fol_immuno.csv", index_col=0, low_memory=False)

    # Filter out follow-up after the paper was published
    # prev_txf_li['TXF_DT'] = pd.to_datetime(prev_txf_li['TFL_PX_STAT_DT'], yearfirst=True)
    # prev_txf_li = prev_txf_li[prev_txf_li.apply(lambda row: True if (row['TXF_DT'] <= pd.Timestamp(year=2019, month=9, day=1,)) else False, axis=1)]
    # prev_txf_li.drop('TXF_DT', axis=1, inplace=True)

    print(f"Number of patients in original dataset:{prev_tx_li['TRR_ID'].nunique()}")

    # clean byte prefix from csv columns
    def clean_strings(df):
        return df.applymap(lambda x: re.findall("b'([^']*)'", x)[0]
                    if type(x) is str and len(re.findall("b'([^']*)'", x)) > 0
                    else x)

    #prev_tx_li = clean_strings(prev_tx_li)
    #prev_txf_li = clean_strings(prev_txf_li)

    # filter out multi-transplant patients
    prev_tx_li = prev_tx_li[(prev_tx_li['REC_TX_ORG_TY'] == 'LI')]
    prev_txf_li = prev_txf_li[(prev_txf_li['REC_TX_ORG_TY'] == 'LI')]

    print(f"Number of patients after filtering out multi-transplant patients:{prev_tx_li['TRR_ID'].nunique()}")

    # create arrays with codes for causes of death of interest
    cardio_codes = [4246, 4620, 4623, 4624, 4625, 4626, 4630]
    inf_codes = [4800, 4801, 4802, 4803, 4804, 4805, 4806, 4810, 4811, 4660, 4645]
    malig_codes = [4850, 4851, 4855, 4856]
    gf_codes = [4600, 4601, 4602, 4603, 4604, 4605, 4606, 4610]

    prev_txf_li = prev_txf_li.sort_values(['TRR_ID', 'TFL_PX_STAT_DT'])

    prev_txf_li = prev_txf_li.merge(prev_tx_li[['TRR_ID', 'PERS_OPTN_DEATH_DT', 'REC_AGE_AT_TX', 'CAN_PREV_TX']],
                                    how='left', left_on='TRR_ID', right_on='TRR_ID')
    prev_txf_li = prev_txf_li[prev_txf_li['CAN_PREV_TX'] == 0]
    prev_txf_li.drop('CAN_PREV_TX', axis=1, inplace=True)
    print(f"Number of patients after filtering multi-transplant patients: {prev_tx_li['TRR_ID'].nunique()}")

    prev_txf_li = prev_txf_li[prev_txf_li['REC_AGE_AT_TX'] >= 18]
    print(f"Number of patients after filtering patients under 18: {prev_txf_li['TRR_ID'].nunique()}")


    # filter out transplant before 2002 and after Sept 30, 2014
    prev_txf_li['TX_DT'] = pd.to_datetime(prev_txf_li['REC_TX_DT'], yearfirst=True)
    prev_txf_li = prev_txf_li[(prev_txf_li['REC_TX_DT'] >= '2002-01-01') & (prev_txf_li['REC_TX_DT'] <= '2014-09-30')]
    print(f"Number of patients after filtering pre 2002, post Sept 30, 2014: {prev_txf_li['TRR_ID'].nunique()}")

    # # Remove follow-ups past 5 years after transplant
    # prev_txf_li['fol_yr'] = pd.to_numeric(prev_txf_li['TFL_PX_STAT_DT'].str[:4])
    # prev_txf_li['rec_yr'] = pd.to_numeric(prev_txf_li['REC_TX_DT'].str[:4])
    # prev_txf_li = prev_txf_li[prev_txf_li.apply(lambda row:
    #                                             True if (row['rec_yr']+6 >= row['fol_yr']) else False, axis=1)]
    # prev_txf_li.drop('rec_yr', axis=1, inplace=True)

    # fill COD values
    prev_txf_li['TFL_COD'] = prev_txf_li.groupby('TRR_ID')['TFL_COD'].transform('last')
    prev_txf_li['TFL_COD2'] = prev_txf_li.groupby('TRR_ID')['TFL_COD2'].transform('last')
    prev_txf_li['TFL_COD3'] = prev_txf_li.groupby('TRR_ID')['TFL_COD3'].transform('last')

    # filter out multi-COD patients
    def get_cod_patients(codes):
        return prev_txf_li[(prev_txf_li['TFL_COD'].isin(codes))
                & (prev_txf_li['TFL_COD2'].isna())
                & (prev_txf_li['TFL_COD3'].isna())]['TRR_ID'].unique()

    patients = {}
    patients['cardio'] = get_cod_patients(cardio_codes)
    patients['cancer'] = get_cod_patients(malig_codes)
    patients['gf'] = get_cod_patients(gf_codes)
    patients['inf'] = get_cod_patients(inf_codes)
    patients['lived'] = prev_txf_li[(prev_txf_li['TFL_COD'].isna())
                                 & (~prev_txf_li['TRR_ID'].isin(patients['cardio']))
                                 & (~prev_txf_li['TRR_ID'].isin(patients['cancer']))
                                 & (~prev_txf_li['TRR_ID'].isin(patients['gf']))
                                 & (~prev_txf_li['TRR_ID'].isin(patients['inf']))]['TRR_ID'].unique()

    all_patients = np.concatenate([patients[x] for x in patients])

    # Remove follow-ups past 5 years after transplant
    prev_txf_li['fol_yr'] = pd.to_numeric(prev_txf_li['TFL_PX_STAT_DT'].str[:4])
    prev_txf_li['last_fol_yr'] = prev_txf_li.groupby('TRR_ID')['fol_yr'].transform('last')
    prev_txf_li['out_yr'] = prev_txf_li.apply(lambda row:
                                              row['last_fol_yr'] - 5 if row['TRR_ID'] in patients['lived'] else row[
                                                  'last_fol_yr'], axis=1)
    prev_txf_li.drop('last_fol_yr', axis=1, inplace=True)
    prev_txf_li = prev_txf_li[prev_txf_li.apply(lambda row:
                                                True if (row['fol_yr'] <= row['out_yr']) else False, axis=1)]

    # remove patients with DOD but no COD
    dod_array = prev_txf_li[~prev_txf_li['PERS_OPTN_DEATH_DT'].isna()]['TRR_ID'].unique()
    remove_array = np.intersect1d(dod_array, patients['lived'])
    prev_txf_li = prev_txf_li[prev_txf_li.apply(lambda row:
            False if row['TRR_ID'] in remove_array else True, axis=1)]

    txf_li = prev_txf_li[prev_txf_li.apply(lambda row:
            True if row['TRR_ID'] in all_patients else False, axis=1)]
    trr_ids = txf_li['TRR_ID'].unique()

    tx_li = prev_tx_li[prev_tx_li.apply(lambda row:
            True if row['TRR_ID'] in trr_ids else False, axis=1)]


    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(tx_li.groupby('TRR_ID').transform('last')['TFL_COD'].value_counts())

    del prev_tx_li
    del prev_txf_li

    del dod_array
    del remove_array
    del trr_ids

    ids = txf_li['TRR_ID'].unique()

    print(f"Number of patients after multi-COD filtering: {txf_li['TRR_ID'].nunique()}")
    for class_name in CLASSES:
        print(f'Number of {class_name} patients: {np.intersect1d(patients[class_name], ids).shape[0]}')

    tx_len = tx_li.shape[0]
    txf_len = txf_li.shape[0]

    tx_study = pd.DataFrame(index=range(tx_len), columns=range(0))
    txf_study = pd.DataFrame(index=range(txf_len), columns=range(0))

    tx_vars = ["TRR_ID", "REC_DGN", "REC_DGN2",
                "CAN_AGE_AT_LISTING", "CAN_GENDER", "DON_GENDER",
                  "CAN_LAST_SERUM_SODIUM", "CAN_LAST_SERUM_CREAT", "CAN_LAST_SRTR_LAB_MELD",
                  "DON_AGE", "REC_AGE_AT_TX", "REC_POSTX_LOS",
                  "REC_PREV_HR", "REC_PREV_KI", "REC_PREV_LI", "PERS_OPTN_DEATH_DT",
                  "REC_TX_DT", "TFL_COD"]

    txf_vars = ["fol_yr", "TRR_ID", "TRR_FOL_ID", "TFL_COD", "TFL_COD2", "TFL_COD3", "TFL_CREAT",
                 "TFL_INR", "TFL_REJ_EVENT_NUM", "TFL_SGOT", "TFL_SGPT", "TFL_TOT_BILI", "TFL_ALKPHOS",
                 "TFL_BMI", "REC_TX_DT", "TFL_PX_STAT_DT", "TFL_FAIL_DT", "TFL_TXFER_DT"]

    for name in tx_vars:
        tx_study = pd.concat([tx_study.reset_index(drop=True),
                tx_li[name].reset_index(drop=True)], axis=1)

    for name in txf_vars:
        txf_study = pd.concat([txf_study.reset_index(drop=True),
                txf_li[name].reset_index(drop=True)], axis=1)

    tx_study['CAN_GENDER'] = tx_study['CAN_GENDER'].apply(lambda x : 0 if x == 'M' else 1)
    tx_study['DON_GENDER'] = tx_study['DON_GENDER'].apply(lambda x : 0 if x == 'M' else 1)

    tx_study = pd.concat([tx_study.reset_index(drop=True), tx_li['CAN_EDUCATION'].apply(lambda x :
                            np.nan if x not in range(7) else x).reset_index(drop=True)], axis=1)

    tx_study = pd.concat([tx_study.reset_index(drop=True), tx_li['DON_HIST_DIAB'].apply(lambda x :
                            np.nan if x not in range(6) else x).reset_index(drop=True)], axis=1)

    tx_study = pd.concat([tx_study.reset_index(drop=True), tx_li['DON_HIST_HYPERTEN'].apply(lambda x :
                            np.nan if x not in range(6) else x).reset_index(drop=True)], axis=1)

    def diab(cd, cdt):
        if ((cd == 1 and cdt == 1) or
            ( cd == 998 and cdt == 1) or
            (cd == 1 and cdt == 998) or
            (pd.isna(cd) and cdt == 1) or
            (cd == 1 and pd.isna(cdt))):
            return 0
        else:
            return 1

    #tx_study['DIAB'] = tx_li.apply(lambda row: diab(row['CAN_DIAB'], row['CAN_DIAB_TY']), axis=1)
    tx_study = pd.concat([tx_study.reset_index(drop=True), tx_li.apply(lambda row:
                diab(row['CAN_DIAB'], row['CAN_DIAB_TY']), axis=1).rename('CAN_HIST_DIAB').reset_index(drop=True)], axis=1)

    tx_study = pd.concat([tx_study.reset_index(drop=True), tx_li['DON_TY'].replace(
                {'L':1, 'C':0}).reset_index(drop=True)], axis=1)

    na_angina_ids = tx_li[tx_li.apply(lambda row :
            True if ((pd.isna(row['CAN_ANGINA']) or row['CAN_ANGINA'] == 998) and
                (pd.isna(row['CAN_ANGINA_CAD']) or row['CAN_ANGINA_CAD'] == 998))
            else False, axis=1)]['TRR_ID'].unique()

    no_angina_ids = tx_li[tx_li.apply(lambda row:
        True if ((row['CAN_ANGINA'] == 1 and (pd.isna(row['CAN_ANGINA_CAD'])
            or row['CAN_ANGINA_CAD'] in [998, 1])) or (row['CAN_ANGINA_CAD'] == 1
                and (pd.isna(row['CAN_ANGINA']) or row['CAN_ANGINA'] in [998, 1])))
            else False, axis=1)]['TRR_ID'].unique()

    tx_study['ANGINA'] = tx_study['TRR_ID'].apply(lambda x:
                            0 if x in no_angina_ids else (np.nan if x in na_angina_ids else 1))

    del na_angina_ids
    del no_angina_ids

    def to_binary(df, col, pos, neg, na):
        d = {}
        for i in pos:
            d[i] = 1
        for i in neg:
            d[i] = 0
        for i in na:
            d[i] = -1
        tmp = df[col]
        tmp = tmp.replace(d)
        tmp = tmp.fillna(-1)
        return tmp

    cols = ["CAN_CEREB_VASC", "CAN_DRUG_TREAT_COPD", "CAN_DRUG_TREAT_HYPERTEN",
                  "CAN_PERIPH_VASC", "CAN_PULM_EMBOL", "CAN_MALIG",
                  "REC_IMMUNO_MAINT_MEDS", "REC_LIFE_SUPPORT",
                  "REC_FAIL_BILIARY", "REC_FAIL_HEP_DENOVO", "REC_FAIL_HEP_RECUR", "REC_FAIL_INFECT",
                  "REC_FAIL_PRIME_GRAFT_FAIL", "REC_FAIL_RECUR_DISEASE", "REC_FAIL_REJ_ACUTE",
                  "REC_FAIL_VASC_THROMB", "REC_MALIG"]

    for col in cols:
        tmp = to_binary(tx_li, col, ['Y'], ['N'], ['U'])
        tx_study = pd.concat([tx_study.reset_index(drop=True),
            tmp.reset_index(drop=True)], axis=1)

    rec_fail_cols = tx_study.columns.str.contains('REC_FAIL_')
    rec_fail = tx_study[tx_study.columns[rec_fail_cols]]
    rec_fail = rec_fail.replace(to_replace=-1, value=np.nan)
    tx_study['REC_FAIL_is_na'] = rec_fail.isna().max(axis=1).apply(lambda x: 0 if x==0 else 1)

    tmp = to_binary(tx_li, 'REC_HIV_STAT', ['P'], ['N'], ['U', 'I', 'C', 'ND'])
    tx_study = pd.concat([tx_study.reset_index(drop=True),
        tmp.reset_index(drop=True)], axis=1)

    tmp = to_binary(tx_li, 'REC_MED_COND', [1,2], [3], [np.nan])
    tx_study = pd.concat([tx_study.reset_index(drop=True),
        tmp.rename('REC_MED_COND_HOSP').reset_index(drop=True)], axis=1)

    tmp = to_binary(tx_li, 'REC_MED_COND', [1], [2,3], [np.nan])
    tx_study = pd.concat([tx_study.reset_index(drop=True),
        tmp.rename('REC_MED_COND_ICU').reset_index(drop=True)], axis=1)

    tmp = to_binary(tx_li, 'REC_ACUTE_REJ_EPISODE', [1,2], [3], [np.nan])
    tx_study = pd.concat([tx_study.reset_index(drop=True),
        tmp.rename('TX_ACUTE_REJ_EPISODE').reset_index(drop=True)], axis=1)

    tmp = to_binary(tx_li, 'REC_ACUTE_REJ_EPISODE', [1], [2,3], [np.nan])
    tx_study = pd.concat([tx_study.reset_index(drop=True),
        tmp.rename('TX_ACUTE_REJ_EPISODE_TREATED_ADDITIONAL').reset_index(drop=True)], axis=1)

    tx_study['REC_DGN'] = tx_study['REC_DGN'].apply(lambda x: np.nan if x == 999 else x)
    tx_study['REC_DGN2'] = tx_study['REC_DGN2'].apply(lambda x: np.nan if x == 999 else x)

    levels = np.union1d(tx_study['REC_DGN'].unique(), tx_study['REC_DGN2'].unique())
    levels = levels[~np.isnan(levels)]
    dummies1 = pd.get_dummies(tx_study['REC_DGN'])
    dummies2 = pd.get_dummies(tx_study['REC_DGN2'])
    dummies12 = pd.DataFrame(index=range(tx_study.shape[0]), columns=range(0))

    for l in levels:
        s = pd.Series([], dtype=int)
        if l in dummies1 and l in dummies2:
            s = dummies1[l].combine(dummies2[l], max)
        elif l not in dummies1 and l in dummies2:
            s = pd.Series(np.zeros(dummies12.shape[0]), copy=False).combine(dummies2[l], max)
        elif l in dummies1 and l not in dummies2:
            s = dummies1[l].combine(pd.Series(np.zeros(dummies12.shape[0]), copy=False), max)
        name = "REC_DGN_" + str(int(l))
        s = s.rename(name)
        dummies12 = pd.concat([dummies12.reset_index(drop=True), s.reset_index(drop=True)], axis=1)

    tx_study = pd.concat([tx_study.reset_index(drop=True), dummies12.reset_index(drop=True)], axis=1)

    del dummies1
    del dummies2
    del dummies12

    tmp = tx_li['REC_FUNCTN_STAT']
    tmp.replace(to_replace=[996.0, 998.0], value=np.nan, inplace=True)
    tmp.replace(to_replace=[2100.0, 2090.0, 2080.0, 2070.0, 1.0], value=0.0, inplace=True)
    tmp.replace(to_replace=[2060.0, 2050.0, 2.0], value=1.0, inplace=True)
    tmp.replace(to_replace=[2040.0, 2030.0, 3.0], value=2.0, inplace=True)
    tmp.replace(to_replace=[2020.0, 2010.0], value=3.0, inplace=True)
    tx_study = pd.concat([tx_study.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)

    for col in ['CAN_RACE_SRTR', 'REC_PRIMARY_PAY']:
        dummy = pd.get_dummies(tx_li[col])
        headers = dummy.columns
        new_headers = []
        for h in headers:
            if type(h) is np.float64:
                h = int(h)
            new_headers.append(col + '_' + str(h))
        dummy.columns = new_headers

        tx_study = pd.concat([tx_study.reset_index(drop=True), dummy.reset_index(drop=True)], axis=1)

    tmp = tx_li['CAN_ETHNICITY_SRTR']
    tmp = tmp.replace("LATINO", 1)
    tmp = tmp.apply(lambda x: 0 if type(x) is str else x)
    tmp = tmp.rename('CAN_ENTHNICITY_SRTR_LATINO')
    tx_study = pd.concat([tx_study.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)

    tx_ignore_mean_fill = ['REC_DGN', 'REC_DGN2', 'PERS_OPTN_DEATH_DT', 'TFL_COD']

    # num_rows = tx_study.shape[0]
    # tmp = tx_study
    # for col in tmp.columns:
    #     if is_numeric_dtype(tmp[col]) and col not in tx_ignore_mean_fill:
    #         tmp[col] = tmp[col].replace(-1, np.nan)
    #
    # #    na_num = tmp[col].isna().sum()
    # #    percent = na_num/num_rows
    # #    percent = 100*percent
    # #    print(f'Percent Missing in {col}: {percent}')
    #     if is_numeric_dtype(tmp[col]) and col not in tx_ignore_mean_fill:
    #         tmp[col] = tmp[col].fillna(tmp[col].mean())

    def fol_cd(row):
        if row['TFL_FOL_CD'] == 6:
            return 0.5
        if row['TFL_FOL_CD'] < 800:
            return row['TFL_FOL_CD']/10

        transplant = pd.to_datetime(row['REC_TX_DT'])
        if row['TFL_FOL_CD'] == 800:
            failure = pd.to_datetime(row['TFL_FAIL_DT'])
            diff = failure - transplant
            return (diff/np.timedelta64(1, 'D'))/365

        death = pd.to_datetime(row['TFL_PX_STAT_DT'])
        if (row['TFL_FOL_CD'] == 998) or (row['TFL_FOL_CD'] == 999):
            diff = death - transplant
            return (diff/np.timedelta64(1, 'D'))/365

    tmp = to_binary(txf_li, 'TFL_DIAB_DURING_FOL', ["Y"], ["N"], [np.nan, "U"])
    tmp = tmp.rename('TFL_DIAB_DURING_FOL')
    txf_study = pd.concat([txf_study.reset_index(drop=True),
        tmp.reset_index(drop=True)], axis=1)

    for c in ["TFL_INSULIN_DEPND"]:
        tmp = to_binary(txf_li, c, ['Y'], ['N'], [np.nan, 'U'])
        txf_study = pd.concat([txf_study.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)

    tmp = to_binary(txf_li, 'TFL_ACUTE_REJ_EPISODE', [1,2], [3], [998])
    txf_study = pd.concat([txf_study.reset_index(drop=True),
        tmp.rename('TFL_ACUTE_REJ_EPISODE').reset_index(drop=True)], axis=1)

    for c in ["TFL_FAIL_BILIARY", "TFL_GRAFT_STAT", "TFL_HOSP", "TFL_IMMUNO_DISCONT",
                  "TFL_MALIG", "TFL_MALIG_LYMPH","TFL_MALIG_RECUR_TUMOR", "TFL_MALIG_TUMOR",
                  "TFL_PX_NONCOMP", "TFL_REJ_TREAT"]:
        tmp = to_binary(txf_li, c, ['Y'], ['N'], ['U'])
        txf_study = pd.concat([txf_study.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)

    tfl_malig_cols = txf_study.columns.str.contains('TFL_MALIG')
    tfl_malig = txf_study[txf_study.columns[tfl_malig_cols]]
    tfl_malig = tfl_malig.replace(to_replace=-1, value=np.nan)
    txf_study['TFL_MALIG_is_na'] = tfl_malig.isna().max(axis=1).apply(lambda x: 0 if x == 0 else 1)

    # Get prescriptions info
    temp = pd.DataFrame()
    temp['TRR_FOL_ID'] = fol_immuno['TRR_FOL_ID']

    sir = fol_immuno.loc[fol_immuno['TFL_IMMUNO_DRUG_CD'].isin([6, 58])]
    temp['sirolimus_prescription'] = sir['TFL_IMMUNO_DRUG_MAINT_CUR']
    temp.groupby('TRR_FOL_ID')['sirolimus_prescription'].max()

    tac = fol_immuno.loc[fol_immuno['TFL_IMMUNO_DRUG_CD'].isin([5, 53, 54, 59])]
    temp['tacrolimus_prescription'] = tac['TFL_IMMUNO_DRUG_MAINT_CUR']
    temp.groupby('TRR_FOL_ID')['tacrolimus_prescription'].max()

    tac = fol_immuno.loc[fol_immuno['TFL_IMMUNO_DRUG_CD'].isin([-2, 46, 48])]
    temp['cyclosporine_prescription'] = tac['TFL_IMMUNO_DRUG_MAINT_CUR']
    temp.groupby('TRR_FOL_ID')['cyclosporine_prescription'].max()

    temp = temp.fillna(0)
    txf_study = txf_study.merge(temp, on='TRR_FOL_ID')

    def series_to_binary(series, pos, neg, na):
        d = {}
        for i in pos:
            d[i] = 1
        for i in neg:
            d[i] = 0
        for i in na:
            d[i] = -1
        tmp = series
        tmp = tmp.replace(d)
        return tmp

    cols = ['TFL_CREAT', 'TFL_REJ_EVENT_NUM','TFL_SGOT','TFL_SGPT','TFL_TOT_BILI','TFL_DIAB_DURING_FOL']

    for col in cols:
        txf_study[col] = txf_study.groupby('TRR_ID')[col].ffill()
        txf_study[col].fillna(0, inplace=True)

    dup = np.intersect1d(tx_study.columns, txf_study.columns).tolist()
    combined_data = tx_study.merge(txf_study, on=dup)

    combined_data = combined_data.sort_values(['TRR_ID', 'TFL_PX_STAT_DT'])

    def time_since_transplant(row):
        transplant = pd.to_datetime(row['REC_TX_DT'])
        fol = pd.to_datetime(row['TFL_PX_STAT_DT'])
        diff = fol - transplant
        return (diff/np.timedelta64(1, 'D'))/365

    def time_to_death(row):
        if pd.isna(row['PERS_OPTN_DEATH_DT']):
            return 100
        fol = pd.to_datetime(row['TFL_PX_STAT_DT'])
        death = pd.to_datetime(row['PERS_OPTN_DEATH_DT'])
        diff = death - fol
        return (diff/np.timedelta64(1, 'D'))/365

    valid_ids = []
    for i in combined_data['TRR_ID'].unique():
        if i not in patients['lived']:
            valid_ids.append(i)
        elif combined_data[combined_data['TRR_ID'] == i].shape[0] > 1:
            valid_ids.append(i)

    combined_data = combined_data[combined_data['TRR_ID'].isin(valid_ids)]

    year_of_transplant = combined_data.apply(lambda x: int(x['REC_TX_DT'][:4]), axis=1)

    tmp = combined_data.apply(lambda x: time_since_transplant(x), axis=1)
    tmp = tmp.rename('time_since_transplant')
    combined_data = pd.concat([combined_data.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)

    tmp = combined_data.apply(lambda x: time_to_death(x), axis=1)
    tmp = tmp.rename('time_to_death')
    combined_data = pd.concat([combined_data.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)

    def label_death(row, yr):
        if pd.isna(row['time_to_death']):
            return 0
        elif row['time_to_death'] < yr:
            return 1
        else:
            return 0

    label1 = combined_data.apply(lambda x: label_death(x, 1), axis=1)
    label1 = label1.rename('label1')
    combined_data = pd.concat([combined_data.reset_index(drop=True), label1.reset_index(drop=True)], axis=1)

    label5 = combined_data.apply(lambda x: label_death(x, 5), axis=1)
    label5 = label5.rename('label5')
    combined_data = pd.concat([combined_data.reset_index(drop=True), label5.reset_index(drop=True)], axis=1)

    trr_ids = combined_data['TRR_ID'].unique()
    ids, split_data = {}, { 'train': {}, 'valid': {}, 'test': {}, 'prob_analysis': {} }
    for class_name in CLASSES:
        ids[class_name] = list(np.intersect1d(patients[class_name], trr_ids))
        split_data['train'][class_name], other = train_test_split(ids[class_name], test_size=0.25, shuffle=True) #75% train
        split_data['valid'][class_name], other = train_test_split(other, test_size=0.60, shuffle=True) #10% valid
        split_data['test'][class_name], split_data['prob_analysis'][class_name] = train_test_split(other, test_size=0.33, shuffle=True) #10% test, 5% prob_analysis
        print(f'{class_name} train: {len(split_data["train"][class_name])}, '
              f'{class_name} valid: {len(split_data["valid"][class_name])}, '
              f'{class_name} test: {len(split_data["test"][class_name])}, '
              f'{class_name} prob_analysis: {len(split_data["prob_analysis"][class_name])}')

    for name, group in split_data.items():
        print(f'all {name}: {len([x for y in group.values() for x in y])}')

    all_data = combined_data.copy()

    drop = ['REC_DGN', 'REC_DGN2', 'PERS_OPTN_DEATH_DT', 'REC_TX_DT', 'TFL_COD',
            'fol_yr', 'TRR_FOL_ID', 'TFL_COD2', 'TFL_COD3', 'TFL_PX_STAT_DT',
            'TFL_FAIL_DT', 'TFL_TXFER_DT', 'time_to_death', 'label1', 'label5',
            'REC_LIFE_SUPPORT']

    for col in drop:
        combined_data.drop(col, axis=1, inplace=True)

    # Only take common features between SRTR and UHN
    if DATASET == 'BOTH':
        combined_data = combined_data[['TRR_ID'] + COMMON_FEATURES]

    # Forward fill null values
    combined_data = combined_data.replace(to_replace=-1, value=np.nan)
    groups = combined_data.groupby('TRR_ID')
    groups.ffill()

    # add ..._is_na cols
    null_cols = combined_data.columns[combined_data.isna().any()].tolist()
    null_cols = list(filter(lambda col: 'TFL_MALIG_' not in col and 'REC_FAIL_' not in col, null_cols))
    for col in null_cols:
        combined_data[f"{col}_is_na"] = combined_data[col].isna().astype(int)

    # normalize cols and set na to min-1
    numeric_cols = combined_data.select_dtypes(include=[np.number]).columns.tolist()
    norm_cols = list(filter(lambda col: col != 'TRR_ID', numeric_cols))
    for col in norm_cols:
        combined_data[col] = combined_data[col].replace(to_replace=-1, value=np.nan)
        combined_data[col] = stats.zscore(combined_data[col], nan_policy='omit')
        combined_data[col] = combined_data[col].fillna(combined_data[col].min()-1.0)

    combined_data.fillna(0, inplace=True)

    dummy_cols = [f'dummy_{x}' for x in range(201 - len(combined_data.columns))]
    combined_data = combined_data.reindex(combined_data.columns.tolist() + dummy_cols, axis=1)

    # 5 year labels
    for class_name in CLASSES:
        if class_name == 'lived':
            combined_data[f'{class_name}_5_yr'] = all_data.apply(lambda x: 0 if x['label1'] or x['label5'] else 1, axis=1)
        else:
            combined_data[f'{class_name}_5_yr'] = all_data.apply(lambda x: 1 if x['label5'] and x['TRR_ID'] in ids[class_name] else 0, axis=1)
        # combined_data[f'{class_name}_5_yr'] = combined_data.groupby('TRR_ID')[f'{class_name}_5_yr'].transform( 'first')

    # 1 year labels
    for class_name in CLASSES:
        if class_name == 'lived':
            combined_data[f'{class_name}_1_yr'] = all_data.apply(lambda x: 0 if x['label1'] else 1, axis=1)
        else:
            combined_data[f'{class_name}_1_yr'] = all_data.apply(lambda x: 1 if x['label1'] and x['TRR_ID'] in ids[class_name] else 0, axis=1)
        # combined_data[f'{class_name}_1_yr'] = combined_data.groupby('TRR_ID')[f'{class_name}_1_yr'].transform( 'first')

    combined_data['dummy'] = 0
    combined_data['time_to_death'] = all_data['time_to_death'].copy()
    combined_data['year_of_transplant'] = year_of_transplant

    groups = combined_data.groupby('TRR_ID')

    print(combined_data)

    combined_data.to_csv('./combined_data.csv', index=False)

    for name, set in split_data.items():
        for class_name in CLASSES:
            index = 0
            dir_path = f'./processed_data/{name}_tensors/{class_name}'
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            for i in set[class_name]:
                df = groups.get_group(i)
                df = df.drop('TRR_ID', axis=1)
                save_path = f'{dir_path}/{str(index)}.pt'
                tensor = torch.tensor(df.values)
                torch.save(tensor, save_path)
                index += 1


    print('complete')

    #for k, v in groups:
    #    df = groups.get_group(k)
    #    df.drop('TRR_ID', axis=1, inplace=True)
    #    path = "processed_data/" + str(index) + ".csv"
    #    df.to_csv(path, index=False)
    #    index += 1

