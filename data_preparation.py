import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import pickle
import numpy as np


class Create_dataset_Buildup:
    def __init__(self):
        pass

    def generate_training_PM(self, n_samples, file_name="dataset_train_PM"):
        treatment_fc = self.random_treatment_PM
        train_df = self.create_samples_PM(n_samples)
        train_df["outcome"] = 0
        case_nrs = train_df["case_nr"].unique()
        for case_nr in case_nrs:
            sample_df = train_df[train_df['case_nr'] == case_nr]
            indices = list(sample_df.index)
            t_idx = treatment_fc(sample_df)
            sample_df_updated = self.treatment_effect_training(sample_df, t_idx)
            train_df.loc[indices, :] = sample_df_updated[list(train_df)].values
        train_df.to_csv(self.path + file_name + ".csv")



    def generate_test_PM(self, n_samples, file_name="dataset_test_PM"):
        df = self.create_samples_PM(n_samples)
        df.rename(columns={"case_nr": "orig_case_nr"}, inplace=True)
        case_lengths = df.groupby("orig_case_nr")["orig_case_nr"].count()

        test_df_length = np.sum((case_lengths.values) * case_lengths)
        test_df = pd.DataFrame(columns=list(df), index=range(test_df_length))
        test_df["case_nr"] = -1
        test_df["outcome_control"] = 0
        test_df["outcome_treatment"] = 0

        orig_case_nrs = df["orig_case_nr"].unique()
        counter = 0
        case_nr = 0
        for orig_case_nr in orig_case_nrs:
            sample_df = df[df['orig_case_nr'] == orig_case_nr]
            for t_idx in range(len(sample_df)):
                indices = np.arange(counter, counter + len(sample_df), 1, int)
                from copy import deepcopy
                sample_df_updated = self.treatment_effect_test(deepcopy(sample_df), t_idx)
                sample_df_updated["case_nr"] = case_nr
                test_df.loc[indices, :] = sample_df_updated[list(test_df)].values
                counter += len(sample_df)
                case_nr += 1
        test_df.to_csv(self.path + file_name + ".csv")

        
        
class Create_dataset_PM_3step:
    def __init__(self):
        pass

    def generate_training_PM(self, n_samples, file_name="dataset_train_PM"):
        treatment_fc = self.random_treatment_PM
        train_df = self.create_samples_PM(n_samples)
        case_nrs = train_df["case_nr"].unique()
        for case_nr in case_nrs:
            sample_df = train_df[train_df['case_nr'] == case_nr]
            indices = list(sample_df.index)
            t_idx = treatment_fc(sample_df)
            if not t_idx is None:
                train_df.loc[indices[t_idx], "treatment"] = 1
            sample_df = train_df[train_df['case_nr'] == case_nr]
            outcome = self.calc_outcome_PM(sample_df, t_idx)
            train_df.loc[indices, "outcome"] = outcome
        train_df.to_csv(self.path + file_name + ".csv")


    def generate_test_PM(self, n_samples, file_name="dataset_test_PM"):
        df = self.create_samples_PM(n_samples)
        df.rename(columns={"case_nr": "orig_case_nr"}, inplace=True)
        case_lengths = df.groupby("orig_case_nr")["orig_case_nr"].count()
        test_df_length = np.sum((case_lengths.values) * case_lengths)
        test_df = pd.DataFrame(columns=list(df), index=range(test_df_length))
        test_df["case_nr"] = -1

        orig_case_nrs = df["orig_case_nr"].unique()
        counter = 0
        case_nr = 0
        for orig_case_nr in orig_case_nrs:
            sample_df = df[df['orig_case_nr'] == orig_case_nr]
            for t_idx in range(len(sample_df)):
                indices = np.arange(counter, counter + len(sample_df), 1, int)
                test_df.loc[indices, list(df)] = sample_df.values
                test_df.loc[indices, "case_nr"] = case_nr
                control_outcome = self.calc_outcome_PM(sample_df, len(sample_df))
                test_df.loc[indices, "outcome_control"] = control_outcome
                test_df.loc[indices[t_idx], "treatment"] = 1
                updated_sample_df = test_df[test_df['case_nr'] == case_nr]
                treatment_outcome = self.calc_outcome_PM(updated_sample_df, t_idx)
                test_df.loc[indices, "outcome_treatment"] = treatment_outcome
                counter += len(sample_df)
                case_nr += 1
        test_df.to_csv(self.path + file_name + ".csv")


    def generate_df_sample(self, case_vars, process, case_cols, process_cols):
        df = pd.DataFrame(index=list(range(len(process))), columns=case_cols+process_cols+["treatment"])
        df.loc[:, case_cols] = case_vars
        df.loc[:, process_cols] = process
        return df
