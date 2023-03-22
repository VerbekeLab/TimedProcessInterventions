
from Code.data_preparation import Datapreparation, PrepareData
from Code.inference import Predict
from Code.training import Train_model
from Code.data_generation import Threesteps, Process_02
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
#import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

from Code.models import LSTM, CNN
#from Code.data_preparation import Datapreparation

from IPython.display import clear_output
from matplotlib import pyplot as plt



class ITE_sim(PrepareData, Datapreparation, Train_model, Predict):
    def __init__(self, dataset_params, process_params):
        self.dataset = dataset_params["DATASET"]
        self.filename = dataset_params["FILENAME"]
        self.path = dataset_params["PATH"]
        self.trainsize = dataset_params["TRAINSIZE"]
        self.testsize = dataset_params["TESTSIZE"]
        self.val_share = dataset_params["VAL_SHARE"]
        self.case_cols = dataset_params["CASE_COLS"]
        self.process_cols = dataset_params["PROCESS_COLS"]
        self.cat_cols = dataset_params["CAT_COLS"]
        self.scale_x_cols = dataset_params["SCALE_X_COLS"]
        self.scale_y = dataset_params["SCALE_Y"]
        self.create_prefixes = dataset_params["CREATE_PREFIXES"]
        self.length = dataset_params["LENGTH"]

        self.trainfile = self.filename + "_train_" + str(self.trainsize)
        self.testfile = self.filename + "_test_" + str(self.testsize)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nr_iters = 1

        self.case_cols_orig = deepcopy(self.case_cols)
        self.process_cols_orig = deepcopy(self.process_cols)

        self.ITE_preds = {}  # collects the predictions
        self.Y_T_preds = {}  # collects the predictions
        self.Y_C_preds = {}  # collects the predictions
        self.ITE_pred_future = {} # collects future forecasts over iterations
        self.ITE_unc_future = {} # collects future uncertainties over iterations
        self.uncertainties_ITE = {}  # collects the uncertainties
        self.uncertainties_T = {}  # collects the uncertainties
        self.uncertainties_C = {}  # collects the uncertainties

        if self.dataset == Threesteps:
            self.read_process_params_Threesteps(process_params)
        elif self.dataset == Process_02:
            self.read_process_params_Process_02(process_params)
        else:
            print(f"Process '{self.dataset}' unknown")


    def read_process_params_Threesteps(self, process_params):
        self.activities = process_params["ACTIVITIES"]
        self.prob_events = process_params["PROB_EVENTS"]
        self.attrib_values = process_params["ATTRIB_VALUES"]


    def read_process_params_Process_02(self, process_params):
        self.prob_B_C = process_params["PROB_B_C"]
        self.values_D = process_params["VALUES_D"]
        self.treatment_cost = process_params["TREATMENT_COST"]
        self.multip_B_C = process_params["MULTIP_BC"]


    def read_process_params(self, process_params):
        self.prob_B_C = process_params["PROB_B_C"]
        self.values_D = process_params["VALUES_D"]
        self.treatment_cost = process_params["TREATMENT_COST"]


    def get_data_ready_PM(self, new_trainset=True, new_testset=True):
        # TO BE OPTIMIZED
        # the training set is generated with the sole purpose to create a onehot-encoder and scaler
        if self.dataset == Threesteps:
            self.env = self.dataset(self.path, self.length, self.activities, self.prob_events, self.attrib_values)
        elif self.dataset == Process_02:
            self.env = self.dataset(self.path, self.prob_B_C, self.multip_B_C, self.values_D, self.treatment_cost)
        else:
            pass

        if new_trainset:
            self.env.generate_training_PM(n_samples=self.trainsize, file_name=self.trainfile)
        if new_testset:
            self.env.generate_test_PM(n_samples=self.testsize, file_name=self.testfile)

        self.DPP = Datapreparation(path=self.path, case_cols=self.case_cols, process_cols=deepcopy(self.process_cols),
                                   cat_cols=deepcopy(self.cat_cols),
                                   scale_x_cols=self.scale_x_cols, scale_y=self.scale_y)
        self.DPP.preprocess_train(self.trainfile, prefix=self.create_prefixes)
        self.DPP.preprocess_test(self.testfile, prefix=self.create_prefixes)

        self.columns = list(pd.read_csv(self.path + self.testfile + ".csv", nrows=1))


    def get_data_ITE(self):
        self.init_prepare_data(self.path, self.trainfile, self.testfile)


    def read_training_params(self, training_params):
        self.batch_size = training_params["BATCH_SIZE"]
        self.calc_val = training_params["CALC_VAL"]
        self.earlystop = training_params["EARLY_STOP"]
        self.es_patience = training_params["ES_PATIENCE"]
        self.es_delta = training_params["ES_DELTA"]
        self.nb_epochs = training_params["NB_EPOCHS"]
        self.aleatoric = training_params["ALEATORIC"]
        self.verbose = training_params["VERBOSE"]
        self.nr_future_est = training_params["NR_FUTURE_EST"]



    def preprocess_X(self, case_vars, state, state_t):
        state_df = self.env.generate_df_sample(case_vars, state, self.case_cols_orig, self.process_cols_orig)
        X_case, X_process = self.DPP.preprocess_sample_X(state_df, self.length)

        X_t = [0] * self.length
        X_t[-len(state_t):] = state_t
        X_t = torch.unsqueeze(torch.Tensor(X_t), 0)

        X_case = X_case.to(self.device)
        X_process = X_process.to(self.device)
        X_t = X_t.to(self.device)
        return X_case, X_process, X_t


    def initialize_LSTM(self, model_params):
        self.input_size_case = model_params["INPUT_SIZE_CASE"]
        self.input_size_process = model_params["INPUT_SIZE_PROCESS"]
        self.nr_lstm_layers = model_params["NR_LSTM_LAYERS"]
        self.lstm_size = model_params["LSTM_SIZE"]
        self.nr_dense_layers = model_params["NR_DENSE_LAYERS"]
        self.dense_width = model_params["DENSE WIDTH"]
        self.p = model_params["P"]
        self.n_actions = self.env.nr_treatments
        self.model = LSTM(input_size_case=self.input_size_case, input_size_process=self.input_size_process,
                               nr_outputs=self.n_actions, nr_lstm_layers=self.nr_lstm_layers,
                               lstm_size=self.lstm_size, nr_dense_layers=self.nr_dense_layers,
                               dense_width=self.dense_width, p=self.p)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True)
        self.losses = []
        self.val_result = 0
        self.val_results = []
        self.steps_done = 0  # used for exploration
        self.model_type = LSTM
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"model contains {pytorch_total_params} trainable parameters")


    def initialize_CNN(self, model_params):
        self.input_size_case = model_params["INPUT_SIZE_CASE"]
        self.input_size_process = model_params["INPUT_SIZE_PROCESS"]
        self.nr_cnn_layers = model_params["NR_CNN_LAYERS"]
        self.nr_out_channels = model_params["NR_OUT_CHANNELS"]
        self.kernel_size = model_params["KERNEL_SIZE"]
        self.stride = model_params["STRIDE"]
        self.nr_dense_layers = model_params["NR_DENSE_LAYERS"]
        self.dense_width = model_params["DENSE WIDTH"]
        self.p = model_params["P"]
        self.n_actions = self.env.nr_treatments

        self.model = CNN(input_size_case=self.input_size_case, input_size_process=self.input_size_process,
                              length=self.length, nr_out_channels=self.nr_out_channels,
                              nr_cnn_layers=self.nr_cnn_layers, kernel_size=self.kernel_size, stride=self.stride,
                              nr_dense_layers=self.nr_dense_layers, dense_width=self.dense_width,
                              p=self.p, nr_outputs=self.n_actions)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), amsgrad=True)
        self.losses = []
        self.val_result = 0
        self.val_results = []
        self.steps_done = 0  # used for exploration
        self.model_type = CNN
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"model contains {pytorch_total_params} trainable parameters")


    def run(self):
        for iter in range(self.nr_iters):
            print("iteration {}/{}".format(iter, self.nr_iters - 1))
            counter = 0

            # data are read
            #self.get_data(self.seq_len, self.val_share)
            self.get_data(seq_len=self.length, val_share=self.val_share, device=self.device)

            self.train_model()
            self.predict()
            if self.nr_future_est:
                self.predict_future()
            if iter == 0:
                self.ITE_preds[counter] = self.ITE_pred / self.nr_iters
                self.Y_T_preds[counter] = self.Y_pred_T / self.nr_iters
                self.Y_C_preds[counter] = self.Y_pred_C / self.nr_iters
                self.uncertainties_ITE[counter] = self.unc_ITE / self.nr_iters
                self.uncertainties_T[counter] = self.unc_T / self.nr_iters
                self.uncertainties_C[counter] = self.unc_C / self.nr_iters
                if self.nr_future_est:
                    self.ITE_pred_future[counter] = self.ITE_pred_fut
                    self.ITE_unc_future[counter] = self.ITE_unc_fut
            else:
                self.ITE_preds[counter] += self.ITE_pred / self.nr_iters
                self.Y_T_preds[counter] += self.Y_pred_T / self.nr_iters
                self.Y_C_preds[counter] += self.Y_pred_C / self.nr_iters
                self.uncertainties_ITE[counter] += self.unc_ITE / self.nr_iters
                self.uncertainties_T[counter] += self.unc_T / self.nr_iters
                self.uncertainties_C[counter] += self.unc_C / self.nr_iters
            counter += 1


    def show_single_state(self, state_case, state_proc, threshold):
        self.model.eval()
        state_proc = np.array([np.array(x) for x in state_proc])
        state_t_C = torch.tensor([0] * self.length).to(self.device)
        state_t_T = torch.tensor([1] * self.length).to(self.device)
        state_preproc_case, state_preproc_process, _ = self.preprocess_X(state_case, np.array(state_proc),
                                                                         [0])
        with torch.no_grad():
            pred_C, logvar_C = self.model(state_preproc_case, state_preproc_process, state_t_C)
            pred_C = pred_C.detach().cpu().numpy()[0]
            pred_C = self.DPP.scaler_y.inverse_transform(pred_C)[0]

            pred_T, logvar_T = self.model(state_preproc_case, state_preproc_process, state_t_T)
            pred_T = pred_T.detach().cpu().numpy()[0]
            pred_T = self.DPP.scaler_y.inverse_transform(pred_T)[0]
        print(pred_T, pred_C)
        pred_ITE = pred_T - pred_C
        if pred_ITE > threshold:
            choice = "action"
        else:
            choice = "non-action"
        print(f"predicted ITE: {pred_ITE}  --> choice is {choice}")
