import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

from Code.models import LSTM, CNN
from Code.data_preparation import Datapreparation
from .utils import EarlyStopping
from IPython.display import clear_output
from matplotlib import pyplot as plt
from .loss_functions import mse_loss
from IPython import get_ipython
from Code.data_generation import Threesteps, Process_02


class RL():
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
        self.earlystopfile = self.path + self.filename + "_ES_" + str(np.random.randint(0, 10000)) + '.pt'


        self.transition = namedtuple('Transition', ('state_case', 'state_proc', 'state_t', 'action',
                                                    'next_state_case', 'next_state_proc', 'next_state_t', 'reward'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.process_cols_orig = deepcopy(self.process_cols)

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
        self.penalty = process_params["PENALTY"]


    def read_process_params_Process_02(self, process_params):
        self.prob_B_C = process_params["PROB_B_C"]
        self.multip_B_C = process_params["MULTIP_BC"]
        self.values_D = process_params["VALUES_D"]
        self.treatment_cost = process_params["TREATMENT_COST"]
        self.penalty = process_params["PENALTY"]


    def read_process_params(self, process_params):
        self.prob_B_C = process_params["PROB_B_C"]
        self.values_D = process_params["VALUES_D"]
        self.treatment_cost = process_params["TREATMENT_COST"]
        self.penalty = process_params["PENALTY"]


    def live_plot(self):
        if self.steps_done >= self.batch_size - 1:
            figsize = (7, 5)
            clear_output(wait=True)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            losses_t = torch.tensor(self.losses, dtype=torch.float)  # + 1e-10
            plt.title('Training...')
            plt.xlabel('Transitions')
            ax.set_ylabel('Losses')
            ax.plot(losses_t, label="losses", c="tab:blue")
            # Take 100 episode averages and plot them too
            if len(losses_t) >= 100:
                means = losses_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                ax.plot(means.numpy(), label="moving average losses", c="tab:orange")
            # Show policy results on second y-axis
            ax2 = ax.twinx()
            ax2.plot(self.val_results / self.godmax_val, label="policy val_results", c="tab:green")
            ax2.set_ylabel("validation score (share of omniscient optimal)")
            ax2.vlines(self.best_step, ymin=0, ymax=1, colors="tab:red")
            # Finetuning
            if not self.aleatoric:
                ax.set_yscale("log")
                ax.set_ylabel('Losses (log scale)')
            plt.grid(True)
            fig.legend(loc="upper right")
            plt.show();


    def get_data_ready_PM(self, new_trainset=True, new_testset=True):
        # TO BE OPTIMIZED
        # the training set is generated with the sole purpose to create a onehot-encoder and scaler
        if self.dataset == Threesteps:
            self.env = self.dataset(self.path, self.length, self.activities, self.prob_events, self.attrib_values, self.penalty)
        elif self.dataset == Process_02:
            self.env = self.dataset(self.path, self.prob_B_C, self.multip_B_C, self.values_D, self.treatment_cost, self.penalty)
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


    def get_data_RL(self):
        df = pd.read_csv(self.path + self.testfile + ".csv")
        df["ITE"] = df["outcome_treatment"] - df["outcome_control"]
        df_val = df[:int(len(df) * self.val_share)]
        df_pos = df_val[df_val["ITE"] > 0]
        self.godmax_val = df_pos.groupby(["orig_case_nr"])["ITE"].max().sum()
        df_pos = df[df["ITE"] > 0]
        self.godmax = df_pos.groupby(["orig_case_nr"])["ITE"].max().sum()
        print(f"Omniscient optimal result for test set / validation set: {self.godmax} / {self.godmax_val}")
        self.df = df[df["treatment"] == 1]
        self.df_val = df_val[df_val["treatment"] == 1]


    def read_training_params(self, training_params):
        self.batch_size = training_params["BATCH_SIZE"]
        self.gamma = training_params["GAMMA"]
        self.eps_start = training_params["EPS_START"]
        self.eps_end = training_params["EPS_END"]
        self.eps_decay = training_params["EPS_DECAY"]
        self.memory_size = training_params["MEMORY_SIZE"]
        self.target_update = training_params["TARGET_UPDATE"]
        self.calc_val = training_params["CALC_VAL"]
        self.aleatoric = training_params["ALEATORIC"]
        self.earlystop = training_params["EARLY_STOP"]
        self.es_patience = training_params["ES_PATIENCE"]
        self.es_delta = training_params["ES_DELTA"]
        self.optimizer = training_params["OPTIMIZER"]


    def preprocess_X_PM(self, state_df, state_t):
        X_case, X_process = self.DPP.preprocess_sample_X(state_df, self.length)
        X_t = [0] * self.length
        X_t[-len(state_t):] = state_t
        X_t = torch.unsqueeze(torch.Tensor(X_t), 0)
        X_case = X_case.to(self.device)
        X_process = X_process.to(self.device)
        X_t = X_t.to(self.device)
        return X_case, X_process, X_t


    def preprocess_reward(self, reward):
        if self.scale_y:
            return self.DPP.scaler_y.transform(torch.Tensor([[reward]]))  # .to(device=self.device)
        else:
            return [[reward]]


    def select_action(self, net, state_preproc_case, state_preproc_process, state_preproc_t, exploit=False):
        sample = random.random()
        if exploit:
            eps_threshold = 0
        else:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                            math.exp(-1. * self.steps_done / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                net.eval()
                result, _ = net(state_preproc_case, state_preproc_process, state_preproc_t)
                net.train()
                return result.max(1)[1].view(1, 1)  # t.max(1) will return largest column value of each row.
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)


    class ReplayMemory(object):
        def __init__(self, capacity, transition):
            self.memory = deque([], maxlen=capacity)
            self.transition = transition

        def push(self, *args):
            self.memory.append(self.transition(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)


    def optimize_model(self):
        loss = 0
        if len(self.memory) < self.batch_size:
            return loss
        transitions = self.memory.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state_case)), device=self.device, dtype=torch.bool)
        non_final_next_states_case = torch.cat([s for s in batch.next_state_case
                                                if s is not None])
        non_final_next_states_proc = torch.cat([s for s in batch.next_state_proc
                                                if s is not None])
        non_final_next_states_t = torch.cat([s for s in batch.next_state_t
                                             if s is not None])

        state_batch_case = torch.cat(batch.state_case)
        state_batch_proc = torch.cat(batch.state_proc)
        state_batch_t = torch.cat(batch.state_t)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_forward_pass = self.policy_net(state_batch_case, state_batch_proc,
                                              state_batch_t)
        state_action_values = state_forward_pass[0].gather(1, action_batch)  # = Q(s,a)
        state_action_logvars = state_forward_pass[1]

        preproc_zero = torch.tensor(self.preprocess_reward(0), device=self.device)[0].float()
        next_state_values = torch.ones(self.batch_size, device=self.device) * preproc_zero
        next_state_values[non_final_mask] = self.target_net(non_final_next_states_case, non_final_next_states_proc,
                                                            non_final_next_states_t)[0].max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # new
        #expected_state_action_values = (expected_state_action_values - self.DPP.scaler_y_mean) / self.DPP.scaler_y_std


        if self.aleatoric:
            loss = mse_loss(state_action_values, expected_state_action_values.unsqueeze(1),
                            logvar=state_action_logvars, aleatoric=self.aleatoric, device=self.device)
        else:
            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if not param.grad is None:  # CHECK reason
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss


    def calc_RL_results_PM(self, net, df, treatment_until=3):
        df.reset_index(inplace=True, drop=True)
        current_case = -1
        self.outcomes, self.actions = [], []
        for index, row in df.iterrows():
            if row["orig_case_nr"] == current_case:
                running_df = df[(df.index > index - counter - 1) & (df.index <= index)]
                state_t.append(action)
                counter += 1
                #if len(running_df) > treatment_until:
                #    treatment_done = True
            else:
                current_case = row["orig_case_nr"]
                running_df = df[df.index == index]
                state_t = [0]
                counter = 1
                treatment_done = False
            action = 0
            if treatment_done is False:
                state_preproc_case, state_preproc_process, state_preproc_t = self.preprocess_X_PM(running_df, state_t)
                action = int(self.select_action(net, state_preproc_case, state_preproc_process, state_preproc_t, exploit=True))
            self.actions.append(action)
            if action == 1:
                self.outcomes.append(row["ITE"])
                treatment_done = True
            else:
                self.outcomes.append(0)   #ok because always same for whole case
        return sum(self.outcomes)


    def calc_RL_results_testset_PM(self):
        outcome = self.calc_RL_results_PM(self.final_net, self.df)
        self.df["actions"] = self.actions
        self.df["outcomes"] = self.outcomes
        return outcome, self.df


    def initialize_LSTM(self, model_params):
        self.nr_lstm_layers = model_params["NR_LSTM_LAYERS"]
        self.lstm_size = model_params["LSTM_SIZE"]
        self.generic_pre_initialisation(model_params)

        self.policy_net = LSTM(input_size_case=self.input_size_case, input_size_process=self.input_size_process,
                               nr_outputs=self.n_actions, nr_lstm_layers=self.nr_lstm_layers,
                               lstm_size=self.lstm_size, nr_dense_layers=self.nr_dense_layers,
                               dense_width=self.dense_width, p=self.p)
        self.target_net = LSTM(input_size_case=self.input_size_case, input_size_process=self.input_size_process,
                               nr_outputs=self.n_actions, nr_lstm_layers=self.nr_lstm_layers,
                               lstm_size=self.lstm_size, nr_dense_layers=self.nr_dense_layers,
                               dense_width=self.dense_width, p=self.p)
        self.generic_post_initialisation()


    def initialize_CNN(self, model_params):
        self.nr_cnn_layers = model_params["NR_CNN_LAYERS"]
        self.nr_out_channels = model_params["NR_OUT_CHANNELS"]
        self.kernel_size = model_params["KERNEL_SIZE"]
        self.stride = model_params["STRIDE"]
        self.generic_pre_initialisation(model_params)

        self.policy_net = CNN(input_size_case=self.input_size_case, input_size_process=self.input_size_process,
                              length=self.length, nr_out_channels=self.nr_out_channels,
                              nr_cnn_layers=self.nr_cnn_layers, kernel_size=self.kernel_size, stride=self.stride,
                              nr_dense_layers=self.nr_dense_layers, dense_width=self.dense_width,
                              p=self.p, nr_outputs=self.n_actions)
        self.target_net = CNN(input_size_case=self.input_size_case, input_size_process=self.input_size_process,
                              length=self.length, nr_out_channels=self.nr_out_channels,
                              nr_cnn_layers=self.nr_cnn_layers, kernel_size=self.kernel_size, stride=self.stride,
                              nr_dense_layers=self.nr_dense_layers, dense_width=self.dense_width,
                              p=self.p, nr_outputs=self.n_actions)
        self.generic_post_initialisation()


    def generic_pre_initialisation(self, model_params):
        self.input_size_case = model_params["INPUT_SIZE_CASE"]
        self.input_size_process = model_params["INPUT_SIZE_PROCESS"]
        self.nr_dense_layers = model_params["NR_DENSE_LAYERS"]
        self.dense_width = model_params["DENSE WIDTH"]
        self.p = model_params["P"]
        self.n_actions = self.env.nr_treatments


    def generic_post_initialisation(self):
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = self.optimizer(self.policy_net.parameters())
        #self.optimizer = optim.Adam(self.policy_net.parameters(), amsgrad=True)
        self.memory = self.ReplayMemory(self.memory_size, self.transition)
        self.losses = []
        self.val_result = 0
        self.val_results = []
        self.steps_done = 0  # used for exploration
        if self.earlystop:
            self.early_stopping = EarlyStopping(patience=self.es_patience, verbose=False, delta=self.es_delta,
                                                path=self.earlystopfile)
        self.best_val_result, self.best_step = 0, 0


    def run_PM(self, num_episodes):
        self.num_episodes = num_episodes

        self.policy_net.train()

        for i_episode in range(self.num_episodes):
            # Initialize first state
            self.env.create_case_PM()
            case_df, _ = self.env.create_event_PM(action=0)
            state_t = [0]
            #print("______")
            #print("original case:")
            #print(self.env.case)
            #print(case_df)
            state_preproc_case, state_preproc_process, state_preproc_t = self.preprocess_X_PM(case_df, state_t)
            for t in count():
                # Select and perform an action
                action = self.select_action(self.policy_net, state_preproc_case, state_preproc_process, state_preproc_t)
                self.steps_done += 1
                state_t.append(int(action))
                t_idx = None
                if action == 1:
                    t_idx = t
                next_case_df, terminal = self.env.create_event_PM(action=action)
                #print("action:", action)
                #print("terminal:", terminal)
                #print("t_idx/state_t:", t_idx, state_t)
                #print("next prefix:")
                #print(next_case_df)
                #print("case becomes:")
                next_state_preproc_case, next_state_preproc_process, next_state_preproc_t = \
                    self.preprocess_X_PM(next_case_df, state_t)
                reward = self.env.calc_reward_PM(t_idx, state_t)
                #print(self.env.case)
                #print("reward:", reward)
                #print("-> next step")
                reward_preproc = torch.tensor(self.preprocess_reward(reward), device=self.device)[0]
                #reward_preproc = torch.tensor([reward], device=self.device)

                # Observe the new state
                if terminal or reward <= self.penalty:
                    next_state_preproc_case, next_state_preproc_process, next_state_preproc_t = None, None, None
                    #print("terminal! or penalty")

                # Store the transition in memory
                self.memory.push(state_preproc_case, state_preproc_process, state_preproc_t, action,
                                 next_state_preproc_case,
                                 next_state_preproc_process, next_state_preproc_t, reward_preproc)

                # Move to the next state
                #case_df = next_case_df
                state_preproc_t = deepcopy(next_state_preproc_t)
                state_preproc_case = deepcopy(next_state_preproc_case)
                state_preproc_process = deepcopy(next_state_preproc_process)

                # Perform one step of the optimization (on the policy network)
                loss = self.optimize_model()
                self.losses.append(loss)
                self.val_results.append(self.val_result)
                if terminal or reward <= self.penalty:
                    #print("terminal! or penalty")
                    if type(get_ipython()).__module__.startswith('ipykernel.'):
                        self.live_plot()
                    break
                #print("len state_t:", len(state_t))
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if (self.steps_done > self.batch_size) and (i_episode % self.calc_val == 0):
                self.val_result = self.calc_RL_results_PM(self.policy_net, self.df_val)
                if self.val_result > self.best_val_result:
                    self.best_val_result = self.val_result
                    self.best_step = len(self.losses) - 1
                if self.earlystop:
                    self.early_stopping(-self.val_result, self.policy_net)
                    #if self.early_stopping.early_stop:
                    #    print("\nEarly stopping at episode {}".format(i_episode))
        self.final_net = deepcopy(self.policy_net)
        print('Complete')

    def load_best_model(self):
        self.final_net = deepcopy(self.policy_net)
        self.final_net.load_state_dict(torch.load(self.earlystopfile))

