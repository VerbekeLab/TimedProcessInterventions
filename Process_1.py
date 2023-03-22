from Code.data_preparation import Create_dataset_Buildup, Create_dataset_PM_3step
from pm4py.objects.petri.petrinet import PetriNet
from pm4py.objects.petri import utils
from pm4py.objects.petri.petrinet import Marking
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.conversion.log import converter
from pm4py.visualization.petrinet import visualizer as pn_visualizer
import numpy as np

class Threesteps(Create_dataset_PM_3step):
    def __init__(self, path,
            length=7,
            activities = ["A", "B"],
            prob_events = [.3, .7],
            attrib_values = [0, 1, 2],
            penalty = -100):
        self.path = path
        self.length = length
        self.activities = activities
        self.prob_events = prob_events
        self.attrib_values = attrib_values
        self.penalty = penalty
        self.nr_treatments = 2 # (0 and 1)
        self.case_vars = []
        self.process_vars = ["f0", "f1"]
        self.create_PN()


    def create_PN(self):
        self.net = PetriNet("Petri_net")

        source = PetriNet.Place("source")
        sink = PetriNet.Place("sink")

        p_21 = PetriNet.Place("p_21")
        p_41 = PetriNet.Place("p_41")

        self.net.places.add(source)
        self.net.places.add(sink)
        self.net.places.add(p_21)
        self.net.places.add(p_41)

        t_A = PetriNet.Transition("name_A", "dummy")
        t_B = PetriNet.Transition("name_B", "dummy")
        t_E = PetriNet.Transition("name_E", "dummy")

        self.net.transitions.add(t_A)
        self.net.transitions.add(t_B)
        self.net.transitions.add(t_E)

        utils.add_arc_from_to(source, t_A, self.net)
        utils.add_arc_from_to(t_A, p_21, self.net)
        utils.add_arc_from_to(p_21, t_B, self.net)
        utils.add_arc_from_to(t_B, p_41, self.net)

        utils.add_arc_from_to(p_41, t_E, self.net)

        utils.add_arc_from_to(t_E, sink, self.net)

        self.initial_marking = Marking()
        self.initial_marking[source] = 1
        self.final_marking = Marking()
        self.final_marking[sink] = 2


    def vizualize_net(self):
        gviz = pn_visualizer.apply(self.net, self.initial_marking, self.final_marking)
        pn_visualizer.view(gviz)


    def create_samples_PM(self, max_nr=5):
        simulated_log = simulator.apply(self.net, self.initial_marking, variant=simulator.Variants.BASIC_PLAYOUT,
                                        parameters={
                                            simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: max_nr})
        df = converter.apply(simulated_log, variant=converter.Variants.TO_DATA_FRAME)
        df.columns = ['Dummy', 'timestamp', 'case_nr']
        df.drop(["Dummy", "timestamp"], axis=1, inplace=True)
        df["treatment"] = 0
        df["f1"] = 0
        df["f0"] = "UNK"
        df["f1"].astype(float)

        for row_nr, row in df.iterrows():
            df.loc[row_nr, "f0"] = np.random.choice(self.activities,  p=self.prob_events)
            df.loc[row_nr, "f1"]  = np.random.choice(self.attrib_values)

        return df


    def random_treatment_PM(self, df):
        t_idx = np.random.randint(0, high=len(df)+1)
        if t_idx == len(df):
            # no treatment, all control case
            t_idx = None
        return t_idx


    def create_case_PM(self):
        self.case = self.create_samples_PM(max_nr=1)
        self.case_length = len(self.case)
        self.counter = 0
        self.terminal = False


    def create_event_PM(self, action=0):
        if self.counter == self.case_length:
            self.terminal = True
        self.counter += 1
        return self.case[:self.counter], self.terminal


    def calc_reward_PM(self, t_idx, state_t):
        reward = 0
        if t_idx and sum(state_t) > 1:
            reward = self.penalty
        else:
            if len(state_t) - 1 == self.case_length:
                if sum(state_t) > 0:
                    t_idx = int(np.argwhere(state_t)[0] - 1)
                else:
                    t_idx = self.length
                reward = self.calc_decide_PM(self.case, t_idx)
        return reward


    def calc_outcome_PM(self, df, t_idx):
        outcome = self.calc_decide_PM(df, t_idx)
        return outcome


    def calc_decide_PM(self, df, t_idx):
        df.reset_index(inplace=True, drop=True)
        #print(df)
        #print("t_idx:", t_idx)
        outcome = 0
        if (not t_idx is None) and (t_idx < len(df)):
            # treatment
            outcome = df.loc[t_idx, "f1"]#.astype(float)
            if np.sum(df.loc[:, "f0"].values == "A") == 0:
                outcome *= - 2 ##########NEW 31-8-2022#######################, was -1
            #else:
            #    outcome *= 2  ##########NEW 31-8-2022#######################
        #print("outcome:", outcome)
        return np.float32(outcome)
