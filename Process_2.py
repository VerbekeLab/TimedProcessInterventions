

class Process_02(Create_dataset_Buildup):
    # treatment cost immediately when treatment is applied
    def __init__(self, path,
                 prob_B_C=[.3, .7],
                 multip_B_C = [2, -3],
                 values_D = [1, 3],
                 treatment_cost = -5,
                 penalty=-100):
        self.path = path
        self.prob_B_C = prob_B_C
        self.multip_B_C = multip_B_C
        self.values_D = values_D
        self.treatment_cost = treatment_cost
        self.penalty = penalty
        self.nr_treatments = 2  # (0 and 1)
        self.create_PN()

    def create_PN(self):
        self.net = PetriNet("Petri_net")

        source = PetriNet.Place("source")
        sink = PetriNet.Place("sink")

        p_21 = PetriNet.Place("p_21")
        p_22 = PetriNet.Place("p_22")
        p_41 = PetriNet.Place("p_41")

        self.net.places.add(source)
        self.net.places.add(sink)
        self.net.places.add(p_21)
        self.net.places.add(p_22)
        self.net.places.add(p_41)

        t_A = PetriNet.Transition("name_A", "A")
        t_B= PetriNet.Transition("name_B", "B")
        t_C = PetriNet.Transition("name_C", "C")
        t_D1 = PetriNet.Transition("name_D", "D1")
        t_D2 = PetriNet.Transition("name_E", "D2")

        self.net.transitions.add(t_A)
        self.net.transitions.add(t_B)
        self.net.transitions.add(t_C)
        self.net.transitions.add(t_D1)
        self.net.transitions.add(t_D2)

        utils.add_arc_from_to(source, t_A, self.net)
        utils.add_arc_from_to(t_A, p_21, self.net)
        utils.add_arc_from_to(t_A, p_22, self.net)
        utils.add_arc_from_to(p_21, t_D1, self.net)
        utils.add_arc_from_to(p_22, t_B, self.net)
        utils.add_arc_from_to(p_22, t_C, self.net)
        utils.add_arc_from_to(t_D1, p_41, self.net)


        utils.add_arc_from_to(p_41, t_D2, self.net)

        utils.add_arc_from_to(t_D2, sink, self.net)
        utils.add_arc_from_to(t_B, sink, self.net)
        utils.add_arc_from_to(t_C, sink, self.net)

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
        df.columns = ['activity', 'timestamp', 'case_nr']
        df.drop(["timestamp"], axis=1, inplace=True)
        treatment = 0
        value = 0

        df["amount"] = 0
        case_nrs = df["case_nr"].unique()
        for case_nr in case_nrs:
            mask = df["case_nr"] == case_nr
            df.loc[mask, "amount"] = np.random.random_integers(low=1, high=10)

        for row_nr, row in df.iterrows():
            if row["activity"] == "D1":
                df.loc[row_nr, "value"] = np.random.random_integers(low=1, high=self.values_D[0] + 1)
            elif row["activity"] == "D2":
                df.loc[row_nr, "value"] = np.random.random_integers(low=1, high=self.values_D[0] + 2)
            else:
                df.loc[row_nr, "value"] = value
            # we override the simulation, which assumes 50/50 probabilities for B and C
            if row["activity"] in ["B", "C"]:
                df.loc[row_nr, "activity"] = np.random.choice(["B", "C"], p=self.prob_B_C)
            df.loc[row_nr, "treatment"] = treatment

        return df


    def random_treatment_PM(self, df):
        idx = np.random.randint(0, high=len(df)+1)
        if idx == len(df):
            idx = None
        return idx


    def treatment_effect_training(self, df, t_idx):
        df = self.calc_treatment_effect(df, t_idx)
        # compute outcome
        df.loc[:, "outcome"] = self.calc_outcome_PM(df, t_idx)
        return df


    def treatment_effect_test(self, df, t_idx):
        # compute outcomes
        df.loc[:, "outcome_control"] = self.calc_outcome_PM(df, None)
        # perform treatment
        df = self.calc_treatment_effect(df, t_idx)
        df.loc[:, "outcome_treatment"] = self.calc_outcome_PM(df, t_idx)
        return df.copy(deep=True)


    def calc_treatment_effect(self, df, t_idx):  # GEEN COPY SLICE
        df.reset_index(inplace=True, drop=True)
        from copy import deepcopy
        df = deepcopy(df)
        # include treatment in dataset
        if not t_idx is None:
            df.loc[t_idx, "treatment"] = 1
            # compute effect treatment on dataset
            if len(df[df["activity"] == "B"]) > 0:
                multiplicator = self.multip_B_C[0]
            else:
                multiplicator = self.multip_B_C[1]
            D1_idx, D2_idx = df[df["activity"] == "D1"].index[0], df[df["activity"] == "D2"].index[0]
            if t_idx < D1_idx:
                new_value = df.loc[D1_idx, "value"] * multiplicator
                df.loc[D1_idx, "value"] = deepcopy(new_value)
            else:
                if t_idx < D2_idx:
                    new_value = df.loc[D2_idx, "value"] * multiplicator
                    df.loc[D2_idx, "value"] = deepcopy(new_value)
        return df


    def create_case_PM(self):
        self.case = self.create_samples_PM(max_nr=1)
        self.case_length = len(self.case)
        self.counter = 0
        self.terminal = False


    def create_event_PM(self, action=0):
        if action == 1:
            self.case = self.calc_treatment_effect(self.case, self.counter - 1).copy(deep=True)
        if self.counter == self.case_length:
            #print("self.counter == self.case_length -> terminal!")
            self.terminal = True
        self.counter += 1
        return self.case[:self.counter], self.terminal


    def calc_reward_PM(self, t_idx, state_t):   #GEEN COPY SLICE
        reward = 0
        if t_idx and sum(state_t) > 1:
            reward = self.penalty
        else:
            if self.terminal:
                # remove treatment cost as already included when treatment was made
                reward = self.calc_decide_PM(self.case, t_idx) - self.treatment_cost
        if not t_idx is None:
            reward += self.treatment_cost
        return reward


    def calc_outcome_PM(self, df, t_idx):
        outcome = self.calc_decide_PM(df, t_idx)
        return outcome


    def calc_decide_PM(self, df, t_idx):        # GEEN COPY SLICe
        df.reset_index(inplace=True, drop=True)
        from copy import deepcopy
        df = deepcopy(df)
        outcome = df.loc[0, "amount"] * df["value"].sum()
        if df["treatment"].sum() > 0:
            outcome += self.treatment_cost
        return outcome
