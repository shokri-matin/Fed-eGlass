import numpy as np
import random
import copy
import os
import tensorflow as tf


class SMCtools:
    def __init__(self, num_parties, party_id, num_participating_parties, secure_aggregation_parameter_k, scenario):
        self.num_parties = num_parties
        self.party_ID = int(party_id)
        self.num_participating_parties = num_participating_parties  # all parties in this case
        self.participating_parties = []
        self.identify_participating_parties()
        self.Secure_Aggregation_Parameter_k = secure_aggregation_parameter_k
        self.SSA_self = []
        self.SSA_others = []
        self.SSA_self_state = []  # None
        self.SSA_others_state = []  # None
        self.scenario = scenario
        self.set_seeds_from_file()


    def set_seeds_from_file(self):
        """ Reading and setting seeds for secure aggregation (SSA) from .txt file """

        # root = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        scenario_path = os.path.join("Scenario", "Scenario {}".format(self.scenario))
        seeds_folder_path = os.path.join(scenario_path, "Seeds")
        src_path = os.path.join(seeds_folder_path)  # root,
        seeds_path = os.path.join(src_path, str(self.party_ID) + '.txt')
        seeds_mat = np.loadtxt(seeds_path, dtype=str).astype(int)

        for i in range(0, self.num_parties):
            self.SSA_self.append(seeds_mat[i])
            self.SSA_self_state.append(None)
            self.SSA_others.append(seeds_mat[i + self.num_parties])
            self.SSA_others_state.append(None)

    def identify_participating_parties(self):
        """ Identify which parties will participate in this round
         input: no input
         output: IDs (indices) of participating parties """

        parties = list(range(0, self.num_parties))

        participating_parties_ids = parties[0: self.num_participating_parties]
        self.participating_parties = np.sort(participating_parties_ids)

    def exclude_my_id(self, party_ids):
        """ If my ID is among the received IDs, this function will remove it
        Input: party_IDs
        Output: party_IDs with no self.party_ID in it """

        if any(party_ids == self.party_ID):
            index = np.where(party_ids == self.party_ID)[0][0]
            party_ids = np.delete(party_ids, index)

        return party_ids

    def check_my_presence(self, party_ids):
        """ Check if my ID is among the received IDs
        Input: party_IDs
        Output: a flag showing my presence state """

        my_presence = False
        if any(party_ids == self.party_ID):
            my_presence = True

        return my_presence

    def identify_parties_self(self):
        """ Identifying the peer parties of mine for secure aggregation.
        Identifying for which parties I am among the peer parties for secure aggregation
        input: participating parties in this round
        output: IDs (indices) of Peer parties, and a Flag that shows I was included in peer parties """
        participating_parties_temp = copy.deepcopy(self.participating_parties)
        participating_parties_temp = self.exclude_my_id(participating_parties_temp)

        peer_parties = participating_parties_temp[0: self.Secure_Aggregation_Parameter_k + 1]
        peer_parties = np.array(peer_parties)
        peer_parties = np.sort(peer_parties)

        return peer_parties

    def identify_parties_others(self):
        """ Identifying the peer parties of others for secure aggregation, and
        if I need to participate in their secure aggregation by generating random masks
        input: participating parties in this round """

        peer_parties = []
        for ID in self.participating_parties:
            participating_parties_temp = copy.deepcopy(self.participating_parties)

            participating_parties_temp = participating_parties_temp[0: self.Secure_Aggregation_Parameter_k + 1]
            if self.check_my_presence(participating_parties_temp):
                peer_parties.append(ID)

        peer_parties = np.array(peer_parties)
        return peer_parties

    def generate_and_aggregate_random_masks(self, rnd_vec_shape, party_ids, mask_type):
        """ Generates random masks based on the received IDs, seeds, and states
        Input: party_IDs, mask_type= 'self' or 'others'
        Output: rnd_sum """

        rnd_sum = np.zeros(rnd_vec_shape)
        max_val = 10 ** 4  # this can be changed (by user)
        min_val = -1 * max_val

        if mask_type == 'self':
            for ID in party_ids:
                # !!! USING PARTICULAR RANDOM SEED AND STATE !!!
                np.random.seed(self.SSA_self[ID])

                if self.SSA_self_state[ID] is not None:
                    np.random.set_state(self.SSA_self_state[ID])

                # rnd_sum += np.random.rand(*rnd_vec_shape)
                rnd_sum += np.random.uniform(low=min_val, high=max_val, size=rnd_vec_shape)

                self.SSA_self_state[ID] = np.random.get_state()
                # !!! USING PARTICULAR RANDOM SEED AND STATE !!!
        elif mask_type == 'others':
            for ID in party_ids:
                # !!! USING PARTICULAR RANDOM SEED AND STATE !!!
                np.random.seed(self.SSA_others[ID])

                if self.SSA_others_state[ID] is not None:
                    np.random.set_state(self.SSA_others_state[ID])

                # rnd_sum += np.random.rand(*rnd_vec_shape)
                rnd_sum += np.random.uniform(low=min_val, high=max_val, size=rnd_vec_shape)

                self.SSA_others_state[ID] = np.random.get_state()
                # !!! USING PARTICULAR RANDOM SEED AND STATE !!!

        return rnd_sum

    def generate_mask(self, rnd_vec_shape):
        """ Generate masks based on the seeds
        Input: no input
        Output: masks to be used mask my secret values """

        # identify my peer parties for Secure Aggregation
        identified_parties_self = self.identify_parties_self()
        identified_parties_self = self.exclude_my_id(identified_parties_self)

        # calculate rnd_sum_self
        rnd_sum_self = self.generate_and_aggregate_random_masks(rnd_vec_shape=rnd_vec_shape,
                                                                party_ids=identified_parties_self, mask_type='self')

        # identify parties which I need to collaborate for Secure Aggregation
        identified_parties_others = self.identify_parties_others()
        identified_parties_others = self.exclude_my_id(identified_parties_others)

        # calculate rnd_sum_others
        rnd_sum_others = \
            self.generate_and_aggregate_random_masks(rnd_vec_shape=rnd_vec_shape,
                                                     party_ids=identified_parties_others, mask_type='others')

        return rnd_sum_self, rnd_sum_others

    def mask(self, model_parameters):
        """ Generates and adds the masks to the received values
        input: model parameters
        Output: masked model parameters """


        for i in range(len(model_parameters)):

            # model_parameters_vec = model_parameters[i]
            rnd_vec_shape = model_parameters[i].shape
            rnd_sum_self, rnd_sum_others = self.generate_mask(rnd_vec_shape)

            # masking
            model_parameters[i] += rnd_sum_others
            model_parameters[i] -= rnd_sum_self

        return model_parameters

    def update_seeds_sates(self):
        """ To update the random function's state for different seeds when I am not participating in this round
        input: no input
        output: no output """

        parties = list(range(0, self.num_parties))
        # to update random function's state for SSP_self
        _ = self.identify_parties_self()

        # to update random function's state for SSP_others
        _ = self.identify_parties_others()
