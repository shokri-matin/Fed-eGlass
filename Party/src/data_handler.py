import os
import numpy as np

import tensorflow as tf

class DataHandler:

    def __init__(self, username=2,  batch_size=32, number_of_parties=2, scenario=1) -> None:
        self.batch_size = batch_size
        self.number_of_parties = number_of_parties
        self.username = username
        self.scenario = scenario
        self.tr_x, self.tr_y = self.load()

    def load(self):

        model_base = tf.keras.models.load_model("../Model/model_base.h5")

        root = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        scenario_path = os.path.join("Scenario", "Scenario {}".format(self.scenario))
        dataset_path = os.path.join(scenario_path, "Dataset")
        src_path = os.path.join(root, dataset_path)
        file_path_x =os.path.join(src_path, "tr_x_party_num_{}.npy".format(self.username-1))
        file_path_y =os.path.join(src_path, "tr_y_party_num_{}.npy".format(self.username-1))
        
        tr_x_raw = np.load(file_path_x)
        tr_y = np.load(file_path_y)

        tr_x = model_base.predict(tr_x_raw)

        return tr_x, tr_y

    def batch(self):
        index = np.random.choice(self.tr_x.shape[0], self.batch_size, replace=False)
        xtr = self.tr_x[index]
        ytr = self.tr_y[index]
        return xtr, ytr


if __name__ == "__main__":
    handler = DataHandler()
    handler.load()