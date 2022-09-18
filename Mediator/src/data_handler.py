import os
import numpy as np

import tensorflow as tf

class DataHandler:

    def load(self):
        root = os.path.normpath(os.getcwd() + os.sep + os.pardir)

        model_path = os.path.join(root, "Model")
        model_path = os.path.join(model_path, "model_base.h5")
        model_base = tf.keras.models.load_model(model_path)

        
        data_path = os.path.join(root, "Test")
        file_path_x =os.path.join(data_path, "x_test_transfer.npy")
        file_path_y =os.path.join(data_path, "y_test_transfer.npy")
        
        tr_x_raw = np.load(file_path_x)
        tr_y = np.load(file_path_y)

        tr_x = model_base.predict(tr_x_raw)

        return tr_x, tr_y

if __name__ == "__main__":
    handler = DataHandler()
    handler.load()