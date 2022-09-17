import tensorflow as tf

class Model:

    def __init__(self) -> None:

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model_base,  self.model_head = self.create_model()

    def create_model(self):

        head_size = 16
        num_class = 2
        dropout_rate = .3

        # model_base = tf.keras.models.Sequential([
        #                 tf.keras.layers.Flatten(input_shape=(2048, 1)),
        #                 tf.keras.layers.Conv1D(512, kernel_size= 2, strides=2),
        #                 tf.keras.layers.BatchNormalization(),
        #                 tf.keras.layers.Activation('relu'),
        #                 tf.keras.layers.Dropout(dropout_rate),
        #                 tf.keras.layers.MaxPooling1D(pool_size=2),
        #                 tf.keras.layers.Conv1D(128, kernel_size= 2, strides=2),
        #                 tf.keras.layers.BatchNormalization(),
        #                 tf.keras.layers.Activation('relu'),
        #                 tf.keras.layers.Dropout(dropout_rate), 
        #                 tf.keras.layers.Conv1D(64, kernel_size= 2, strides=2),
        #                 tf.keras.layers.BatchNormalization(),
        #                 tf.keras.layers.Activation('relu'),
        #                 tf.keras.layers.Dropout(dropout_rate), 
        #                 tf.keras.layers.Flatten()                      
        #             ])
        
        model_base = tf.keras.models.load_model("../Model/model_base.h5")

        model_head = tf.keras.models.Sequential([
                        tf.keras.layers.InputLayer(input_shape=[8192]),
                        tf.keras.layers.Dense(head_size, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation('relu'),
                        tf.keras.layers.Dropout(dropout_rate),
                        tf.keras.layers.Dense(num_class, activation=tf.nn.softmax)
                    ])
        
        model_head.compile(optimizer=self.optimizer , loss=self.loss_fn, metrics=['accuracy'])

        return model_base, model_head

    def get_weights(self):
        weights = self.model_head.get_weights()
        return weights

    def set_weights(self, weights):
        self.model_head.set_weights(weights)

    def backpropagate(self, grads):
        pass

    def feedforward(self, x):
        x_i = self.model_base(x, training=True)
        return self.model_head(x_i, training=True)

    def evaluate(self, x, y):
        x_i = self.model_base(x, training=True)
        test_loss, test_acc = self.model_head.evaluate(x_i,  y, verbose=2)
        return test_loss, test_acc