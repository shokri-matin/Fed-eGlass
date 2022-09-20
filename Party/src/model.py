import tensorflow as tf

class Model(object):

    def __init__(self) -> None:

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model_head = self.create_model()

    def create_model(self):

        head_size = 8
        num_class = 2
        dropout_rate = .3

        model_head = tf.keras.models.Sequential([
                        tf.keras.layers.InputLayer(input_shape=(1024)),
                        tf.keras.layers.Dense(head_size, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation('relu'),
                        tf.keras.layers.Dropout(dropout_rate),
                        tf.keras.layers.Dense(num_class, activation=tf.nn.softmax)
                    ])
        
        model_head.compile(optimizer=self.optimizer , loss=self.loss_fn, metrics=['accuracy'])
        return model_head

    def set_weights(self, weights):
        self.model_head.set_weights(weights)

    def feedforward(self, x):
        return self.model_head(x, training=True)

    def get_gradient(self, x, y):
        with tf.GradientTape() as tape:
            ypred = self.feedforward(x)
            loss_value = self.loss_fn(y, ypred)
        grads = tape.gradient(loss_value, self.model_head.trainable_weights)
        return grads

    def loss(self, x, y):
        with tf.GradientTape() as tape:
            ypred = self.feedforward(x)
            loss_value = self.loss_fn(y, ypred)
            return loss_value.numpy()
