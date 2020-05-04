import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train, x_test = x_train.reshape([-1,784]), x_test.reshape([-1,784])
x_train, x_test = x_train / 255., x_test/255.

learning_rate = 0.02
training_epochs = 50
batch_size = 256
display_step = 1
examples_to_show = 10
input_size = 784
hidden1_size = 256
hidden2_size = 128

train_data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(x_train.shape[0]).batch(batch_size)

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(hidden1_size,
                                                    activation='sigmoid',
                                                    kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                    bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0,seed=None))
        self.hidden_layer_2 = tf.keras.layers.Dense(hidden2_size,
                                                    activation='sigmoid',
                                                    kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                    bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))
        self.hidden_layer_3 = tf.keras.layers.Dense(hidden1_size,
                                                    activation='sigmoid',
                                                    kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                    bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))
        self.output_layer = tf.keras.layers.Dense(input_size,
                                                  activation='sigmoid',
                                                  kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                  bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None))

    def call(self, x):
        h1_output = self.hidden_layer_1(x)
        h2_output = self.hidden_layer_2(h1_output)
        h3_output = self.hidden_layer_3(h2_output)
        reconstructed_x = self.output_layer(h3_output)

        return reconstructed_x


@tf.function
def mse_loss(y_pred, y_true):
    return tf.reduce_mean(tf.pow(y_true - y_pred, 2))

optimizer = tf.optimizers.RMSprop(learning_rate)

@tf.function
def train_step(model, x):
    y_true = x
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = mse_loss(y_pred, y_true)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


AutoEncoder_model = AutoEncoder()

for epoch in range(training_epochs):
    for batch_x in train_data:
        _, current_loss = train_step(AutoEncoder_model, batch_x), mse_loss(AutoEncoder_model(batch_x),batch_x)
    if epoch % display_step == 0:
        print(epoch+1, current_loss)

reconstructed_result = AutoEncoder_model(x_test[:examples_to_show])

f, a = plt.subplot(2,10,figsize=(10,2))

for i in range(examples_to_show):
    a[0][i]