from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import umap

class Dense(tf.Module):
    def __init__(self, out_features, units=1, name=None):
        super().__init__(name=name)
        weight_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=weight_init(
                shape=(out_features, units),
                dtype="float32"),
            name='w',
            trainable=True
        )
        bias_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=bias_init(
                shape=(units,),
                dtype="float32"
            ),
            name='b',
            trainable=True
        )
    
    def __call__(self, x):
        return tf.matmul(x, self.w) + self.b    
    
class NeuralNet(Model):
    def __init__(self, X_out, optimizer, units=5, num_layers=None, learning_rate=0.9):
        super(NeuralNet, self).__init__()
        self.units = units
        self.number_of_vanilla_dense = 0
        self.dense_layers = []
        for _ in range(1, num_layers):
            self.dense_layers.append(
                Dense(X_out, units=units)
            )
            self.number_of_vanilla_dense += 1
        self.optimizer = optimizer
        self.tape = None
        self.learning_rate = learning_rate
        

    def call(self, x, epoch=False, threshold=False, train=False):
        if train:
            for index, layer in enumerate(self.dense_layers):
                x = layer(x)
                if threshold:
                    if epoch > threshold:
                        self.dense_layers[index].w = self.weight_mutation(index)
        else:
            for layer in self.dense_layers:
                x = layer(x)
        return tf.reduce_mean(x, axis=1)

    
    def step(self, x, y, epoch=False, threshold=False):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            pred = self.call(
                x, epoch=epoch,
                threshold=threshold, train=True
            )
            loss = mse(pred, y)
            self.tape = tape
        gradients = self.tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        
    def weight_mutation(self, index):
        if index == len(self.dense_layers):
            return self.dense_layers[-1].w
        else:
            # change this to a clustering algorithm so that
            # we can have an arbitrary number of dimensions.
            # k-means
            # knn
            # t-distribution mixture model (can be used for clustering)
            # and therefore dimensionality reduction.
            # u_map = umap.UMAP(
            #     n_components=self.units, n_epochs=100
            # )
            # mutated_weight = u_map.fit_transform(cat_layers)
            kmeans = KMeans(n_clusters=self.units)
            cat_layers = np.concatenate([
                self.dense_layers[index].w.numpy(),
                self.dense_layers[-1].w.numpy()
            ], axis=1)
            mutated_weight = kmeans.fit_transform(cat_layers)
            return tf.cast(mutated_weight, tf.float32)
        
def mse(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return tf.metrics.MSE(x, y)

X = np.random.random(10000).reshape(1000, 10)
y = sum(
    [X[:, i] for i in range(X.shape[1])]
)
learning_rate = 0.9
num_steps = 500
X_train, X_test, y_train, y_test = train_test_split(X, y)
optimizer = tf.optimizers.Adam(learning_rate)
nn = NeuralNet(
    X.shape[1], optimizer,
    units=10, num_layers=10, learning_rate=learning_rate
)

for step in range(num_steps):
    nn.step(
        X, y,
        epoch=step,
        threshold=10
    )
    pred = nn(X_test)
    loss_mse = mse(pred, y_test)
