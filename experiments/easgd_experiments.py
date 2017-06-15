"""Script which collects the results of all EASGD experiments
summarized in Table 3.1 of the thesis. To goal of this experiment
is to collect data on AGN for differint distributed hyperparameters.

The script will produced a pickled output file (dictionary), which later
can be used for further processing and analysis.

Author:    Joeri R. Hermans
"""


from distkeras.evaluators import *
from distkeras.predictors import *
from distkeras.trainers import *
from distkeras.transformers import *
from distkeras.utils import *

from keras.layers.convolutional import *
from keras.layers.core import *
from keras.models import *
from keras.optimizers import *

from multiprocessing import Pool

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import pickle


def allocate_spark_context(num_workers, using_spark_two=False):
    """Allocates a Spark context with the specified number of
    workers (Spark executors).
    """
    conf = SparkConf()
    conf.set("spark.app.name", "AGN Experimental Validation")
    conf.set("spark.master", "yarn-client")
    conf.set("spark.executor.cores", "1")
    conf.set("spark.executor.instances", str(num_workers))
    conf.set("spark.executor.memory", "5g")
    conf.set("spark.locality.wait", "0")
    conf.set("spark.kryoserializer.buffer.max", "2000")
    conf.set("spark.executor.heartbeatInterval", "6000s")
    conf.set("spark.network.timeout", "10000000s")
    conf.set("spark.shuffle.spill", "true")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
    # Check if we are using Apache Spark 2.
    if using_spark_two:
        from pyspark.sql import SparkSession
        sc = SparkSession.builder.config(conf=conf) \
                                 .appName(application_name) \
                                 .getOrCreate()
        reader = sc
    else:
        # Create the Spark context.
        sc = SparkContext(conf=conf)
        # Add the missing imports
        from pyspark import SQLContext
        sqlContext = SQLContext(sc)
        reader = sqlContext

    return sc, reader


def obtain_training_accuracy(history):
    """Returns the training accuracy of the model."""
    return history[len(history) - 1][1]


def obtain_validation_accuracy(model, validation):
    """Returns the validation accuracy of the model."""
    evaluator = AccuracyEvaluator(prediction_col="prediction_index", label_col="label")
    predictor = ModelPredictor(keras_model=model, features_col="features_normalized_dense")
    transformer = LabelIndexTransformer(output_dim=10)
    validation_set = validation.select("features_normalized_dense", "label")
    validation_set = predictor.predict(validation_set)
    validation_set = transformer.transform(validation_set)
    score = evaluator.evaluate(validation_set)

    return score


def construct_model():
    """Constructs the Keras model that will be used during training."""
    mlp = Sequential()
    mlp.add(Dense(1000, input_shape=(784,)))
    mlp.add(Activation('relu'))
    mlp.add(Dense(2000))
    mlp.add(Activation('relu'))
    mlp.add(Dense(1000))
    mlp.add(Activation('relu'))
    mlp.add(Dense(10))
    mlp.add(Activation('softmax'))

    return mlp


def run_experiment(t):
    """Runs the EASGD experiment with the specified number of workers, and
    communication frequency.
    """
    data = {}
    num_workers = t[0]
    communication_frequency = t[1]
    # Allocate a Spark context with the specified number of executors.
    sc, reader = allocate_spark_context(num_workers)
    # Read the training and validation set.
    training_set = reader.read.parquet("data/mnist_train.parquet").repartition(num_workers)
    training_set.persist(StorageLevel.MEMORY_AND_DISK)
    validation_set = reader.read.parquet("data/mnist_test.parquet")
    validation_set.persist(StorageLevel.MEMORY_AND_DISK)
    training_set.count()
    # Construct the Keras model.
    model = construct_model()
    # Allocate the AGN optimizer.
    optimizer = AEASGD(keras_model=model, worker_optimizer='adam', loss='categorical_crossentropy', num_workers=num_workers,
                       batch_size=128, communication_window=communication_frequency, num_epoch=40, learning_rate=0.01,
                       features_col="features_normalized_dense", label_col="label_encoded")
    # Collect the training data, and train the model.
    trained_model = optimizer.train(training_set)
    history = optimizer.get_averaged_history()
    training_accuracy = obtain_training_accuracy(history)
    validation_accuracy = obtain_validation_accuracy(trained_model, validation_set)
    training_time = optimizer.get_training_time()
    # Debug info at test start.
    print("Test: n = " + str(num_workers) + " - lambda = " + str(communication_frequency))
    # Store the metrics.
    data['history'] = history
    data['training_accuracy'] = training_accuracy
    data['validation_accuracy'] = validation_accuracy
    data['training_time'] = training_time
    # Debug info at test end.
    print("TRAINING ACCURACY: " + str(training_accuracy))
    print("VALIDATION ACCURACY: " + str(validation_accuracy))
    print("TRAINING TIME: " + str(training_time))
    # Close the Spark context.
    sc.stop()

    return data


def main():
    # Allocate a data dictionary to store experimental results.
    data = {}
    # Set the hyperparameters that need to be tested.
    workers = np.arange(10, 41, 5) # 10, 15, 20, 25, 30, 35, 40
    lambdas = np.arange(10, 41, 5) # 5. 10, 15, 20, 25, 30, ...
    for w in workers:
        data[w] = {}
        for l in lambdas:
            # Run the experiment with the specified hyperparameters.
            with Pool(1) as p:
                data[w][l] = p.map(run_experiment, [(int(w), int(l))])
    # Save the data dictionary.
    with open('aeasgd_results.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
