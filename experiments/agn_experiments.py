"""Script which collects the results of all AGN experiments
summarized in Table 3.1 of the thesis.

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

from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel

import matplotlib
import matplotlib.pyplot as plt

import numpy as np


def allocate_spark_context(num_workers, using_spark_two=False):
    """Allocates a Spark context with the specified number of
    workers (Spark executors).
    """
    sc = SparkConf()
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
    if using_spark_2:
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


def main():
    sc, reader = allocate_spark_context(num_workers=20)
    sc.close()


if __name__ == '__main__':
    main()
