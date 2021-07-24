import os
from random import random, randint

import mlflow
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    mlflow.set_experiment(experiment_name='fgroup2')
    # Log a parameter (key-value pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())


    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")