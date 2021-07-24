import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    mlflow.set_experiment(experiment_name='TestExperiment')
    X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 1, 0])
    lr = LogisticRegression()
    lr.fit(X, y)
    score = lr.score(X, y)
    print("Score: %s" % score)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(lr, "model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)


    # # Log an artifact (output file)
    # if not os.path.exists("outputs"):
    #     os.makedirs("outputs")
    # with open("outputs/test.txt", "w") as f:
    #     f.write("hello world!")
    # log_artifacts("outputs")