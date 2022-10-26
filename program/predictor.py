# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback
import data_management

import flask
import pandas as pd

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.



# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = data_management.load_pipeline_keras() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    print(flask.request.content_type)

    # Convert from CSV to pandas
    if flask.request.content_type == "text/plain":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        data = pd.read_csv(s, header=None)
    elif "image" in flask.request.content_type:
        print("process byte string")
        print(type(flask.request.data))
        data = flask.request.data
        #save image from bytes


    else:
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )

    
    # Do the prediction
    pipeline = data_management.load_pipeline_keras()
    predictions = pipeline.predict(data)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")