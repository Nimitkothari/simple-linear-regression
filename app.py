#!flask/bin/python
# Web server
from flask import Flask,jsonify,request,Response
# Get request parameters
from flask import request
# This is needed for logistic regression
from sklearn import linear_model
# Save and load models to/from disk
import os
import json
import pickle

port = int(os.getenv("PORT", 3000))
# API server
app = Flask(__name__)
# Output is the probability that the given
# input (ex. email) belongs to a certain
# class (ex. spam or not)
logReg = linear_model.LogisticRegression()

# Samples (your features, they should be normalized
# and standardized). Normalization scales the values
# into a range of [0,1]. Standardization scales data
# to have a mean of 0 and standard deviation of 1
# Note that we are using fake data here just to
# demonstrate the concept
X = [[1.0, 1.0, 2.1], [2.0, 2.2, 3.3], [0.3, 0.1, 0.3] ,[0.2, 0.1, 0.3],[0.6, 0.3, 0.2],[0.4, 0.1, 0.5],[0.5, 0.2, 0.1],[0.1, 0.2, 0.3],[0.1, 0.1, 0.3]]

# Labeled data (Spam or not)
Y = [1, 0, 1, 0, 1, 1, 0, 0, 0]

# Build the model
logReg.fit(X, Y)
# Save it to disk
pickle.dump(logReg, open('logReg.pkl', 'wb'))
# Define end point
@app.route('/predict', methods=['POST'])
def get_prediction():
    req_body = request.get_json(force=True)
    # We are using 3 features. For example:
    # subject line, word frequency, etc
    param1 = req_body['p1']
    param2 = req_body['p2']
    param3 = req_body['p3']
    # Load model from disk
    logReg = pickle.load(open('logReg.pkl', 'rb'))

    # Predict
    pred = logReg.predict([[param1, param2, param3]])

    if (pred[0] == 0):
        result = "spam"
    else :
        result = "valid"

    msg = {
            "message": "Email is %s" % (result)
        }
    resp = Response(response=json.dumps(msg),
                        status=200,\
                        mimetype="application/json")
    return resp

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)

        # Main app
