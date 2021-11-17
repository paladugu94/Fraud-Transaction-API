# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

# API definition
Fraud_detect_app = Flask(__name__)

@Fraud_detect_app.route('/is-fraud', methods=['POST'])
def predict():    
    if clf:
        try:
            json_ = request.json
            
            query = pd.DataFrame(json_,index=[0])
            
            query = query.drop(['nameOrig', 'nameDest'], axis = 1)
            
            if ((query.type == 'TRANSFER') | (query.type == 'CASH_OUT')).any():
                query.loc[query.type == 'TRANSFER', 'type'] = 0
                query.loc[query.type == 'CASH_OUT', 'type'] = 1
                query.type = query.type.astype(int)
                query.loc[(query.oldbalanceDest == 0) & (query.newbalanceDest == 0) & (query.amount != 0), ['oldbalanceDest', 'newbalanceDest']] = -1
                query.loc[(query.oldbalanceOrig == 0) & (query.newbalanceOrig == 0) & (query.amount != 0), ['oldbalanceOrig', 'newbalanceOrig']] = np.nan
                query['errorbalanceOrig'] = query.newbalanceOrig + query.amount - query.oldbalanceOrig
                query['errorbalanceDest'] = query.oldbalanceDest + query.amount - query.newbalanceDest
                prediction = list(clf.predict(query))
                prediction = list(bool(x) for x in prediction)
                return jsonify({'isFraud': str(prediction)})
            else:
                return jsonify({'isFraud': str('False')})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    clf = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    Fraud_detect_app.run(port=port, debug=True)