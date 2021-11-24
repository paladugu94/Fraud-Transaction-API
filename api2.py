# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import json

# API definition
Fraud_detect_app = Flask(__name__)

@Fraud_detect_app.route('/is-fraud', methods=['POST'])
def predict():    
    if clf:
        try:
            json_ = request.json            
            len_ = len(list(json_))
            
            input_ = pd.DataFrame(json_, index=np.arange(len_))
            query = input_.drop(['nameOrig', 'nameDest'], axis = 1)
            
            query['type'] = np.where((query.type == 'TRANSFER'),int(0),query['type'])
            query['type'] = np.where((query.type == 'CASH_OUT'),int(1),query['type'])
            
            query.loc[(query.oldbalanceDest == 0) & (query.newbalanceDest == 0) & (query.amount != 0), ['oldbalanceDest', 'newbalanceDest']] = -1
            query.loc[(query.oldbalanceOrig == 0) & (query.newbalanceOrig == 0) & (query.amount != 0), ['oldbalanceOrig', 'newbalanceOrig']] = np.nan
            query['errorbalanceOrig'] = query.newbalanceOrig + query.amount - query.oldbalanceOrig
            query['errorbalanceDest'] = query.oldbalanceDest + query.amount - query.newbalanceDest
            
            df = pd.DataFrame([])
            result = np.array([])
            
            for i in range(0,len_):
                
                if ((query.type.iloc[i] == int(0)) | (query.type.iloc[i] == int(1))):
                    prediction = list(clf.predict(query[i:i+1]))
                    prediction = list(bool(x) for x in prediction)
                    result = np.append(result,str(prediction))     
                else:
                    result = np.append(result,str('[False]'))
                
                results = pd.DataFrame(result, columns=['isFraud'])
                df = pd.concat([input_, results],axis=1)
                df.to_csv('file_name.csv', index=False)    
            
            result = result.tolist()
            json_format = json.dumps(result)

            return jsonify({'isFraud': json_format})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 9999# If you don't provide any port the port will be set to 12345

    clf = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    Fraud_detect_app.run(port=port, debug=True)