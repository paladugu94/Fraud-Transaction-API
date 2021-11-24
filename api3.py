# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import json
import os.path

# API definition
Fraud_detect_app = Flask(__name__)

@Fraud_detect_app.route('/is-fraud', methods=['POST'])
def predict():    
    if clf:
        try:
            json_ = request.json            
            
            input_ = pd.DataFrame(json_, index=[0])
            
            file_exists = os.path.isfile('file_name.csv') 
            
            if file_exists:
                temp = pd.read_csv('file_name.csv')
                old_data = temp.loc[:,temp.columns!="isFraud"]
                new_data = pd.concat([old_data,input_],axis=0)
                
                group = new_data.groupby(new_data.columns.tolist(),as_index=False).size()
                dup_check = int(pd.merge(group,input_)['size'])
                
                if dup_check>=3 :
                    result = str('[True]')
                else :                    
                    query = input_.drop(['nameOrig', 'nameDest'], axis = 1)
                    query['type'] = np.where((query.type == 'TRANSFER'),int(0),query['type'])
                    query['type'] = np.where((query.type == 'CASH_OUT'),int(1),query['type'])
                    query.loc[(query.oldbalanceDest == 0) & (query.newbalanceDest == 0) & (query.amount != 0), ['oldbalanceDest', 'newbalanceDest']] = -1
                    query.loc[(query.oldbalanceOrig == 0) & (query.newbalanceOrig == 0) & (query.amount != 0), ['oldbalanceOrig', 'newbalanceOrig']] = np.nan
                    query['errorbalanceOrig'] = query.newbalanceOrig + query.amount - query.oldbalanceOrig
                    query['errorbalanceDest'] = query.oldbalanceDest + query.amount - query.newbalanceDest
                    
                    if ((query.type.iloc[0] == int(0)) | (query.type.iloc[0] == int(1))):
                        prediction = list(clf.predict(query[0:1]))
                        prediction = list(bool(x) for x in prediction)
                        result = str(prediction)     
                    else:
                        result = str('[False]')
                        
                input_['isFraud'] = result
                
                save_csv = pd.concat([temp,input_],axis=0)
                save_csv.to_csv('file_name.csv', index=False)
                                
            else:
                query = input_.drop(['nameOrig', 'nameDest'], axis = 1)
                query['type'] = np.where((query.type == 'TRANSFER'),int(0),query['type'])
                query['type'] = np.where((query.type == 'CASH_OUT'),int(1),query['type'])
                query.loc[(query.oldbalanceDest == 0) & (query.newbalanceDest == 0) & (query.amount != 0), ['oldbalanceDest', 'newbalanceDest']] = -1
                query.loc[(query.oldbalanceOrig == 0) & (query.newbalanceOrig == 0) & (query.amount != 0), ['oldbalanceOrig', 'newbalanceOrig']] = np.nan
                query['errorbalanceOrig'] = query.newbalanceOrig + query.amount - query.oldbalanceOrig
                query['errorbalanceDest'] = query.oldbalanceDest + query.amount - query.newbalanceDest
                
                if ((query.type.iloc[0] == int(0)) | (query.type.iloc[0] == int(1))):
                    prediction = list(clf.predict(query[0:1]))
                    prediction = list(bool(x) for x in prediction)
                    result = str(prediction)     
                else:
                    result = str('[False]')
                        
                input_['isFraud'] = result

                input_.to_csv('file_name.csv', index=False) 
            
            return jsonify({'isFraud': result})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 4444# If you don't provide any port the port will be set to 12345

    clf = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    Fraud_detect_app.run(port=port, debug=True)