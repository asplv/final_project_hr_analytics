import pandas as pd
import numpy as np
from datetime import datetime
import urllib.request
import json
from sklearn.preprocessing import StandardScaler

# Job seeker status prediction
# https: // www.kaggle.com / arashnic / hr - analytics - job - change - of - data - scientists)
# addressing the server and making prediction 

def get_user_input():
    action = input('Press q to quit or r for new prediction:')
    while True:
        if action in ['r', 'q']:
            return action
        action = input('Unknown command. Press q to quit or r for new prediction:')



def df_jsonify(x_test):
    """converting df to json
    """
    results = []

    for i in range(x_test.shape[0]):
        obj = {}
        for col in x_test.columns:
            value = x_test[col].iloc[i]
            if value:
                if isinstance(value, np.generic):
                    value = value.item()
                    obj[col] = int(value)
                obj[col] = value
            else:
                obj[col] = None

        results.append(obj)
    return results



def get_prediction(x_test, obj_index, predict_proba=False):
    """
    addressing the server and making prediction of object class:

    x_test - dataframe
    obj_index - object index
    predict_proba - predict proba instead of class

    """
    x_test_json = df_jsonify(x_test)

    body = x_test_json[obj_index]

    myurl = "http://127.0.0.1:5000//predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')  # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    response = urllib.request.urlopen(req, jsondataasbytes)

    if predict_proba:
        return json.loads(response.read())['predictions'], body

    return json.loads(response.read())['predicted_class'], body



stop = False
while not stop:
# uploading file
    print('Place your  with shape (n, 13) to input folder with name "test.csv".')

    try:
        x_test = pd.read_csv("input/test.csv")
    except:
        print('Upload error')
        exit(1)

    if x_test.shape[0] > 1:
        print(f'{datetime.now()} Input is successfully uploaded.')
    else:
        print('Dataframe is empty.')


    n = int(input(f'Insert number of objects you would like to get predistions for: '))
    result = []  
    for i in list(range(n)):
        preds = {}
        pred, obj = get_prediction(x_test, i)
        for key in obj.keys():
            preds[key] = obj[key]

        preds['target'] = pred
        result.append(preds)

    test_preds = pd.DataFrame(result)
    test_preds.to_csv(f"output/test_predictions_{datetime.now()}.csv", index=False)
    print(f'{datetime.now()} Prediction for {n} rows is ready and saved in output.')
    if get_user_input() == 'q':
        print('Bye')
        stop = True
