from flask import Flask, jsonify, request, make_response
import pandas as pd
import numpy as np
import sklearn
import pickle
import json

app= Flask(__name__)

@app.route("/", methods=["GET"])
def hello():
    return jsonify("Hello from Bean Forecast App!")


@app.route("/predictions" , methods=['GET'])
def predictions():

    data = request.get_json()
    df=pd.DataFrame(data['data'])
    # define explanatory vars
    cols=['sepal_length','sepal_width','petal_length','petal_width']

    data_all_x_cols = cols
    try:
        # preprocess the data for ML
        new_data = df.dropna() # deletes Na and NaN
        preprocessed_df = new_data
    except:
        return jsonify("Error occured while preprocessing your data for our model!")
    # filename = model_to_fit
    saved_model = os.path.join(folder, str("final_model.sav")) 
    loaded_model = pickle.load(open(saved_model, 'rb'))
    try:
        predictions= loaded_model.predict(preprocessed_df[data_all_x_cols])
    except:
        return jsonify("Error occured while processing your data into our model!")
    print("done")
    response={'data':[]}
    response['data']=list(predictions)
    return make_response(jsonify(response),200)

if __name__=='__main__':
    app.run(debug=True)