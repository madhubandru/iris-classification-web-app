import pandas as pd
from flask import Flask, jsonify, request
import pickle



#best pipeline
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('iris_data.csv')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9663043478260869
exported_pipeline = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.55, min_samples_leaf=14, min_samples_split=2, n_estimators=100)

exported_pipeline.fit(training_features, training_target)
# results = exported_pipeline.predict(testing_features)

#app
app = Flask(__name__)

#routes
@app.route('/', methods=['POST'])

def predict():
    #get data
    
    data = request.get_json(force=True)
    
    #convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    #predictions
    result = exported_pipeline.predict(data_df)
    
    #Load target encoder pickle file
    with open('target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)

    #decode the output
    result = target_encoder.inverse_transform(result)
    
    #send back to browser
    output = {'results': result[0]}
    
    #return data
    return jsonify(results=output)
    # return str(result[0])


if __name__ == "__main__":
    # app.run(debug = True)
    app.run(host ='0.0.0.0', port = 8080, debug = True)