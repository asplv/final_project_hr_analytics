# USAGE
# Start the server:
# 	python run_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submit a a request via Python:
#	python simple_request.py

from sklearn.preprocessing import StandardScaler

import dill
import pandas as pd
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)

@app.route("/", methods=["GET"])
def general():
	return "Welcome to job seeker prediction process"

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		request_json = flask.request.get_json()
		print(model)
		preds = model.predict_proba(pd.DataFrame(request_json, index=[0]))
		preds_class = model.predict(pd.DataFrame(request_json, index=[0]))
		data["predictions"] = preds[:, 1][0]
		data["predicted_class"] = int(preds_class[0])
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return data

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	modelpath = "models/lgbm_pipeline.dill"
	load_model(modelpath)
	app.run()
