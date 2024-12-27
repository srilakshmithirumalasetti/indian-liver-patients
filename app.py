from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle


app = Flask(__name__)

model = joblib.load('logmodel.pkl')
onehot = joblib.load('ohe_joblib')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features =[[x for x in request.form.values()]]
	print("$"*30)
	print(int_features)
	c = ["Gender","Age","TB","DB","Alkphos","Sgpt","Sgot","TP","ALB","AG"]
	df = pd.DataFrame(int_features,columns=c)
	l = onehot.transform(df.iloc[:,:1])
	c = onehot.get_feature_names_out()
	t = pd.DataFrame(l,columns=c)
	l2 = df.iloc[:,1:]
	final =pd.concat([l2,t],axis=1)
	result = model.predict(final)
	if result == 0:
		pumba = "he/she is a liver patient"
	else:
		pumba = "he/she is not a liver patient"



	print(int_features)

	return render_template("main.html",prediction_text="The predicted Lab-Report is : {}".format(pumba))


if __name__ == "__main__":
	app.debug=True
	app.run(host='0.0.0.0', port=8000)
