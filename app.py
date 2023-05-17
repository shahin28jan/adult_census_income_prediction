from flask import Flask,request,render_template,jsonify
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            age = float(request.form.get('age')),
            workclass = request.form.get('workclass'),
            education = request.form.get('education'),
            marital_status = request.form.get('marital_status'),
            occupation = request.form.get('occupation'),
            relationship = request.form.get('relationship'),
            race = request.form.get('race'),
            gender = request.form.get('gender'),
            capital_gain = float(request.form.get('capital_gain')),
            capital_loss = float(request.form.get('capital_loss')),
            hours_per_week = float(request.form.get('hours_per_week')),
            native_country = request.form.get('native_country')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result=pred
        if result == 0:
            return render_template("result.html",final_result = "your Income is Less than equal to 50K")#: {}".format(result))
        elif result == 1:
            return render_template("result.html",final_result = "your Income is more than equal to 50K")#: {}".format(result))







if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
