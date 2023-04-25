from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    
    if request.method=='GET':
        return render_template('form.html')
    
    # when we call predict_datapoint from form then it is get method and that time it call form.html and as we press submit it
    # goes in the else section and then intializing the all variables here
    
    else :
        
        data = CustomData(
            LIMIT_BAL = float(request.form.get('LIMIT_BAL')),    # we are converting them into float because bydafault it become string because of form
            SEX = float(request.form.get('SEX')),
            EDUCATION = float(request.form.get('EDUCATION')),
            AGE = float(request.form.get('AGE')),
            avg_default = float(request.form.get('avg_bill_amt')),
            avg_bill_amt = float(request.form.get('avg_bill_amt')),
            avg_pay_amt = float(request.form.get('avg_pay_amt'))
        )
        
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)
        
        # results=str(pred)
        int(pred[0])
        
        if int(pred[0]) == 1:
            results = "He May Defaults!"
            
        else:
            results = "He May Not Defaults!"    
            
        
        
        return render_template('results.html',final_result=results)



if __name__=="__main__":
    # app.run(host='0.0.0.0',port=5001)
    app.run(host='0.0.0.0',debug=True,port=5001)
    # http://127.0.0.1:5001/
  