from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

app = Flask(__name__)

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def cancerPrediction():
    dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv"
    df = pd.read_csv(dataset_url)

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']  
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']      

    df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

    train, test = train_test_split(df, test_size = 0.2)

    features = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']

    train_X = train[features]
    train_y=train.diagnosis

    test_X= test[features]
    test_y =test.diagnosis

    model=RandomForestClassifier(n_estimators=100, n_jobs=-1)
    model.fit(train_X,train_y)

    prediction=model.predict(test_X)
    metrics.accuracy_score(prediction,test_y)

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
    new_df = pd.DataFrame(data, columns=['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean'])
    single=model.predict(new_df)
    probability=model.predict_proba(new_df)[:,1]
    if single==1:
        o1 = "Patient is diagnosed with breast cancer."
        o2 = "confidence: {}".format(probability*100)
    else:
        o1 = "Patient is not diagnosed with breast cancer."
        o2 = "confidence: {}".format((probability)*100)


    return render_template('home.html', output1=o1, output2=o2, query1 = request.form['query1'], query2 = request.form['query2'])
                            


app.run()