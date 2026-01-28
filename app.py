from plistlib import dumps

import joblib
from flask import Flask, render_template, request
from flask import Flask
from joblib.parallel import method
from sklearn.neighbors import KNeighborsClassifier

app=Flask("__name__")

X = [[20], [30], [40], [50],[55],[65],[75]]
y = ['Fail', 'Fail',  'Fail', 'Pass', 'Pass', 'Pass', 'Pass']

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y)
joblib.dump(knn,"model.pkl")
@app.route("/",methods=["GET","POST"])
def KNN():
    prediction=None
    if request.method=="POST":
        num=float(request.form.get("number"))
        prediction=knn.predict([[num]])
    return render_template('index.html',pred=prediction)

if __name__ == "__main__":
    app.run(debug=True)
