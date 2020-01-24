#Import for Flask
import csv
import os
import requests
import matlab.engine
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from pager import Pager
import time

# Imports for ML Models
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import itertools
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# ML Models

# SVM

#Declaring the lists for SVM 
training_accuracy_svm = []
test_accuracy_svm = []
arrsvm=[]
resultsvm = []

#Declaring the lists for Logistic Regression
training_accuracy_lr = []
test_accuracy_lr = []
arrlr=[]
resultlr = []


# Reading the training data
biopslides=pd.read_csv('Train.csv')
X = biopslides.drop('y', axis=1)  
y = biopslides['y']

#Splitting the data into Train-Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.60)

#Declaring the SVM Classifier
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)


#Declaring the Logistic Regression Classifier
logreg = LogisticRegression(C=100).fit(X_train, y_train)


##y_pred = svclassifier.predict(X_test)


#Flask

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
eng = matlab.engine.start_matlab()

def read_table(url):
    with open(url) as f:
        return [row for row in csv.DictReader(f.readlines())]
    
APPNAME = "Identification and Quantification of Immune Stain TILs"
STATIC_FOLDER = 'output'
TABLE_FILE = "output/displaytabledata.csv"



table = read_table(TABLE_FILE)
pager = Pager(len(table))


app = Flask(__name__, static_folder=STATIC_FOLDER)
app.config.update(
    APPNAME=APPNAME,
    )

@app.route('/')
def index():
    return render_template('upload.html')


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'processed images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
        print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        f = open("imagepath.txt", "w")
        f.write(destination)
        f.close()
        eng.Testing(nargout=0)
        time.sleep(10)

        testdata = pd.read_csv('TestImage.csv')
        for index, row in testdata.iterrows():
            arrsvm.append(row)


        for i in range(len(arrsvm)):
            resultsvm.append(svclassifier.predict([arrsvm[i]]))
            #print(svclassifier.predict([arrsvm[i]]))

        for index, row in testdata.iterrows():
            arrlr.append(row)


        for i in range(len(arrlr)):
            resultlr.append(logreg.predict([arrlr[i]]))
            #print(logreg.predict([arrlr[i]]))

        ###Confusion Matrix SVM
        ##print(confusion_matrix(y_test,y_pred))  
        ##print(classification_report(y_test,y_pred))



        

        ###Printing the accuracies for SVM
        svmtrainacc = svclassifier.score(X_train, y_train)
        svmtestacc = svclassifier.score(X_test, y_test)

        ### Print Tumor positive and negative scores using SVM
        positivesvm = resultsvm.count([0])
        negativesvm = resultsvm.count([1])

        svmresultlist = [svmtrainacc, svmtestacc, positivesvm, negativesvm]
        
        with open ('./output/displayresultsvm.csv', 'w') as writeFileDict:
            writer= csv.DictWriter(writeFileDict, fieldnames = ['Training Accuracy', 'Testing Accuracy', 'Positive Tumor Score', 'Negative Tumor Score'])
            writer.writeheader()

   
        with open('./output/displayresultsvm.csv', 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows([svmresultlist])


        ###Printing the accuracies for LR
        lrtrainacc = logreg.score(X_train, y_train)
        lrtestacc = logreg.score(X_test, y_test)
        
            

        ### Print Tumor positive and negative scores using LR
        positivelr = resultlr.count([0])
        print(positivelr)
        negativelr = resultlr.count([1])
        print(negativelr)


        lrresultlist = [lrtrainacc, lrtestacc, positivelr, negativelr]
        
        with open ('./output/displayresultlr.csv', 'w') as writeFileDict:
            writer= csv.DictWriter(writeFileDict, fieldnames = ['Training Accuracy', 'Testing Accuracy', 'Positive Tumor Score', 'Negative Tumor Score'])
            writer.writeheader()

   
        with open('./output/displayresultlr.csv', 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows([lrresultlist])
        
    return redirect('/0')


    
@app.route('/<int:ind>/')
def image_view(ind=None):
    if ind >= pager.count:
        return render_template("404.html"), 404
    else:
        pager.current = ind
        return render_template(
            'imageview.html',
            index=ind,
            pager=pager,
            data=table[ind])

@app.route('/resultsvm<int:ind>/')
def show_resultsvm(ind=None):
    ind=0
    displaysvmcsv = "output/displayresultsvm.csv"
    svmtable = read_table(displaysvmcsv)
    return render_template('displaysvmresults.html', index = ind, pager=pager, data = svmtable[ind])

@app.route('/resultlr<int:ind>/')
def show_resultlr(ind=None):
    displaylrcsv = "output/displayresultlr.csv"
    lrtable = read_table(displaylrcsv)
    return render_template('displaylrresults.html', index = ind, pager=pager, data = lrtable[ind])

@app.route('/goto', methods=['POST', 'GET'])    
def goto():
    return redirect('/' + request.form['index'])

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./output/images')
    print(image_names)
    return render_template('gallery.html', image_names=image_names)


if __name__ == '__main__':
    app.run(debug=True)
