# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 23:54:00 2022
@author: mayan
"""

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
from static.Uploads.Federated import *
import zipfile

IMG_FOLDER = os.path.join('static', 'Uploads')
#IMG_FOLDER = str(os.getcwd())

app = Flask(__name__)
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = IMG_FOLDER



"""
def images():
    
    cv_img = []
    # get the path/directory
    folder_dir = "Static/Uploads"
    for images in os.listdir(folder_dir):
        file_path = os.path.abspath(os.path.join(folder_dir, images))
        if images[-3:] == "png": 
            #name, ext = os.path.splitext(images)
            #im = Image.open(file_path)
            #imgplot = plt.imshow(im)
            #print(imgplot)
            f=file_path
        
   
    return f
"""        
    



@app.route("/") 
def main():
	return render_template("index.html")


@app.route("/", methods = ['POST'])
def getvalue():
    
    ROC=""
    CM=""
    act=""
    con=""
    ma=""
    
    dataset = request.files['dataset'] 
    print(type(dataset)) 
    dataset_path = "static/" + dataset.filename
    dataset.save(dataset_path)
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
         zip_ref.extractall("static/Uploads")
         zip_ref.extractall("")
    #dataset_path = "static/" + dataset.filename	
		#dataset.save(dataset_path)
    
    
    title = request.form.get("Project title", False)
    fed_learning = request.form.get("fed_learning", False)
    client = request.form.get("client", False)
    SVM = request.form.get("SVM", False)
    NB = request.form.get("NB", False)
    DT = request.form.get("DT", False)
    RF = request.form.get("RF", False)
    KNN = request.form.get("KNN", False)
    LR = request.form.get("LR", False)
    CNN = request.form.get("CNN", False)
    CNN_layers = request.form.get("CNN_layers", 2)
    Aug = request.form.get("Aug", False)
    size_Aug = request.form.get("size_Aug", False)
    Accuracy = request.form.get("Accuracy", False)
    precision = request.form.get("precision", False)
    Recall = request.form.get("Recall", False)
    auc = request.form.get("auc", False)
    confusion = request.form.get("confusion", False)
    tpr = request.form.get("tpr", False)
    dcr = request.form.get("dcr", False)
    
    
    name, ext = os.path.splitext(dataset.filename)
    
    
    #print(str(name))
    '''
    print(client)
    print("client", type(client))
    print(CNN_layers)
    print("CNN_layerst", type(CNN_layers))
    client=int(client)
    CNN_layers=int(CNN_layers)
    print(client)
    print("client", type(client))
    print(CNN_layers)
    print("CNN_layerst", type(CNN_layers))
    client=int(client)
    print(222)
    print(type(222))
    
    '''
    cli=int(client)
    
    
    mod=""
    if(SVM != False):
        mod=SVM
    elif(NB != False):
        mod=NB
    elif(DT != False):
        mod=DT
    elif(NB != False):
        mod=NB
    elif(RF != False):
        mod=RF
    elif(KNN != False):
        mod=KNN
    elif(LR != False):
        mod=LR
    elif(CNN != False):
        mod=CNN
    print(mod)
    print(app.config['UPLOAD_FOLDER'])
    if( CNN_layers != "0"):
        lay=int(CNN_layers)
        CNN_fun(str(name), lay,cli ,mod)
    else:
        CNN_fun(str(name), 3, cli, mod)
    
    #CNN_fun(str(name), CNN_layers, int(client) ,str(mod))
    #CNN_fun(str(name), 2,int(client), "")
    
    if(mod == "CNN"):
        
        ROC = os.path.join(app.config['UPLOAD_FOLDER'], name+'_1_ROC.png')
    else:
        ROC = os.path.join(app.config['UPLOAD_FOLDER'], name+'_1_ROC_final.png')
    
 
    CM = os.path.join(app.config['UPLOAD_FOLDER'], name+'_1_CM.png')
    #
    #ROC = os.path.join(app.config['UPLOAD_FOLDER'], name+'_ROC_final.png')
    
    act = os.path.join(app.config['UPLOAD_FOLDER'], name+'_activation.png')
    
    con = os.path.join(app.config['UPLOAD_FOLDER'], name+'_conv2d.png')
    
    ma = os.path.join(app.config['UPLOAD_FOLDER'], name+'_max_pooling2d.png')
    
    
  
    return render_template("pass.html", title=title, fed_learning=fed_learning, client=client, SVM=SVM, NB=NB, DT=DT, RF=RF, KNN=KNN, LR=LR, CNN=CNN, CNN_layers=CNN_layers, Aug=Aug, size_Aug=size_Aug, Accuracy=Accuracy, precision=precision, Recall=Recall, auc=auc, confusion=confusion, tpr=tpr ,dcr=dcr, CM = CM, ROC=ROC, act= act, con=con, ma=ma  )

# prevent cached responses
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ =='__main__':
	#app.debug = True
	#app.run(host='0.0.0.0' , port=8080)
   
    app.run(debug=False)