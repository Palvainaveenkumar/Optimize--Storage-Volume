
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
import random
from SketchDataPlacement import CountMinSketch
import sys

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from SOA import SOA
from SwarmPackagePy import testFunctions as tf
from genetic_selection import GeneticSelectionCV

main = tkinter.Tk()
main.title("Optimize the Storage Volume using Data Mining Techniques")
main.geometry("1300x1200")

global filename
global text
global SOA_accuracy
global hashtable_storage,sketch_base_storage
global dataset
existing_hashtabe = dict()
sketch = CountMinSketch(8, 4)

def upload(): 
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    dataset = pd.read_csv('Dataset/data.csv')
    dataset = dataset.values
    dataset = dataset[:,0]
    text.insert(END,str(dataset))

def runSketchBasePlacement():
    text.delete('1.0', END)
    global hashtable_storage,sketch_base_storage
    for i in range(0,10):
        stream = dataset[random.randint(0, random.randint(0, len(dataset) - 1))]
        sketch.add(stream)
        if stream in existing_hashtabe.keys():
            count = existing_hashtabe.get(stream) + 1
            existing_hashtabe[stream] = count
        else:
            existing_hashtabe[stream] = 1
    print(existing_hashtabe)
    hashtable_storage = sys.getsizeof(existing_hashtabe)
    sketch_base_storage = sys.getsizeof(sketch)
    print(str(hashtable_storage)+" "+str(sketch_base_storage))
    for movie_name, hashtable_count in existing_hashtabe.items():
        sketch_count = sketch.query(movie_name)
        text.insert(END,movie_name+" Hashtable watch count = "+str(hashtable_count)+" Sketch Based count = "+str(sketch_count)+"\n")
    text.insert(END,"\n\nSine Optimization Storage Cost : "+str(hashtable_storage)+"\n")
    text.insert(END,"\n\nSketch Based Data Placement Storage Cost : "+str(sketch_base_storage))    
    
    
def sketchBaseGraph():
    height = [hashtable_storage,sketch_base_storage]
    bars = ('Sine Optimization Storage Size','Sketch Based Data Placement')
    f, ax = plt.subplots(figsize=(5,5))
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    ax.legend(fontsize = 12)
    plt.show()

def difference(datasets, intervals=1):
    difference = list()
    for i in range(intervals, len(datasets)):
        values = datasets[i] - datasets[i - intervals]
        difference.append(values)
    return pd.Series(difference)

def convertDataToTimeseries(dataset, lagvalue=1):
    dframe = pd.DataFrame(dataset)
    cols = [dframe.shift(i) for i in range(1, lagvalue+1)]
    cols.append(dframe)
    dframe = pd.concat(cols, axis=1)
    dframe.fillna(0, inplace=True)
    return dframe


def scaleDataset(trainX, testX):
    scalerValue = MinMaxScaler(feature_range=(-1, 1))
    scalerValue = scalerValue.fit(trainX)
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1])
    trainX = scalerValue.transform(trainX)
    testX = testX.reshape(testX.shape[0], testX.shape[1])
    testX = scalerValue.transform(testX)
    return scalerValue, trainX, testX

def forecastRNN(model, batchSize, testX):
    testX = testX.reshape(1, 1, len(testX))
    forecast = model.predict(testX, batch_size=batchSize)
    return forecast[0,0]
    
def inverseDifference(history_data, yhat_data, intervals=1):
    return yhat_data + history_data[-intervals]

def inverseScale(scalerValue, Xdata, Xvalue):
    newRow = [x for x in Xdata] + [Xvalue]
    array = np.array(newRow)
    array = array.reshape(1, len(array))
    inverse = scalerValue.inverse_transform(array)
    return inverse[0, -1]

def uploadTemporal():
    global filename
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def runTemporal():
    text.delete('1.0', END)
    input_classify = 30
    dataset = pd.read_csv(filename, header=0, parse_dates=[0], index_col=0, squeeze=True)
    rawValues = dataset.values
    dataset = dataset.values
    print(dataset)
    dataset = difference(dataset, 1)
    print("diff "+str(dataset))
    dataset = convertDataToTimeseries(dataset, 1)
    dataset = dataset.values
    print(dataset)
    trainX, testX = dataset[0:-input_classify], dataset[-input_classify:]
    print(trainX.shape)
    print(testX.shape)
    scaler_value, trainX, testX = scaleDataset(trainX, testX)
    text.delete('1.0', END)
    text.insert(END,"Total dataset size             : "+str(dataset.shape[0])+"\n");
    text.insert(END,"Dataset size used for training : "+str(trainX.shape[0])+"\n");
    text.insert(END,"Dataset size used for testing  : "+str(testX.shape[0])+"\n");

    trainXX, trainY = trainX[:, 0:-1], trainX[:, -1]
    trainXX = trainXX.reshape(trainXX.shape[0], 1, trainXX.shape[1])
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, trainXX.shape[1], trainXX.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())	
    for i in range(50):
        model.fit(trainXX, trainY, epochs=1, batch_size=1, verbose=2, shuffle=False)
        model.reset_states()
        print(i)
    trainReshaped = trainX[:, 0].reshape(len(trainX), 1, 1)
    model.predict(trainReshaped, batch_size=1)
    prediction_list = list()
    for i in range(len(testX)):
        X, y = testX[i, 0:-1], testX[i, -1]
        yhat = forecastRNN(model, 1, X)
        yhat = inverseScale(scaler_value, X, yhat)
        yhat = inverseDifference(rawValues, yhat, len(testX)+1-i)
        prediction_list.append(yhat)
        expected = rawValues[len(trainX) + i + 1]
        text.insert(END,'Temporal Period = '+str(i+1)+" Classification Production = "+str(yhat)+" Expected Production = "+str(expected)+"\n")
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Temporal Period')
    plt.ylabel('Classification/Expected Production')
    plt.plot(rawValues[-input_classify:], 'ro-', color = 'red')
    plt.plot(prediction_list, 'ro-', color = 'green')
    plt.legend(['Expected Production', 'Classification Production'], loc='upper left')
    plt.title('Expected Vs Classification Graph')
    plt.show()    
        
    
def uploadLargeDataset():
    global filename
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def runSOAClassification():
    global SOA_accuracy
    text.delete('1.0', END)
    dataset = pd.read_csv(filename)
    dataset = dataset.sample(frac=1)#randomize the whole dataset
    X = dataset.drop(["Time","Class"],axis=1)
    Y = pd.DataFrame(dataset[["Class"]])
    XX = X.values
    print(XX.shape)
    Y = Y.values
    alh = SOA(XX, tf.easom_function, -10, 10, 2, 20)
    data = alh.get_agents()
    X = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            X.append(data[i][j][0:XX.shape[1]])
            
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.90)
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    text.insert(END,'Dataset contains total records : '+str(len(X))+"\n")
    text.insert(END,"Application using 80% dataset records to train Classification : "+str(len(X_train))+"\n")
    text.insert(END,"Application using 20% dataset records to test Classification  : "+str(len(X_test))+"\n\n")

    rfc = RandomForestClassifier(n_estimators=5, random_state=0)
    rfc.fit(X_train, y_train)
    predict = rfc.predict(X_test)
    classification_acc = accuracy_score(y_test,predict)*100
    classification_error = 100 - classification_acc

    text.insert(END,'Classification Accuracy on Large Dataset : '+str(classification_acc)+"\n")
    text.insert(END,'Classification Error on Large Dataset : '+str(classification_error)+"\n")
    SOA_accuracy = classification_acc

    height = [classification_acc,classification_error]
    bars = ('Classification Accuracy','Classification Error')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def runmultiObjective():
    global SOA_accuracy
    dataset = pd.read_csv(filename)
    dataset = dataset.sample(frac=1)#randomize the whole dataset
    X = dataset.drop(["Time","Class"],axis=1)
    Y = pd.DataFrame(dataset[["Class"]])
    X = X.values
    Y = Y.values
    rfc = RandomForestClassifier(n_estimators=5, random_state=0)
    multi_objective = GeneticSelectionCV(rfc,
                                  cv=2,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=10,
                                  n_population=3,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=3,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=3,
                                  caching=True,
                                  n_jobs=-1)
    multi_objective = multi_objective.fit(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X,Y,train_size=0.90)
    predict = multi_objective.predict(X_test)
    multi_objective_acc = accuracy_score(y_test,predict)*100
    text.insert(END,'SOA Classification Accuracy on Large Dataset : '+str(SOA_accuracy)+"\n")
    text.insert(END,'Multiobjective Classification Accuracy on Large Dataset : '+str(multi_objective_acc)+"\n")
    height = [SOA_accuracy,multi_objective_acc]
    bars = ('SOA Classification Accuracy','Multiobjective Classification Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()
    
def showGUI():
    global text
    font = ('times', 15, 'bold')
    title = Label(main, text='Optimize the Storage Volume using Data Mining Techniques')
    title.config(bg='greenyellow', fg='dodger blue')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)

    font1 = ('times', 12, 'bold')
    text=Text(main,height=17,width=150)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=30,y=120)
    text.config(font=font1)


    font1 = ('times', 12, 'bold')
    uploadButton = Button(main, text="Upload Storage Data", command=upload)
    uploadButton.place(x=50,y=500)
    uploadButton.config(font=font1)  

    sbdpButton = Button(main, text="Hashtable Sine Optimization & Sketch Based Storage", command=runSketchBasePlacement)
    sbdpButton.place(x=260,y=500)
    sbdpButton.config(font=font1) 

    sbdpgraphButton = Button(main, text="Sine Optimization & Sketch Based Storage Comparison Graph", command=sketchBaseGraph)
    sbdpgraphButton.place(x=710,y=500)
    sbdpgraphButton.config(font=font1)

    temporaluploadButton = Button(main, text="Upload Temporal Pattern Dataset (Temporal Dataset)", command=uploadTemporal)
    temporaluploadButton.place(x=50,y=550)
    temporaluploadButton.config(font=font1)

    temporalclassificationButton = Button(main, text="Run Temporal Pattern Classification using Data Placement", command=runTemporal)
    temporalclassificationButton.place(x=440,y=550)
    temporalclassificationButton.config(font=font1)

    uploadlargeButton = Button(main, text="Upload Large Classification Dataset (Credit Card)", command=uploadLargeDataset)
    uploadlargeButton.place(x=50,y=600)
    uploadlargeButton.config(font=font1)

    soaButton = Button(main, text="Run Classification with SOA Algorithm", command=runSOAClassification)
    soaButton.place(x=440,y=600)
    soaButton.config(font=font1)

    moButton = Button(main, text="Multiobjective & SOA Prediction Accuracy on Large Dataset", command=runmultiObjective)
    moButton.place(x=50,y=650)
    moButton.config(font=font1)

    main.config(bg='LightSkyBlue')
    main.mainloop()

if __name__ == "__main__":
    showGUI()
