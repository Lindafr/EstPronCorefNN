#IMPORT
import csv
import pandas as pd
import keras 
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Lambda
from keras.layers import ELU, GaussianNoise, GaussianDropout, AlphaDropout
from keras import initializers
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
import statistics, math
from keras.utils.vis_utils import plot_model
import matplotlib 
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from random import randint
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from keras import regularizers

#GET DATA
def _getCategoricalFeatures ():
    """The categorical features are stored in an external file"""
    categoricalFeaturesFile="allCategoricalFeatures.txt"

    categoricalFeatures=[]
    fi = open(categoricalFeaturesFile, "r")
    for line in fi :
        line = line.rstrip()
        categoricalFeatures.append(line)

    return categoricalFeatures


def _getXy (myFile) :
    """Get X and y for all classification algorithms"""

    df = pd.read_csv(myFile)
    features = df.columns

  #of course the last feature is the category
    X = df[features[:-1]]
    categoricalFeatures=_getCategoricalFeatures()
    ct = ColumnTransformer(
      [('one_hot_encoder', OneHotEncoder(categories='auto'), categoricalFeatures)],
      remainder='passthrough'
    )
    X_ohe = ct.fit_transform(X)

    y = np.array(df['category'])
    return X_ohe,y

X_data, y_data = _getXy("ccorpusSciKitLearn.txt")

X_datapos = []
y_datapos = []
X_dataneg = []
y_dataneg = []
for i, y in enumerate(y_data):
    if y == 1 and i + 32487 < len(y_data):
        X_datapos += [X_data[i]]
        y_datapos += [y]
    elif i + 32487 < len(y_data):
        X_dataneg += [X_data[i]]
        y_dataneg += [y]


X_val = X_data[-32487:] #test set
y_val = y_data[-32487:] #test set


X_val_ = []
for x in X_val:
    X_val_ += [x.toarray()[0]]
X_val = X_val_
    

#TRAINING'S METADATA

REPEATS = range(15) #how many time a k-fold is run. How many averages will I see
TEST_NAME = 'NN1_epcoh20' #name of this test, used for saving things.
KFOLD = 7
EPOCHS = 20 #epochs of each model.
BATCH_SIZE = 256 #batch_size of each trained model

#TRAIN THE MODEL

def do_model(train_i,test_i):
    '''train a model, evaluate it, return model, history and results of evaluations'''
    input_dim = 494 #num of features
    elu_alpha = 1.0
    seq = Sequential()

    seq.add(Dense(494, input_shape=(input_dim,), kernel_regularizer=regularizers.l2(0.01)))
    seq.add(Dropout(0.2))
    seq.add(BatchNormalization())
    seq.add(ELU(alpha=elu_alpha))

    seq.add(Dense(490))
    seq.add(Dropout(0.1))
    seq.add(BatchNormalization())
    seq.add(ELU(alpha=elu_alpha))

    seq.add(Dense(200))
    seq.add(Dropout(0.2))
    seq.add(BatchNormalization())
    seq.add(ELU(alpha=elu_alpha))

    seq.add(Dense(25))
    seq.add(Dropout(0.1))
    seq.add(ELU(alpha=elu_alpha))
    
    seq.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    seq.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']) 
    history = seq.fit(np.array(X_new)[train_i], np.array(y_new)[train_i], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, validation_data=(np.array(X_new)[test_i], np.array(y_new)[test_i]))
    
    #get accuracy and loss on validation data. From history get the loss and acc on test and train data.
    evaluation = seq.evaluate(np.array(X_val), np.array(y_val), verbose=0)
    
    #From classification report get data.
    y_pred = [np.round(i) for i in seq.predict(np.array(X_val))]  
    report = classification_report(y_val, y_pred)
    report_dict = {'0': [], '1': [], 'macro': [], 'weighted': []}
    for line in report.split('\n'):
        items = line.split()
        if len(items) > 0 and items[0] in report_dict.keys():
            for i in items[1:]:
                if i != 'avg':
                    report_dict[items[0]] += [float(i)]
					
    #Get TP etc for average confusion matrix. 
    cm = confusion_matrix(y_true=y_val, y_pred=y_pred, labels=[1, 0])
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]
    #Get MCC
    mcc = matthews_corrcoef(y_val, y_pred)
    return evaluation[1], evaluation[0], TP, FN, FP, TN, history, seq, report_dict,mcc
 

#Average value of several k-foldings
f_ev_acc = list()
f_ev_loss = list()
f_f11s = list()
f_f10s = list()
f_TPs = list()
f_FNs = list()
f_FPs = list()
f_TNs = list()
f_mccs = list()
f_train_acc = []
f_test_acc = []
f_train_loss = []
f_test_loss = []
f_report_dict = {'0': [], '1': [], 'macro': [], 'weighted': []}

for i in REPEATS:
    print('Starting REPEAT no.'+ str(i))
	
    #Average values of one k-folding iterations
    ev_acc = list()
    ev_loss = list()
    TPs = list()
    FNs = list()
    FPs = list()
    TNs = list()
    mccs = list()
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    report_dicts = []
    
    #Start k-folding. Each time shuffle differently
    seed = randint(1,20000)
    kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=seed)
    
    #each repeat the new kfolding test is done with different negs (can repeat themselves)
    X_new = random.sample(X_dataneg, 51010)+X_datapos+X_datapos+X_datapos+X_datapos+X_datapos+X_datapos+X_datapos+X_datapos+X_datapos+X_datapos #no need to shuffle, kfold does it.
    y_new = random.sample(y_dataneg, 51010)+y_datapos+y_datapos+y_datapos+y_datapos+y_datapos+y_datapos+y_datapos+y_datapos+y_datapos+y_datapos

    X_new_ = []
    for x in X_new:
        X_new_ += list(x.toarray())
    X_new = X_new_


    for train, test in kfold.split(X_new, y_new):
        acc, loss, TP, FN, FP, TN, history, seq, report_dict, mcc = do_model(train, test)
        ev_acc.append(acc)
        ev_loss.append(loss)
        TPs.append(TP)
        FNs.append(FN)
        FPs.append(FP)
        TNs.append(TN)
        mccs.append(mcc)
        train_acc += [history.history['accuracy']]
        test_acc += [history.history['val_accuracy']]
        train_loss += [history.history['loss']]
        test_loss += [history.history['val_loss']]
        report_dicts += [report_dict]
    f_ev_acc.append(np.mean(ev_acc))
    f_ev_loss.append(np.mean(ev_loss))
    f_TPs.append(np.mean(TPs))
    f_FNs.append(np.mean(FNs))
    f_FPs.append(np.mean(FPs))
    f_TNs.append(np.mean(TNs))
    f_mccs.append(np.mean(mccs))
    f_train_acc += [ [np.mean(x) for x in zip(*train_acc)] ]
    f_test_acc += [ [np.mean(x) for x in zip(*test_acc)] ]
    f_train_loss += [ [np.mean(x) for x in zip(*train_loss)] ]
    f_test_loss += [ [np.mean(x) for x in zip(*test_loss)] ]
    for key in f_report_dict.keys():
        average = []
        for dic in report_dicts:
            average += [dic[key]]
        average_key = [np.mean(x) for x in zip(*average)]
        f_report_dict[key] += [average_key]
        
		
#CLASSIFICATION AVERAGE REPORT WITH 95% CONFIDENCE INTERVAL. Create file TEST_NAME+'_results.txt'
def do_statistics(scores):
    '''I have averages of several k-foldings. Find mean of this and also standard error. 
    Formula for that I took here: https://machinelearningmastery.com/evaluate-skill-deep-learning-models/
    (chapter: "Repeat Evaluation Experiments")'''
    
    mean_skill = float(np.mean(scores))
    standard_deviation = statistics.stdev(scores)
    standard_error = standard_deviation / math.sqrt(len(scores))
    interval = standard_error * 1.96
    lower_interval = mean_skill - interval
    upper_interval = mean_skill + interval
    return mean_skill, standard_error, lower_interval, upper_interval

def typeIandIIerror(f_TPs, f_FNs,f_FPs,f_TNs):
    cm = np.array([[np.mean(f_TPs), np.mean(f_FNs)], [np.mean(f_FPs), np.mean(f_TNs)]])
    typeI = cm[1][0] / (cm[1][0]+cm[0][0])
    typeII = cm[0][1] / (cm[0][1] + cm[1][1])
    return cm, typeI, typeII

#AVERAGE FROM THIS: from sklearn.metrics import classification_report
f = open(TEST_NAME+'_results.txt', 'w')
report_1 = [do_statistics([ x[0] for x in f_report_dict['1']]), do_statistics([ x[1] for x in f_report_dict['1']]), do_statistics([ x[2] for x in f_report_dict['1']]),do_statistics([ x[3] for x in f_report_dict['1']])]
report_0 = [do_statistics([ x[0] for x in f_report_dict['0']]), do_statistics([ x[1] for x in f_report_dict['0']]), do_statistics([ x[2] for x in f_report_dict['0']]),do_statistics([ x[3] for x in f_report_dict['0']])]
f.write('     \tprecision\trecall\t\tf1-score\tsupport'+'\n')
f.write('    0\t'+str(round(report_0[0][0],3))+'\t\t'+str(round(report_0[1][0],3))+'\t\t'+str(round(report_0[2][0],3))+'\t\t'+str(round(report_0[3][0],3))+'\n')
f.write('     \t'+str(round(report_0[0][2],3))+'-'+str(round(report_0[0][3],3))+'\t'+str(round(report_0[1][2],3))+'-'+str(round(report_0[1][3],3))+'\t'+str(round(report_0[2][2],3))+'-'+str(round(report_0[2][3],3))+'\t'+str(round(report_0[3][2],3))+'-'+str(round(report_0[3][3],3))+'\n')
f.write('   1\t'+str(round(report_1[0][0],3))+'\t\t'+str(round(report_1[1][0],3))+'\t\t'+str(round(report_1[2][0],3))+'\t\t'+str(round(report_1[3][0],3))+'\n')
f.write('     \t'+str(round(report_1[0][2],3))+'-'+str(round(report_1[0][3],3))+'\t\t'+str(round(report_1[1][2],3))+'-'+str(round(report_1[1][3],3))+'\t'+str(round(report_1[2][2],3))+'-'+str(round(report_1[2][3],3))+'\t'+str(round(report_1[3][2],3))+'-'+str(round(report_1[3][3],3))+'\n')
f.write('\n')
report_1 = [do_statistics([ x[0] for x in f_report_dict['macro']]), do_statistics([ x[1] for x in f_report_dict['macro']]), do_statistics([ x[2] for x in f_report_dict['macro']]),do_statistics([ x[3] for x in f_report_dict['macro']])]
report_w = [do_statistics([ x[0] for x in f_report_dict['weighted']]), do_statistics([ x[1] for x in f_report_dict['weighted']]), do_statistics([ x[2] for x in f_report_dict['weighted']]),do_statistics([ x[3] for x in f_report_dict['weighted']])]

f.write('macro\t'+str(round(report_1[0][0],3))+'\t\t'+str(round(report_1[1][0],3))+'\t\t'+str(round(report_1[2][0],3))+'\t\t'+str(round(report_1[3][0],3))+'\n')
f.write('     \t'+str(round(report_1[0][2],3))+'-'+str(round(report_1[0][3],3))+'\t'+str(round(report_1[1][2],3))+'-'+str(round(report_1[1][3],3))+'\t'+str(round(report_1[2][2],3))+'-'+str(round(report_1[2][3],3))+'\t'+str(round(report_1[3][2],3))+'-'+str(round(report_1[3][3],3))+'\n')
f.write('weighted\t'+str(round(report_w[0][0],3))+'\t\t'+str(round(report_w[1][0],3))+'\t\t'+str(round(report_w[2][0],3))+'\t\t'+str(round(report_w[3][0],3))+'\n')
f.write('     \t'+str(round(report_w[0][2],3))+'-'+str(round(report_w[0][3],3))+'\t'+str(round(report_w[1][2],3))+'-'+str(round(report_w[1][3],3))+'\t'+str(round(report_w[2][2],3))+'-'+str(round(report_w[2][3],3))+'\t'+str(round(report_w[3][2],3))+'-'+str(round(report_w[3][3],3))+'\n')
f.write('\n')

#STATISTICS OF THE RESULTS' ACCURACY, LOSS, AND F1-s, Type 1 and Type 2 errors + confusion_matrix:
s_acc = do_statistics(f_ev_acc)
s_loss = do_statistics(f_ev_loss)
s_mcc = do_statistics(f_mccs)
f.write('Average accuracy on val data: ' + str(round(s_acc[0],3))+' (95% interval being: '+str(round(s_acc[2],3))+'-'+str(round(s_acc[3],3))+')'+'\n')
f.write('Average loss on val data: ' + str(round(s_loss[0],3))+' (95% interval being: '+str(round(s_loss[2],3))+'-'+str(round(s_loss[3],3))+')'+'\n')
f.write('Average MCC value on val data: ' + str(round(s_mcc[0],3))+' (95% interval being: '+str(round(s_mcc[2],3))+'-'+str(round(s_mcc[3],3))+')')
f.write('\n')
cm, typeI, typeII = typeIandIIerror(f_TPs, f_FNs,f_FPs,f_TNs)
f.write('Type I error is: ' + str(round(typeI, 2))+'\n')
f.write('TypeII error is: '+ str(round(typeII, 2))+'\n')
f.write('Confusion matrix is:\n')
f.write(str(round(cm[0][0], 2)) + ' '+ str(round(cm[0][1], 2)) + '\n')
f.write(str(round(cm[1][0], 2)) + ' '+ str(round(cm[1][1], 2)) + 'n')
f.close()


#SEE HOW TRAINNG WENT. THE VALUES ARE AVERAGE OF ALL TRAININGS.
sns.set(font_scale=1.0)

# Plot training & validation average accuracy values
plt.plot([np.mean(x) for x in zip(*f_train_acc)])
plt.plot([np.mean(x) for x in zip(*f_test_acc)])
plt.ylabel('Oigsus')
plt.xlabel('Epohh')
plt.legend(['Treeningandmestik', 'Valideerimisandmestik'], loc='upper left')
plt.savefig(TEST_NAME+'_acc_val.pdf')
plt.close() 
plt.clf()

# Plot training & validation average loss values
plt.plot([np.mean(x) for x in zip(*f_train_loss)])
plt.plot([np.mean(x) for x in zip(*f_test_loss)])
plt.ylabel('Kahju')
plt.xlabel('Epohh')
plt.legend(['Treeningandmestik', 'Valideerimisandmestik'], loc='upper left')
plt.savefig(TEST_NAME+'_loss_val.pdf')
plt.close() 
plt.clf()



#DRAW AVERAGE CONFUSION MATRIX

def plotConfusionMatrix (f_TPs, f_FNs,f_FPs,f_TNs, fName) :    

    """
    Plot the normalized average confusion matrix
    """

    cm = np.array([[np.mean(f_TPs), np.mean(f_FNs)], [np.mean(f_FPs), np.mean(f_TNs)]])
    cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]

    plt.subplots(figsize=(10, 10))

    target_names=["Viitesuhtes","Viitesuhteta"]
    sns.set(font_scale=1.8)
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Tegelik', fontsize=20)
    plt.xlabel('Ennustatud', fontsize=20)
    plt.savefig(fName)


plotConfusionMatrix(f_TPs, f_FNs,f_FPs,f_TNs, TEST_NAME+'_confused')

#MODEL SUMMARY IN A SEPERATE PICTURE

plot_model(seq, to_file=TEST_NAME+'.png', show_shapes=True, show_layer_names=True)
