#%% IMPORTING LIBARIES
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import stats as scst
from scipy.signal import find_peaks
import warnings
import os as os
import math as m
import statistics as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import seaborn as sns
=columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
rawData = pd.read_csv('/Users/harshitakumar/Documents/Research/Dhawan Lab/Radiomics/WISDM Feature/WISDM_ar_v1.1_raw.txt',  on_bad_lines='skip', header = None, names = columns)
rawData = rawData.dropna() #dropping na values
print(rawData.shape)
rawData['z-axis'] = rawData['z-axis'].str.replace(';', '')#remmoving semi-colon
rawData['z-axis'] = rawData['z-axis'].apply(lambda x:float(x)) # transforming the z-axis to float
rawData = rawData[rawData['timestamp'] != 0] #dropping time = 0 values
rawData = rawData.sort_values(by = ['user', 'timestamp'], ignore_index=True)
print(rawData.info())
cartData = rawData[rawData.columns[3:]].to_numpy()

def rectToSph(xyz): #Converting x,y,z to r,theta,pi
    new = np.empty([len(xyz),3])
    xSqySq = xyz[:,0]**2+xyz[:,1]**2 #calculating x^2 + y^2
    new[:,0] = np.sqrt(xSqySq+xyz[:,2]**2) #calculating r 
    new[:,1] = np.arctan2(np.sqrt(xSqySq), xyz[:,2]) #calculating theta
    new[:,2] = np.arctan2(xyz[:,1], xyz[:,0]) #calculating pi
    return new
sphData = rectToSph(cartData)
cartData = np.concatenate((rawData[rawData.columns[0:3]].to_numpy(), cartData), axis = 1)
sphData = np.concatenate((rawData[rawData.columns[0:3]].to_numpy(), sphData), axis = 1)

trainData = cartData[cartData[:,0] <= 27]
testData = cartData[cartData[:,0] > 27] # test data -> Users from User ID = 28 to 36 (i.e. 9 users)
#%% LOADING AND PREPROCESSING THE DATA FILE
columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
rawData = pd.read_csv('/Users/harshitakumar/Documents/Research/Dhawan Lab/Radiomics/WISDM Feature/WISDM_ar_v1.1_raw.txt',  on_bad_lines='skip', header = None, names = columns)
rawData = rawData.dropna() #dropping na values
print(rawData.shape)
rawData['z-axis'] = rawData['z-axis'].str.replace(';', '')#remmoving semi-colon
rawData['z-axis'] = rawData['z-axis'].apply(lambda x:float(x)) # transforming the z-axis to float
rawData = rawData[rawData['timestamp'] != 0] #dropping time = 0 values
rawData = rawData.sort_values(by = ['user', 'timestamp'], ignore_index=True)
print(rawData.info())
cartData = rawData[rawData.columns[3:]].to_numpy()

def rectToSph(xyz): #Converting x,y,z to r,theta,pi
    new = np.empty([len(xyz),3])
    xSqySq = xyz[:,0]**2+xyz[:,1]**2 #calculating x^2 + y^2
    new[:,0] = np.sqrt(xSqySq+xyz[:,2]**2) #calculating r 
    new[:,1] = np.arctan2(np.sqrt(xSqySq), xyz[:,2]) #calculating theta
    new[:,2] = np.arctan2(xyz[:,1], xyz[:,0]) #calculating pi
    return new
sphData = rectToSph(cartData)
cartData = np.concatenate((rawData[rawData.columns[0:3]].to_numpy(), cartData), axis = 1)
sphData = np.concatenate((rawData[rawData.columns[0:3]].to_numpy(), sphData), axis = 1)
#normalizing spherical data
sphNormalizedData = np.concatenate((preprocessing.normalize([sphData[:,3]]), preprocessing.normalize([sphData[:,4]]), preprocessing.normalize([sphData[:,5]])))
sphNormalizedData = np.concatenate((sphData[:,0:3], np.transpose(sphNormalizedData)), axis = 1)
#%% #PLOTTING
for i in ['Walking','Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']: #cartesian plotting
  data_36 = cartData[(cartData[:,0] == 36) & (cartData[:,1] == i)][:400]
  plt.figure(figsize = (15, 6))
  plt.plot(data_36[:,2], data_36[:,3])
  plt.plot(data_36[:,2], data_36[:,4])
  plt.plot(data_36[:,2], data_36[:,5])
  plt.legend(['x-axis', 'y-axis', 'z-axis'])
  plt.ylabel(i)
  plt.xlabel('Timestamp')
  plt.title(i + ' Cartesian Coordinates', fontsize = 15)
  plt.show()

for i in ['Walking','Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']: #spherical plotting
  data_36 = sphData[(sphData[:,0] == 36) & (sphData[:,1] == i)][:400]
  plt.figure(figsize = (15, 6))
  plt.plot(data_36[:,2], data_36[:,3])
  plt.plot(data_36[:,2], data_36[:,4])
  plt.plot(data_36[:,2], data_36[:,5])
  plt.legend(['r', 'theta', 'phi'])
  plt.ylabel(i)
  plt.xlabel('Timestamp')
  plt.title(i + ' Spherical Coordinates', fontsize = 15)
  plt.show()
  
for i in ['Walking','Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']: #normalized spherical plotting
  data_36 = sphNormalizedData[(sphNormalizedData[:,0] == 36) & (sphNormalizedData[:,1] == i)][:400]
  plt.figure(figsize = (15, 6))
  plt.plot(data_36[:,2], data_36[:,3])
  plt.plot(data_36[:,2], data_36[:,4])
  plt.plot(data_36[:,2], data_36[:,5])
  plt.legend(['r', 'theta', 'phi'])
  plt.ylabel(i)
  plt.xlabel('Timestamp')
  plt.title(i + ' Normalized Spherical Coordinates', fontsize = 15)
  plt.show()

#%% SELECTING CARTESIAN VS SPHERICAL DATA
#trainData = cartData[cartData[:,0] <= 27]
#testData = cartData[cartData[:,0] > 27] # test data -> Users from User ID = 28 to 36 (i.e. 9 users)
#trainData = sphData[sphData[:,0] <= 27]
#testData = sphData[sphData[:,0] > 27] # test data -> Users from User ID = 28 to 36 (i.e. 9 users)
trainData = sphNormalizedData[sphNormalizedData[:,0] <= 27]
testData = sphNormalizedData[sphNormalizedData[:,0] > 27] # test data -> Users from User ID = 28 to 36 (i.e. 9 users)
#%% CREATING WINDOWS
x_list = []
y_list = []
z_list = []
train_labels = []

window_size = 100
step_size = 50

#creating overlaping windows of size window-size 100
for i in range(0, trainData.shape[0] - window_size, step_size):
    xs = trainData[i: i + 100,3]
    ys = trainData[i: i + 100,4]
    zs = trainData[i: i + 100,5]
    label = st.mode(trainData[i: i + 100,1])[0][0]
    x_list.append(xs.astype(float))
    y_list.append(ys.astype(float))
    z_list.append(zs.astype(float))
    train_labels.append(label)
#%% GENERATING  FEATURES
def maxMinusMin(data):
    return max(data) - min(data)

def meanAbsoluteDiff(data):
    return np.mean(np.absolute(data - np.mean(data)))

def medianAbsoluteDiff(data):
    return np.median(np.absolute(data - np.median(data)))

def negCount(data):
    return np.sum(data < 0)

def posCount(data):
    return np.sum(data > 0)

def valsAboveMean(data):
    return np.sum(data > data.mean())

def numsPeaks(data):
    return len(find_peaks(data)[0])

def energy(data): #computed by taking the mean of sum of squares of the values in a window in that particular axis
    return np.sum(data**2)/window_size

def argDiff(data): #absolute difference between index of max value in time domain and index of min value in time domain
    return abs(np.argmax(data) - np.argmin(data))

#temporarily removed kurtosis
listofFunctions = [np.mean, np.std, st.median, max, min, maxMinusMin, meanAbsoluteDiff, medianAbsoluteDiff, scst.iqr, negCount, posCount, valsAboveMean, numsPeaks, scst.skew, scst.kurtosis, energy, np.argmin, np.argmax, argDiff]
namesFunctions = ["mean", "std", "median", "maxVal", "minVal", "maxMinusMin", "meanAbsoluteDiff", "medianAbsoluteDiff", "IQR", "negativeCount", "valsAboveMean", "numsPeaks", "Skew", "Kurtosis", "Energy", "argmin", "argmax", "argDiff"]
data = [1, 2, 3, 5, 6]

trainFeatures = pd.DataFrame()
for i in range(0, len(listofFunctions)-1):
    print(namesFunctions[i])
    trainFeatures[namesFunctions[i]+'_x'] = list(map(listofFunctions[i], x_list))
    trainFeatures[namesFunctions[i]+'_y'] = list(map(listofFunctions[i], y_list))
    trainFeatures[namesFunctions[i]+'_z'] = list(map(listofFunctions[i], z_list))

#%% REPEATING FEATURES FOR FFT
#Converting signals to frequency domain
#we are going to consider only first half of the signal. This will ensure that we obtain unbiased statistical features from it.
x_list_fft = pd.Series(x_list).apply(lambda x: np.abs(np.fft.fft(x))[1:51])
y_list_fft = pd.Series(y_list).apply(lambda x: np.abs(np.fft.fft(x))[1:51])
z_list_fft = pd.Series(z_list).apply(lambda x: np.abs(np.fft.fft(x))[1:51])
#Generating features for this data
for i in range(0, len(listofFunctions)-1):
    print(namesFunctions[i])
    trainFeatures[namesFunctions[i]+'_x_fft'] = list(map(listofFunctions[i], x_list_fft))
    trainFeatures[namesFunctions[i]+'_y_fft'] = list(map(listofFunctions[i], y_list_fft))
    trainFeatures[namesFunctions[i]+'_z_fft'] = list(map(listofFunctions[i], z_list_fft))
#%% GENERATING TEST DATA FEATURES
x_list_test = []
y_list_test = []
z_list_test = []
test_labels = []
window_size = 100
step_size = 50
#creating overlaping windows of size window-size 100
for i in range(0, testData.shape[0] - window_size, step_size):
    xs = testData[i: i + 100,3]
    ys = testData[i: i + 100,4]
    zs = testData[i: i + 100,5]
    label = st.mode(testData[i: i + 100,1])[0][0]
    x_list_test.append(xs.astype(float))
    y_list_test.append(ys.astype(float))
    z_list_test.append(zs.astype(float))
    test_labels.append(label)
testFeatures = pd.DataFrame()
for i in range(0, len(listofFunctions)-1):
    print(namesFunctions[i])
    testFeatures[namesFunctions[i]+'_x'] = list(map(listofFunctions[i], x_list_test))
    testFeatures[namesFunctions[i]+'_y'] = list(map(listofFunctions[i], y_list_test))
    testFeatures[namesFunctions[i]+'_z'] = list(map(listofFunctions[i], z_list_test))
#FFT
x_list_test_fft = pd.Series(x_list_test).apply(lambda x: np.abs(np.fft.fft(x))[1:51])
y_list_test_fft = pd.Series(y_list_test).apply(lambda x: np.abs(np.fft.fft(x))[1:51])
z_list_test_fft = pd.Series(z_list_test).apply(lambda x: np.abs(np.fft.fft(x))[1:51])
for i in range(0, len(listofFunctions)-1):
    print(namesFunctions[i])
    testFeatures[namesFunctions[i]+'_x_fft'] = list(map(listofFunctions[i], x_list_test_fft))
    testFeatures[namesFunctions[i]+'_y_fft'] = list(map(listofFunctions[i], y_list_test_fft))
    testFeatures[namesFunctions[i]+'_z_fft'] = list(map(listofFunctions[i], z_list_test_fft))

#%% LOGISTIC REGRESSION MODEL
y_train = np.array(train_labels)
y_test = np.array(test_labels)
#Standardizing data
scaler = StandardScaler()
scaler.fit(trainFeatures)
train_data_lr = scaler.transform(trainFeatures)
test_data_lr = scaler.transform(testFeatures)
#Logistic regression model
lr = LogisticRegression(random_state = 21, max_iter=500)
lr.fit(train_data_lr, y_train)
y_pred = lr.predict(test_data_lr)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))
#%% CONFUSION MATRIX
labels = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]
confusionMatrix = confusion_matrix(y_test, y_pred)
sns.heatmap(confusionMatrix, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt="d", cmap = 'YlGnBu')
plt.title("Confusion matrix", fontsize = 15)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()