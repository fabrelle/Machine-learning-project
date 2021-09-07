#"Building powerful image classification models using very little dat#a" from blog.keras.io.
#https://elitedatascience.com/keras-tutorial-deep-learning-in-python
# https://www.tensorflow.org/tutorials/keras/classification (code was inspired by)
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2D 
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
# numpy and graphing and random
import numpy as np
import matplotlib.pyplot as plt
import random
from random import seed
from random import random
# seed random number generator
#seed(1)
from scipy.optimize import curve_fit
############################################
############################################
def plotClassFnRandom(ClassFn, xlim, ylim): 
    p = np.random.uniform(ClassFn.pmin, ClassFn.pmax)  
    t = np.arange(xlim[0], xlim[1], 0.02)
    plt.figure(figsize=(10,10))
    plt.grid(True)
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel("{} with p = {} ".format(ClassFn.functionName[0],p), fontsize=18)
    plt.plot(t, ClassFn.f(p,t), color='green', linewidth= 8)
    plt.show()        
###################################################
def plotClassFnRandomSave_n(ClassFn, xlim, ylim, WhereToSaveDir, FileNameStr,n):
    for i in range(0,n):
        p = np.random.uniform(ClassFn.pmin, ClassFn.pmax)  
        t = np.arange(xlim[0], xlim[1], 0.02)
        plt.figure(figsize=(10,10))
        plt.grid(False)
        plt.ylim(ylim[0], ylim[1])
        plt.axis('off')
        plt.axis('equal')
       #plt.xlabel("{} with p = {} ".format(ClassFn.functionName[0],p), fontsize=18)
        plt.plot(t, ClassFn.f(p,t), color='black', linewidth= 8)
        plt.savefig(WhereToSaveDir + FileNameStr + str(i))  
        plt.show()        
###################################################
def predictionForFile(imgDir, fileName, modelName, classNamesOfFunctions, showImage = 1):
    LoadedImage = load_img(imgDir + fileName,  target_size=(sx, sy)) # color_mode="grayscale",
    if showImage == 1:
        plt.imshow(LoadedImage)
    ImageAsArray = img_to_array(LoadedImage)
    ImageToTest = np.expand_dims(ImageAsArray, axis = 0)
    predict_theArray = modelName.predict(ImageToTest)
    predict_theFunctionString = classNamesOfFunctions[np.argmax(modelName.predict(ImageToTest))].functionName[0]
    predict_theFunctionClassName = classNamesOfFunctions[np.argmax(modelName.predict(ImageToTest))]
    return predict_theArray,predict_theFunctionString,predict_theFunctionClassName
###################################################
def fitFile(imgDir, fileName, modelName, classNamesOfFunctions): #fileName, imageDir, sx,sy, model_name, ClassNames    
    pred = predictionForFile(imgDir, fileName, modelName, classNamesOfFunctions)
    LoadedImage = load_img(imgDir + fileName) # color_mode="grayscale",
    plt.imshow(LoadedImage)
    plt.title('Original Image', fontsize=32)
    LoadedImage = load_img(imgDir + fileName, color_mode="grayscale") # color_mode="grayscale",
    ImArray = img_to_array(LoadedImage) 
    numRows = ImArray.shape[0] # rows
    numCols = ImArray.shape[1] # cols
    ImArrayNorm = 1 - (ImArray/255)
    ImArrayNorm2d = np.reshape(ImArrayNorm, (numRows, numCols), order = 'C')
    IndicesWhereCurveIs = np.where(ImArrayNorm2d > .1) #.5 # row_array, col_array
    xCoords = IndicesWhereCurveIs[1]*(10/numCols)
    yCoords = ((numRows - 1) - IndicesWhereCurveIs[0])*(10/numCols)
    p_guess = (pred[2].pmin + pred[2].pmax)/2
    popt, pcov = curve_fit(pred[2].ff, xCoords, yCoords, p0=p_guess)
    fig =plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    fontsizeLegend = 20
    fontsizeAxis = 25
    fontsizeTitle = 30
    plt.plot(xCoords,yCoords, 'r.', label='input (hand drawn)')
    t = np.arange(min(xCoords), max(xCoords), 0.2)
    plt.plot(t, pred[2].f(popt, t) , 'k-',label= 'output (fitted curve)', linewidth=8)
    plt.xlabel('x', fontsize = fontsizeAxis)
    plt.ylabel('y', fontsize = fontsizeAxis)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = fontsizeLegend)
    ax.set_title(pred[2].functionName[0], fontsize = fontsizeTitle)
    ax.grid()
    plt.show()
    return popt    
###################################################
print(tf.__version__)
###################################################
# It is a good idea to tell Python where to check for files by appending system path
import sys
sys.path.append("/Users/borellefabricetenemoukam/Project 2/project JUne/ImageFiles_Curves")
###################################################
# Directories or Folders Used
imageDir = "/Users/borellefabricetenemoukam/Project 2/project JUne/ImageFiles_Curves"
TrainingDir    = imageDir + "/Training"
ValidationDir  = imageDir + "/Validation"
where2save     = imageDir + "/Temp/"
DirectoryForImagesToBePredicted = imageDir + "/ImagesToTest/"
#######################################################
n = 20  # 20000 # 20000  #100  #20000      # amount of training data to create for each function
n_test = 1 #  1000  # amount of testing data to create for each function
sx = 128   # pixels in x 
sy = 128   # pixels in y
xlim = np.array([0,10])  # xmin to xmax
ylim = np.array([0,10])  # ymin to ymax
###################################################
# Functino Class Definitions
###################################################
class myArcTan:
    def __init__(self, pmin,pmax, functionName):
        self.pmin = pmin
        self.pmax = pmax
        self.functionName = functionName # example: np.array(['Parabola (concave up)'])
    def ff(self, x, x0, y0, a, b):       # x always first, then parameters
        return y0 + a*np.arctan(b*(x-x0)) 
    def f(self, p, x):                   # f is array version of ff don't modify
        return self.ff(x,  *p.tolist())
###################################################
TheFunctionName = np.array(['arctan'])
Pmin =  np.array([3, np.pi, .5,  .75 ])
Pmax =  np.array([7, 6,       3*(2/np.pi), 5]) 
ArcTan = myArcTan(Pmin,Pmax ,TheFunctionName )
plotClassFnRandom(ArcTan, xlim, ylim)
###################################################
createData_NO_YES = 0
if createData_NO_YES == 1:
    fileNameStr = "ArcTan_6_15_2020_"     
    plotClassFnRandomSave_n(ArcTan, xlim, ylim, where2save, fileNameStr, 20)
###################################################    
# My Line 
###################################################
class myLine:
    def __init__(self, pmin,pmax, functionName):
        self.pmin = pmin
        self.pmax = pmax
        self.functionName = functionName # np.array(['Parabola (concave up)'])
    def ff(self, x, m, x0, y0):
        return y0 + m*(x-x0)
    def f(self, p, x):                   # f is array version of ff don't modify
        return self.ff(x,  *p.tolist())
###################################################
TheFunctionName = np.array(['Line (positve slope)'])
Pmin =  np.array([0.2, 0.5, 0.0])
Pmax =  np.array([2, 9.5, 2.0]) 
LinePositiveSlope = myLine(Pmin,Pmax ,TheFunctionName )
plotClassFnRandom(LinePositiveSlope, xlim, ylim)
###################################################
createData_NO_YES = 0
if createData_NO_YES == 1:
    fileNameStr = "LinePosSlope"     
    plotClassFnRandomSave_n(LinePositiveSlope, xlim, ylim, where2save, fileNameStr, 20)
###################################################
###################################################
TheFunctionName = np.array(['Line (negative slope)'])
Pmin =  np.array([-3.0, 0.5, 0.0])
Pmax =  np.array([ -0.2, 9.5, 2.0]) 
LineNegativeSlope = myLine(Pmin,Pmax ,TheFunctionName )
plotClassFnRandom(LineNegativeSlope, xlim, ylim)
###################################################
createData_NO_YES = 0
if createData_NO_YES == 1:
    fileNameStr = "LineNegSlope_6_15_2020_"     
    plotClassFnRandomSave_n(LineNegativeSlope, xlim, ylim, where2save, fileNameStr, 20)
###################################################    
# Normal Distribution
###################################################
class myNormalDist:
    def __init__(self, pmin,pmax, functionName):
        self.pmin = pmin
        self.pmax = pmax
        self.functionName = functionName # np.array(['Parabola (concave up)'])
    def ff(self, x, mu, y0, sigma, a):
        return y0 + a*(1/np.sqrt(2*np.pi*sigma**2))*np.exp(- ((x-mu)**2)/(2*sigma**2))
    def f(self, p, x):                   # f is array version of ff don't modify
        return self.ff(x,  *p.tolist())
###################################################
TheFunctionName = np.array(['normal_distribution'])
Pmin =  np.array([3, 0, .75  , 10])
Pmax =  np.array([7, 5,  2.1, 12]) 
NormalDist = myNormalDist(Pmin,Pmax ,TheFunctionName )
plotClassFnRandom(NormalDist, xlim, ylim)
###################################################
createData_NO_YES = 0
if createData_NO_YES == 1:
    fileNameStr = "NormalDist_6_15_2020_"     
    plotClassFnRandomSave_n(NormalDist, xlim, ylim, where2save, fileNameStr, 20)
###################################################
# Parablola
###################################################
class myParabola:
    def __init__(self, pmin,pmax, functionName):
        self.pmin = pmin   # np.array([0.1, 2.5, 1.0]) 
        self.pmax = pmax
        self.functionName = functionName # np.array(['Parabola (concave up)'])
    def ff(self, x, a, x0, y0):
        return y0 + a*(x-x0)**2
    def f(self, p, x):                   # f is array version of ff don't modify
        return self.ff(x,  *p.tolist())

###################################################
        #tangent my function
###################################################
class myTangent:
    def __init__(self, pmin,pmax, functionName):
        self.pmin = pmin
        self.pmax = pmax
        self.functionName = functionName # example: np.array(['Parabola (concave up)'])
    def ff(self, x, x0, y0, a, b):       # x always first, then parameters
        return y0 + a*tan(b*(x-x0)) 
    def f(self, p, x):                   # f is array version of ff don't modify
        return self.ff(x,  *p.tolist())
###################################################
TheFunctionName = np.array(['Tan'])
Pmin =  np.array([3, np.pi, .5,  .75 ])
Pmax =  np.array([7, 6,     3*(2/np.pi), 5]) 
tan = myTangent(Pmin,Pmax ,TheFunctionName )
plotClassFnRandom(tan, xlim, ylim)
###################################################
createData_NO_YES = 0
if createData_NO_YES == 1:
    fileNameStr = "Tangent_7_3_2020"     
    plotClassFnRandomSave_n(Tan, xlim, ylim, where2save, fileNameStr, 20)
# END END END END END END END END END END END END
###################################################
# MUST BE in Order of corresponding folders (alphabetical)
ClassNamesOfFunctions = np.array([ArcTan,
                                  LineNegativeSlope,
                                  LinePositiveSlope,
                                  NormalDist,
                                  ParabolaConcaveUp,
                                  ParabolaConcaveDown,
                                  Tangent] )
###################################################
numClasses = ClassNamesOfFunctions.shape[0]
###################################################
# The Model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(sx, sx,3)))
model.add(Conv2D(64, (3, 3),activation='relu', input_shape=(sx, sx,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(numClasses, activation='softmax'))
###################################################
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']) 
###################################################
print(model.summary())
###################################################
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.002,
        zoom_range=0.2,
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip=False)  #True default
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        TrainingDir,   #'data/train',
        target_size= (sx,sy),   #  (150, 150),
        batch_size=32,
        class_mode='categorical')  #binary
validation_generator = test_datagen.flow_from_directory(
        ValidationDir,  #  'data/validation',
        target_size= (sx,sy), #(150, 150),
        batch_size=32, #32
        class_mode='categorical') #binary
###################################################
# steps_per_epoch * epochs should equal how many images we have about
model.fit_generator(
        train_generator,
        steps_per_epoch= 5,   #50 worked good #2000
        epochs= 3,  # 31 #50,  # 3 worked good
        validation_data=validation_generator,
        validation_steps=5)
###################################################
# image files to test NN on
# put these in DirectoryForImagesToBePredicted
fileName19 = "Normal6_14_2020_new1.png"
fileName20 = "ArcTan6_14_2020_new1.png" # no good
fileName21 = "ParabUp6_14_2020_new1.png"
fileName22 = "LineNegSlope6_14_2020_new1.png"
fileName23 = "LinePosSlope6_14_2020_new1.png"  # no good
fileName24 = "ArcTan6_14_2020_new2.png"
fileName25 = "Normal6_14_2020_new2.png"
fileName26 = "Normal6_15_2020_newLightBlue1.png"
fileName27 = "ParabUp6_15_2020_newGreen1.png"
fileName28 = "ParabDown6_16_2020_new1.png"
fileName29 = "ParabUP6_16_2020_new1.png"

fileName30 = "NormalDistPicture copy.png"

########################################################
#  MAIN PREDICTION FUNCTION 
fitFile(DirectoryForImagesToBePredicted, fileName30, model, ClassNamesOfFunctions)
#model.save_weights('first_try.h5')
# Misc
#   Utility Function -- No fitting, just prediction and image show 
run_predictionForFile = 0
if run_predictionForFile == 1:
    predictionForFile(DirectoryForImagesToBePredicted, fileName30, model, ClassNamesOfFunctions, showImage = 1)
########################################################
