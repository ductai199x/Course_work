from idx2ndarray import train_images_array, train_labels_array
import time
import numpy as np
from PIL import Image

stime = time.time()
# from idx2ndarray import train_images_array,test_images_array,train_labels_array,test_labels_array

print("\nTime for loading numpy arrays from idx2ndarray :: " +
      str(time.time()-stime)+" seconds\n")

trainImgshape = train_images_array.shape
trainLabelshape = train_labels_array.shape
# testImgshape = test_images_array.shape
# testLabelshape = test_labels_array.shape

stime = time.time()
training_folderName = 'training_set_images_byte/'
trainingLabel_fileName = 'training_set_labels'
fileNameLen = len(str(trainImgshape[0]))
nIter = trainImgshape[0]

trainingLabel_file = open(trainingLabel_fileName, 'w')

for n in range(0, 100):
    filename = '0'*(fileNameLen - len(str(n)))+str(n)
    # Image.fromarray(train_images_array[n,:,:].astype(np.uint8)).save(training_folderName+filename+'.jpg')
    trainingImage_file = open(training_folderName+filename, 'wb')
    trainingImage_file.write(np.array(train_images_array[n,:,:]).tobytes())
    trainingLabel_file.write(filename + "-" + str(train_labels_array[n][0]) + "\n")
    
trainingLabel_file.close()
    
print ("Time for converting training dataset array to images & labels:: "+str(time.time()-stime)+") seconds\n")
