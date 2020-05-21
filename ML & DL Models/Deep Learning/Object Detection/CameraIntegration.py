
from keras.preprocessing.image import img_to_array,load_img
from keras.applications.vgg16 import VGG16 
from keras.models import Sequential
from keras.layers import Dense
import cv2,numpy
classes = ['bag','bottle','duster','laptop','mobile']
model_vgg = VGG16() 
model = Sequential()
for layers in model_vgg.layers[:-1]:
    model.add(layers)
model.add(Dense(5,activation = 'softmax'))
model.load_weights('C:/Users/omkar/Downloads/nb2/model.h5')
def capture():
    x = cv2.VideoCapture(0)
    ret,array = x.read()
    array = numpy.resize(array,(1,224,224,3))
    return array
def predict(array): 
    return classes[numpy.argmax(model.predict(array))]
#predictions = predict(capture())
def test():
    image = load_img('C:/Users/omkar/Downloads/nb2/test.jpg',target_size = (224,224))
    image = img_to_array(image)
    image = numpy.reshape(image,(1,224,224,3))
    return predict(image)
def continous_feed():
    x = cv2.VideoCapture(0)
    while True:
        ret,array = x.read()
        Gray = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        #cv2.imshow('array',Gray)
        array = numpy.resize(array,(1,224,224,3))
        print(predict(array))
        if cv2.waitKey(1) and 0xFF == ord('q'):
            x.release()
            break
continous_feed()