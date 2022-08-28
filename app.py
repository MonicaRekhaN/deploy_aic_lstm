import base64
import pickle
from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import optimizers
from keras import Input, layers
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
import cv2
from base64 import b64encode
import io
from PIL import Image
from base64 import b64encode
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.layers import add
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras_preprocessing.sequence import pad_sequences
from pickle import load
import tensorflow
#from keras.preprocessing.image import load_img, img_to_array
from keras.utils.image_utils import img_to_array,load_img
#from tensorflow.keras.utils import img_to_array

def preprocess(image_path):
    # Convert all the images to size 224X224 as expected by the inception v3 model
    print(image_path)
    img = load_img(image_path, target_size=(224, 224,3))
    
    # Convert PIL image to numpy array of 3-dimensions
    x = img_to_array(img)
    #print(x.shape())
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    #x = preprocess_input(x)
    return x
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = resnet1.predict(image) # Get the encoding vector for the image
    print("testing")
    print(fea_vec.shape)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec


with open('wordtoix.pkl', "rb") as encoded_pickle:
    wordtoix = load(encoded_pickle)
with open('ixtoword.pkl', "rb") as encoded_pickle:
    ixtoword = load(encoded_pickle)



max_length = 34
vocab_size = 1652
embedding_dim = 200

#Loading LSTM mddel
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256,return_sequences =True)(se2)
decoder1 = add([fe2, se3])
decoder2 = LSTM(256)(decoder1)
decoder3 = Dense(256, activation='relu')(decoder2)
outputs = Dense(vocab_size, activation='softmax')(decoder3)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.load_weights('model.h5')
# with open('model.pkl', "rb") as encoded_pickle:
#     model = load(encoded_pickle)

# resnet = pickle.load(open('resnet_model.pkl', 'rb'))
resnet = ResNet50(include_top=True,weights='imagenet',input_shape=(224, 224,3),pooling="avg")
resnet1 = Model(resnet.input, resnet.layers[-2].output)    
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, resnet, vocab, inv_vocab
    f = request.files['file1'].read()
    buf = BytesIO(f)
    npimg = np.fromstring(f,np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)

    
    image = encode(buf).reshape((1,2048))
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return render_template('after.html', data=final,img_data=uri)

if __name__ == "__main__":
    app.run(debug=True,port=5000)
