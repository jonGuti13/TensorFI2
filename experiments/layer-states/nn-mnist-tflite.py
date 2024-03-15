import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np

import math, sys, time, random

from src import tensorfi2 as tfi

from tensorflow.lite.tools.flatbuffer_utils import read_model_with_mutable_tensors

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

'''
try:
        check = sys.argv[1]
        assert check == "train" or "testy"
except:
        print("Provide either the 'train' or 'test' argument to run.")
        sys.exit()

if(check == "train"):
        # Save the untrained weights for future training with modified dataset
        model.save_weights('h5/nn-untrained.h5')

        model.fit(train_images, train_labels, epochs=5,
                validation_data=(test_images, test_labels))

        model.save_weights('h5/nn-trained.h5')

else:
        test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
        print("Accuracy before faults:", test_acc)

        model.load_weights('h5/nn-trained.h5')

        tfi.inject(model=model, confFile="confFiles/sample.yaml")

        test_loss, test_acc = model.evaluate(test_images,  test_labels)
        print("Accuracy after faults:", test_acc)
'''

#When opening TensorFlow Lite models this way to be able to modify the parameters and inject faults, it is important to be careful choosing the
#layer index where the fault is going to be injected. This is because this format changes the layer indexes to strange values that I have not been
#able to fully understand yet.
#
#For example, if I open /experiments/layer-states/tflite/nn-trained-int8.tflite in Netron, I see that the 'location' of the first FullyConnected layer (10x18)
#is 2 and the location of the second FullyConnected layer (128x784) is 3. However, if I open the model as done in src/tensorfi2.py, the size of layer 3 is 1280 (10x18)
#and the size of layer 4 is 100352 (128x784), so it is as if they were just upside down and 1 has been added for every layer value.
#
#There are no more layers with trainable variables so the random value for layer input should range between 3 and 4 in this case (taking into account what it has
#just been explained).

model_path = sys.argv[1]
conf = sys.argv[2]
filePath = sys.argv[3]
filePath = os.path.join(filePath, "res.csv")

f = open(filePath, "w")
numFaults = int(sys.argv[4])
numInjections = int(sys.argv[5])
offset = 10
num = test_images.shape[0]

totsdc = 0.0

ind = []
init = random.sample(range(num), numInjections+offset)
model = read_model_with_mutable_tensors(model_path)
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for i in init:
    # Check if the input type is quantized, then rescale input data to uint8
    if input_details[0]['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details[0]["quantization"]
        test_image = (test_images[i:i+1] / input_scale + input_zero_point).astype(input_details[0]["dtype"])

    interpreter.set_tensor(input_details[0]["index"], test_image)
    interpreter.invoke()
    segs = interpreter.get_tensor(output_details[0]["index"])
    preds = np.argmax(segs)

    if(test_labels[i:i+1] == preds):
        ind.append(i)

ind = ind[:numInjections]

start = time.time()
for i in range(numFaults):
    model = read_model_with_mutable_tensors(model_path)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tfi.inject(model=model, interpreter=interpreter, confFile=conf)

    sdc = 0.
    for i in ind:
        if input_details[0]['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details[0]["quantization"]
            test_image = (test_images[i:i+1] / input_scale + input_zero_point).astype(input_details[0]["dtype"])

        interpreter.set_tensor(input_details[0]["index"], test_image)
        interpreter.invoke()
        segs = interpreter.get_tensor(output_details[0]["index"])
        preds = np.argmax(segs)

        if(preds != test_labels[i:i+1]):
            sdc = sdc + 1.
    f.write(str(sdc/numInjections))
    f.write("\n")
    totsdc = totsdc + sdc
f.write("\n")
f.write(str(totsdc/(numFaults*numInjections)))
f.write("\n")
f.write("Time for %d injections: %f seconds" % (numFaults*numInjections, time.time() - start))
f.close()