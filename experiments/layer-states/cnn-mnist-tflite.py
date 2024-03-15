import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import numpy as np
import time, sys, math, random

from tensorflow.lite.tools.flatbuffer_utils import read_model_with_mutable_tensors

from src import tensorfi2 as tfi

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
'''
# Change to True if you want to train from scratch
train = False

if(train):
	# Save the untrained weights for future training with modified dataset
	model.save_weights('h5/cnnm-untrained.h5')

	model.fit(train_images, train_labels, batch_size=100, epochs=10,
		validation_data=(test_images, test_labels))

	model.save_weights('h5/cnnm-trained.h5')

else:
	model.load_weights('h5/cnnm-trained.h5')

	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print("Accuracy before faults:", test_acc)

	tfi.inject(model=model, confFile="confFiles/sample.yaml")

	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
	print("Accuracy after faults:", test_acc)
'''

#When applying full-integer quantization in TensorFlow Lite, bias are quantized to 32 bit integers, but as a consequence of how the model is opened, each bias is stored
#as 4 consecutive 8-bit elements in the whole bias 1D array, so the value of the first bias is calculated as follows: b[3] * 2^24 + b[2] * 2^16 + b[1] * 2^8 + b[0].

#When opening TensorFlow Lite models this way to be able to modify the parameters and inject faults, it is important to be careful choosing the
#layer index where the fault is going to be injected. This is because this format changes the layer indexes to strange values that I have not been
#able to fully understand yet.

#For example, if I open /experiments/layer-states/tflite/cnn-trained-int8.tflite in Netron, I see that the 'location' of the first Conv2D layer (32x5x5x1) is 1
#and I guess that the location of the bias (32) of the first Conv2D layer is 2 because I see that the location of the second Conv2D layer (64x5x5x32) is 3.
#However, the location of the MaxPool2D is 4, so where is the bias (64) of the second Conv2D layer? I think that it is a mess.

#If I open the model as done in src/tensorfi2.py the sizes are:
#      40 for layer 3 --> The 10 32-bit bias of FullyConnected layer represented as 40 8-bit integers.
#   10240 for layer 4 --> The 10x1024 8-bit weights of FullyConnected layer.
#     256 for layer 5 --> The 64 32-bit bias of the second Conv2D layer represented as 256 8-bit integers.
#   51200 for layer 6 --> The 64x5x5x32 8-bit weights of the second Conv2D layer.
#     128 for layer 7 --> The 32 32-bit bias of the first Conv2D layer represented as 128 8-bit integers.
#     800 for layer 8 --> The 32x5x5x1 8-bit weights of the first Conv2D layer.

#There are no more layers with trainable variables so the random value for layer index should be [4,6,8] for INT8 weights and [3,5,7] for INT32 weights

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