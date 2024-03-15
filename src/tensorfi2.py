#!/usr/bin/python

import os, logging

import tensorflow as tf
from struct import pack, unpack

import numpy as np
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K
import random, math
from src import config

def bitflip(f, pos):

	""" Single bit-flip in 32 bit floats """

	f_ = pack('f', f)
	b = list(unpack('BBBB', f_))
	[q, r] = divmod(pos, 8)
	b[q] ^= 1 << r
	f_ = pack('BBBB', *b)
	f = unpack('f', f_)
	return f[0]

def bitflip_int8(value, pos):
    """ Single bit-flip in 8-bit signed integers, counting from right to left """

    # Ensure that 'value' is within the valid range for signed 8-bit integers
    if (value > 127) | (value < -128):
        raise("El valor no está en int8")

    # Convert the signed integer to its binary representation
    binary_representation = bin(value & 0xFF)[2:].zfill(8)

    # Flip the specified bit in the binary representation
    flipped_binary = list(binary_representation)
    flipped_binary[7 - pos] = '1' if binary_representation[7 - pos] == '0' else '0'

    # Convert the modified binary representation back to an integer
    modified_value = int(''.join(flipped_binary), 2)

    return modified_value

class inject():
	def __init__(
		self, model, confFile, interpreter=None, log_level="ERROR", **kwargs
		):

		# Logging setup
		logging.basicConfig()
		logging.getLogger().setLevel(log_level)
		logging.debug("Logging level set to {0}".format(log_level))

		# Retrieve config params
		fiConf = config.config(confFile)
		self.Model = model # No more passing or using a session variable in TF v2

		# Call the corresponding FI function
		fiFunc = getattr(self, fiConf["Target"])
		fiFunc(model, fiConf, interpreter=interpreter, **kwargs)

	def layer_states(self, model, fiConf, interpreter=None, **kwargs):

		""" FI in layer states """

		if(fiConf["Mode"] == "single"):

			""" Single layer fault injection mode """

			logging.info("Starting fault injection in a random layer")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]
			fiFormat = fiConf["Format"]

			# Choose a random layer for injection
			# If random layer is chosen
			if(fiConf["Layer"] == "N"):
				if (fiFormat != "fp32"):
					layernum = random.randint(firstIndex, lastIndex)
				else:
					layernum = random.randint(0, len(model.trainable_variables) - 1)

			# If layer position is specified for flip
			else:
				layernum = int(fiConf["Layer"])

			# Get layer states info
			if (fiFormat == "fp32"):
				v = model.trainable_variables[layernum]
				train_variables_numpy = v.numpy()
			elif (fiFormat == "int8"):
				v = model.buffers[layernum].data.reshape(interpreter.get_tensor(layernum-1).shape)
				train_variables_numpy = v
			elif (fiFormat == "int32"):
				v = model.buffers[layernum].data
				train_variables_numpy = v

			if(fiFault == "zeros"):
				fiSz = (fiSz * num) / 100
				fiSz = math.floor(fiSz)

			# If the layer is a 1D array (bias of convolution layer or gamma or beta parameters of batch normalization layers, for example).
			if len(v.shape) == 1:
				if fiFormat != "int32":
					ind0 = random.sample(range(v.shape[0]), fiSz)
				else:
					ind0 = random.sample(range(int(v.shape[0]/4)), fiSz)

			# If the layer is a 2D array (dense layer, for example).
			elif len(v.shape) == 2:
				ind0 = random.sample(range(v.shape[0]), fiSz)
				ind1 = random.sample(range(v.shape[1]), fiSz)

			# If the layer is a 4D array (kernel of convolution layer, for example)
			else:
				# Choose the indices for FI
				ind0 = random.sample(range(v.shape[0]), fiSz)
				ind1 = random.sample(range(v.shape[1]), fiSz)
				ind2 = random.sample(range(v.shape[2]), fiSz)
				ind3 = random.sample(range(v.shape[3]), fiSz)

			# Inject the specified fault into the randomly chosen values
			if(fiFault == "zeros"):
				for i in range(len(ind0)):
					if len(v.shape) == 1:
						train_variables_numpy[ind0[i]] = 0
					elif len(v.shape) == 2:
						train_variables_numpy[ind0[i], ind1[i]] = 0
					else:
						train_variables_numpy[ind0[i], ind1[i], ind2[i], ind3[i]] = 0
			elif(fiFault == "random"):
				for i in range(len(ind0)):
					if (fiFormat == "fp32"):
						randomNumber = np.random.random()
					elif fiFormat == "int32":
						randomNumber = random.randint(0, 4294967295)
					elif fiFormat == "int8":
						randomNumber = random.randint(0, 255)

					if len(v.shape) == 1:
						train_variables_numpy[ind0[i]] = randomNumber
					elif len(v.shape) == 2:
						train_variables_numpy[ind0[i], ind1[i]] = randomNumber
					else:
						train_variables_numpy[ind0[i], ind1[i], ind2[i], ind3[i]] = randomNumber
			elif(fiFault == "bitflips"):
				for i in range(len(ind0)):
					# If random bit chosen to be flipped
					if(fiConf["Bit"] == "N"):
						if (fiFormat == "fp32" or fiFormat == "int32"):
							pos = random.randint(0, 31)
						elif fiFormat == "int8":
							pos = random.randint(0, 7)
						else:
							raise("Formato mal especificado")

					# If bit position specified for flip
					else:
						pos = int(fiConf["Bit"])

					if len(v.shape) == 1:
						if fiFormat == "int32":
							#Each 32-bit bias is represented via 4 consecutive 8-bit integers in the whole bias array,
       						#so to get the index of the 8-bit element that must be changed, the index of the bias and
             				#by specifying bit position that is going to be changed must be known in advance.

							if (pos >= 0) & (pos < 8):
								ind0[i] = 0 + 4 * ind0[i]
							elif (pos >= 8) & (pos < 16):
								ind0[i] = 1 + 4 * ind0[i]
							elif (pos >= 16) & (pos < 24):
								ind0[i] = 2 + 4 * ind0[i]
							elif (pos >= 24) & (pos < 32):
								ind0[i] = 3 + 4 * ind0[i]

						train_parameter = train_variables_numpy[ind0[i]]
					elif len(v.shape) == 2:
						train_parameter = train_variables_numpy[ind0[i], ind1[i]]
					else:
						# Choose the indices for FI
						train_parameter = train_variables_numpy[ind0[i], ind1[i], ind2[i], ind3[i]]

					if fiFormat == "fp32":
						train_parameter_fi = bitflip(train_parameter, pos)
					elif fiFormat == "int32":
						train_parameter_fi = bitflip_int8(train_parameter.view(dtype=np.int8), pos % 8)
					elif fiFormat == "int8":
						train_parameter_fi = bitflip_int8(train_parameter.view(dtype=np.int8), pos)
					else:
						raise("Formato mal especificado")

					if len(v.shape) == 1:
						train_variables_numpy[ind0[i]] = train_parameter_fi
					elif len(v.shape) == 2:
						train_variables_numpy[ind0[i], ind1[i]] = train_parameter_fi
					else:
						train_variables_numpy[ind0[i], ind1[i], ind2[i], ind3[i]] = train_parameter_fi

			if (fiFormat == "fp32"):
				v.assign(train_variables_numpy)
			else:
				None

			logging.info("Completed injections... exiting")

		elif(fiConf["Mode"] == "multiple"):

			""" Multiple layer fault injection mode """

			logging.info("Starting fault injection in all layers")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			# Loop through each available layer in the model
			for n in range(len(model.trainable_variables) - 1):

				# Get layer states info
				v = model.trainable_variables[n]
				num = v.shape.num_elements()

				if(fiFault == "zeros"):
					fiSz = (fiSz * num) / 100
					fiSz = math.floor(fiSz)

				# Choose the indices for FI
				ind = random.sample(range(num), fiSz)

				# Unstack elements into a single dimension
				elem_shape = v.shape
				v_ = tf.identity(v)
				v_ = tf.keras.backend.flatten(v_)
				v_ = tf.unstack(v_)

				# Inject the specified fault into the randomly chosen values
				if(fiFault == "zeros"):
					for item in ind:
						v_[item] = 0.
				elif(fiFault == "random"):
					for item in ind:
						v_[item] = np.random.random()
				elif(fiFault == "bitflips"):
					for item in ind:
						val = v_[item]

						# If random bit chosen to be flipped
						if(fiConf["Bit"] == "N"):
							pos = random.randint(0, 31)

						# If bit position specified for flip
						else:
							pos = int(fiConf["Bit"])
						val_ = bitflip(val, pos)
						v_[item] = val_

				# Reshape into original dimensions and store the faulty tensor
				v_ = tf.stack(v_)
				v_ = tf.reshape(v_, elem_shape)
				v.assign(v_)

			logging.info("Completed injections... exiting")


	def layer_outputs(self, model, fiConf, **kwargs):

		""" FI in layer computations/outputs """

		if(fiConf["Mode"] == "single"):

			""" Single layer fault injection mode """

			logging.info("Starting fault injection in a random layer")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			# Get the input for which dynamic injection is to be done
			x_test = kwargs["x_test"]

			# Choose a random layer for injection
			randnum = random.randint(0, len(model.layers) - 2)

			fiLayer = model.layers[randnum]

			# Get the outputs of the chosen layer
			get_output = K.function([model.layers[0].input], [fiLayer.output])
			fiLayerOutputs = get_output([x_test])

			# Unstack elements into a single dimension
			elem_shape = fiLayerOutputs[0].shape
			fiLayerOutputs[0] = fiLayerOutputs[0].flatten()
			num = fiLayerOutputs[0].shape[0]

			if(fiFault == "zeros"):
				fiSz = (fiSz * num) / 100
				fiSz = math.floor(fiSz)

			# Choose the indices for FI
			ind = random.sample(range(num), fiSz)

			# Inject the specified fault into the randomly chosen values
			if(fiFault == "zeros"):
				for item in ind:
					fiLayerOutputs[0][item] = 0.
			elif(fiFault == "random"):
				for item in ind:
					fiLayerOutputs[0][item] = np.random.random()
			elif(fiFault == "bitflips"):
				for item in ind:
					val = fiLayerOutputs[0][item]
					if(fiConf["Bit"] == "N"):
						pos = random.randint(0, 31)
					else:
						pos = int(fiConf["Bit"])
					val_ = bitflip(val, pos)
					fiLayerOutputs[0][item] = val_

			# Reshape into original dimensions and get the final prediction
			fiLayerOutputs[0] = fiLayerOutputs[0].reshape(elem_shape)
			get_pred = K.function([model.layers[randnum + 1].input], [model.layers[-1].output])
			pred = get_pred([fiLayerOutputs])

			# Uncomment below line and comment next two lines for ImageNet models
			# return pred
			labels = np.argmax(pred, axis=-1)
			return labels[0]

			logging.info("Completed injections... exiting")

		elif(fiConf["Mode"] == "multiple"):

			""" Multiple layer fault injection mode """

			logging.info("Starting fault injection in all layers")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			# Get the input for which dynamic injection is to be done
			x_test = kwargs["x_test"]

			# Get the outputs of the first layer
			get_output_0 = K.function([model.layers[0].input], [model.layers[1].output])
			fiLayerOutputs = get_output_0([x_test])

			# Loop through each available layer in the model
			for n in range(1, len(model.layers) - 2):

				# Unstack elements into a single dimension
				elem_shape = fiLayerOutputs[0].shape
				fiLayerOutputs[0] = fiLayerOutputs[0].flatten()
				num = fiLayerOutputs[0].shape[0]
				if(fiFault == "zeros"):
					fiSz = (fiSz * num) / 100
					fiSz = math.floor(fiSz)

				# Choose the indices for FI
				ind = random.sample(range(num), fiSz)

				# Inject the specified fault into the randomly chosen values
				if(fiFault == "zeros"):
					for item in ind:
						fiLayerOutputs[0][item] = 0.
				elif(fiFault == "random"):
					for item in ind:
						fiLayerOutputs[0][item] = np.random.random()
				elif(fiFault == "bitflips"):
					for item in ind:
						val = fiLayerOutputs[0][item]
						if(fiConf["Bit"] == "N"):
							pos = random.randint(0, 31)
						else:
							pos = int(fiConf["Bit"])
						val_ = bitflip(val, pos)
						fiLayerOutputs[0][item] = val_

				# Reshape into original dimensions
				fiLayerOutputs[0] = fiLayerOutputs[0].reshape(elem_shape)

				"""
				Check if last but one layer reached;
				if not, replace fiLayerOutputs with the next prediction to continue
				"""
				if(n != (len(model.layers) - 3)):
					get_output = K.function([model.layers[n+1].input], [model.layers[n+2].output])
					fiLayerOutputs = get_output([fiLayerOutputs])

				# Get final prediction
				get_pred = K.function([model.layers[len(model.layers)-1].input], [model.layers[-1].output])
				pred = get_pred([fiLayerOutputs])

				# Uncomment below line and comment next two lines for ImageNet models
				# return pred
				labels = np.argmax(pred, axis=-1)
				return labels[0]

				logging.info("Completed injections... exiting")