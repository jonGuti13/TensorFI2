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

class inject():
	def __init__(
		self, model, confFile, log_level="ERROR", **kwargs
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
		fiFunc(model, fiConf, **kwargs)

	def layer_states(self, model, fiConf, **kwargs):

		""" FI in layer states """

		if(fiConf["Mode"] == "single"):

			""" Single layer fault injection mode """

			logging.info("Starting fault injection in a random layer")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			# Choose a random layer for injection
			# If random layer is chosen
			if(fiConf["Layer"] == "N"):
				layernum = random.randint(0, len(model.trainable_variables) - 1)
				#Ojo, los bias están separados de los weights. Ambas son trainable_variables.
				#print("La capa escogida para insertar el peso aleatorio es la número:", layernum)

			# If layer position is specified for flip
			else:
				layernum = int(fiConf["Layer"])

			# Get layer states info
			v = model.trainable_variables[layernum]
			#print("El nombre de la capa es:", v.name)
			# num = v.shape.num_elements()
			#print("La capa tiene", num, "pesos en forma de", v.shape)
			train_variables_numpy = v.numpy()

			if(fiFault == "zeros"):
				fiSz = (fiSz * num) / 100
				fiSz = math.floor(fiSz)

			# Hay dos opciones, o que la capa tenga 4 dimensiones (kernel de convolución) o que la capa tenga 1 dimensión (bias de convolución, mu de BN, sigma de BN).
			if len(v.shape) == 1:
				# Choose the indices for FI
				ind0 = random.sample(range(v.shape[0]), fiSz)
				train_parameter = train_variables_numpy[ind0]

			else:
				# Choose the indices for FI
				ind0 = random.sample(range(v.shape[0]), fiSz)
				ind1 = random.sample(range(v.shape[1]), fiSz)
				ind2 = random.sample(range(v.shape[2]), fiSz)
				ind3 = random.sample(range(v.shape[3]), fiSz)
				train_parameter = train_variables_numpy[ind0, ind1, ind2, ind3]

			# Inject the specified fault into the randomly chosen values
			if(fiFault == "zeros"):
				for item in ind:
					v_[item] = 0.
			elif(fiFault == "random"):
				for item in ind:
					v_[item] = np.random.random()
			elif(fiFault == "bitflips"):

				# If random bit chosen to be flipped
				if(fiConf["Bit"] == "N"):
					pos = random.randint(0, 31)

				# If bit position specified for flip
				else:
					pos = int(fiConf["Bit"])
				train_parameter_fi = bitflip(train_parameter, pos)
				if len(v.shape) == 1:
					train_variables_numpy[ind0] = train_parameter_fi
				else:
					train_variables_numpy[ind0, ind1, ind2, ind3] = train_parameter_fi

			#print("Se ha modificado el bit número:", pos)
			#print("El peso antiguo tenía un valor de", val, "y ahora tiene un valor de", val_)

			v.assign(train_variables_numpy)

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