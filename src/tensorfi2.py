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

def float_to_bin(f):
    # Pack the float into 4 bytes using IEEE-754 single-precision format
    packed = pack('>f', f)
    # Unpack the bytes as integers
    integers = unpack('>I', packed)
    # Convert the integer to binary representation
    binary = bin(integers[0])[2:].zfill(32)
    return binary

def binary_to_float(binary_str):
    # Determine the sign bit, exponent bits, and mantissa bits
    sign_bit = int(binary_str[0])
    exponent_bits = binary_str[1:9]
    mantissa_bits = binary_str[9:]

    # Convert binary parts to decimal
    exponent = int(exponent_bits, 2) - 127
    mantissa = 1  # Implicit leading 1 in IEEE 754 format

    for i in range(len(mantissa_bits)):
        if mantissa_bits[i] == '1':
            mantissa += 2 ** -(i + 1)

    # Calculate the final floating-point value
    result = (-1) ** sign_bit * mantissa * (2 ** exponent)

    return result

def protect(float_parameter, bit, valor, status):
	#Returns the proposed value to protect float_parameter by

	#Sanity check
	if (float_parameter != valor):
		raise("ERROR")
	else:
		binary_representation = float_to_bin(float_parameter)
		signo = binary_representation[0]
		exponente_viejo = binary_representation[1:9]

		if status == "Empty":
			#Quitar uno al exponente, y llenar la mantissa de 1
			exponente_nuevo = '0' + bin((int(exponente_viejo, 2) - 1))[2:]
			mantissa_nueva = '1' * 23

		elif status == "Full":
			#Sumar uno al exponente, y llenar la mantissa de 0
			exponente_nuevo = '0' + bin((int(exponente_viejo, 2) + 1))[2:]
			mantissa_nueva = '0' * 23
		else:
			raise("ERROR")
		#print("- - - - - - - - - - - ")
		#print("Old:", float_parameter)
		#print("New:", binary_to_float(signo + exponente_nuevo + mantissa_nueva))

	return binary_to_float(signo + exponente_nuevo + mantissa_nueva)

def check_exponent_bits_completiveness(binary_representation):
	# Calculate if the exponent is 1-bit far or 2-bit far from completiveness (the eight bits being '1')
	# We're are only interested in the following situations:
		# 1-bit far if that bit is the leftmost one	(from \pm 1 to \pm inf or from 1._____ to Nan)
			#0111 1111 (1) substraction of the exponent is only allowed --> 0111 1110 (0.5)
		# 2-bit far if one of the bits is the leftmost one (from smaller numbers than 1 to higher numbers than 1, leaving the sign bit apart)
			#0011 1111 (5.42101086243e-20) both substraction and addition of the exponent are allowed --> 0011 1110 (2.71050543121e-20) and 0100 0000 (1.08420217249e-19)
			#0101 1111 (2.32830643654e-10) both substraction and addition of the exponent are allowed --> 0101 1110 (1.16415321827e-10) and 0110 0000 (4.65661287308e-10)
			#0110 1111 (1.52587890625e-05) both substraction and addition of the exponent are allowed --> 0110 1110 (7.62939453125e-06)	and 0111 0000 (3.0517578125e-05)
			#0111 0111 (0.00390625)		   both substraction and addition of the exponent are allowed --> 0111 0110 (0.001953125)		and 0111 1000 (0.0078125)
			#0111 1011 (0.0625)			   both substraction and addition of the exponent are allowed --> 0111 1010 (0.03125)			and 0111 1100 (0.125)
			#0111 1101 (0.25)              only substraction of the exponent is allowed               --> 0111 1100 (0.125)				and 0111 1110 (0.5)
			#0111 1110 case is not useful for us

	# The function returns 'Restar', bitLeftPos or 'Ambos', bitLeftPos or None, None

	exponent_bits = binary_representation[1:9]

    # Check if the first bit of the exponent is 0
	if exponent_bits[0] == '0':
    	# Check if the rest of the exponent bits are all '1' except one
		if exponent_bits[1:].count('1') == 7:									#Caso válido	 = 1
			return 'Ambos', 0
		elif exponent_bits[1:].count('1') == 6:
			if exponent_bits[7] == '0':											#Descartar caso 0111 1110 = 0.5
				return None, None
			elif exponent_bits[6] == '0':
				return 'Restar', 6												#Caso válido	0111 1101 = 0.25
			else:
				return 'Ambos', (exponent_bits[1:].find('0') + 1)				#Resto de casos válidos
		else:
			return None, None
	return None, None

def check_mantissa_bits_status(binary_representation, lim_sup, lim_inf):
    # Calculate if the mantissa value is higher than lim_sup or lower than lim_inf
    # thresholds (it is never smaller than 1 by definition)
    # If it is higher, it is considered to be "full", so return values are:
    # 	"Full", mantissa_value
    # If it is smaller, it is considered to be "empty", so return values are:
    #	"Empty", mantissa_value
    # If it is neither higher nor smaller, the return values are:
    #	False, None

	mantissa_bits = binary_representation[9:32]
	mantissa_value = 1

	for i in range(len(mantissa_bits)):
		mantissa_value += int(mantissa_bits[i]) * 2**(-(i+1))

	if mantissa_value >= lim_sup:
		return "Full", mantissa_value
	elif mantissa_value <= lim_inf:
		return "Empty", mantissa_value
	else:
		return False, None

def print_valuable_information(float_parameter, train_parameter_binary, pos, mantissa_value, status):
	print("El peso tiene un valor de : ", float_parameter)
	print("El exponente es: ", train_parameter_binary[1:9])
	print("El hueco en el exponente se encuentra en la posición (empezando por la izquierda): ", pos)
	print("La mantissa es: ", mantissa_value)
	print("La mantissa está: ", status)
	print(" - - - - - - - - - - - - - - - - - - - - - - - - -")
	return


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
				layernum = random.randint(0, len(model.trainable_variables) - 1)
				#Ojo, los bias están separados de los weights. Ambas son trainable_variables.
				#print("La capa escogida para insertar el peso aleatorio es la número:", layernum)

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

			#print("El nombre de la capa es:", v.name)
			# num = v.shape.num_elements()
			#print("La capa tiene", num, "pesos en forma de", v.shape)

			if(fiFault == "zeros"):
				fiSz = (fiSz * num) / 100
				fiSz = math.floor(fiSz)

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
					if (fiFormat == "fp32" | fiFormat == "int32"):
						pos = random.randint(0, 31)
					elif fiFormat == "int8":
						pos = random.randint(0, 7)
					else:
						raise("Formato mal especificado")

				# If bit position specified for flip
				else:
					pos = int(fiConf["Bit"])
					# Hay dos opciones, o que la capa tenga 4 dimensiones (kernel de convolución) o que la capa tenga 1 dimensión (bias de convolución, mu de BN, sigma de BN).
					if len(v.shape) == 1:
						if fiFormat == "int32":
          					#Los datos de 32 bits se almacenan en 4 bytes por lo que si quiero cambiar un bit concreto tengo que elegir el byte adecuado
							#Para el caso particular en el que hay 52 bias de 32 bits, tenemos 52*4 valores de 32/4 bits, por lo que v.shape[0] = 208.
							#v[0], v[1], v[2], v[3] forman el primer elemento así v[3] * 2^24 + v[2] * 2^16 + v[1] * 2^8 + v[0]
							#mult se encarga de seleccionar 1 de esos 52 bias y nos da un valor entre 0 y 51. 4*mult nos pone al inicio de los 4 elementos que conforman un bias.
							#Dependiendo del bit que se quiera cambiar, nos desplazamos 3, 2, 1 o 0 posiciones.
							mult = random.sample(range(int(v.shape[0]/4)), fiSz)[0]
							if (pos >= 0) & (pos < 8):
								ind0 = 0 + 4 * mult
							elif (pos >= 8) & (pos < 16):
								ind0 = 1 + 4 * mult
							elif (pos >= 16) & (pos < 24):
								ind0 = 2 + 4 * mult
							elif (pos >= 24) & (pos < 32):
								ind0 = 3 + 4 * mult
						else:
							# Choose the indices for FI
							ind0 = random.sample(range(0, v.shape[0]), fiSz)
						train_parameter = train_variables_numpy[ind0]

					else:
						# Choose the indices for FI
						ind0 = random.sample(range(v.shape[0]), fiSz)
						ind1 = random.sample(range(v.shape[1]), fiSz)
						ind2 = random.sample(range(v.shape[2]), fiSz)
						ind3 = random.sample(range(v.shape[3]), fiSz)
						train_parameter = train_variables_numpy[ind0, ind1, ind2, ind3]

				if fiFormat == "fp32":
					train_parameter_fi = bitflip(train_parameter, pos)
				elif fiFormat == "int32":
					train_parameter_fi = bitflip_int8(train_parameter.view(dtype=np.int8), pos % 8)
				elif fiFormat == "int8":
					train_parameter_fi = bitflip_int8(train_parameter[0].view(dtype=np.int8), pos)
				else:
					raise("Formato mal especificado")

				if len(v.shape) == 1:
					train_variables_numpy[ind0] = train_parameter_fi
				else:
					train_variables_numpy[ind0, ind1, ind2, ind3] = train_parameter_fi

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


	def custom_layer_states(self, model, fiConf, interpreter=None, **kwargs):

		""" FI in layer states """

		if(fiConf["Mode"] == "single"):

			""" Single layer fault injection mode """

			logging.info("Starting fault injection in a random layer")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]
			fiFormat = fiConf["Format"]

			layernum = int(fiConf["Layer"])

			# Get layer states info
			if (fiFormat == "fp32"):
				v = model.trainable_variables[layernum]
				train_variables_numpy = v.numpy()
			else:
				raise("Not implemented yet")

			#print("El nombre de la capa es:", v.name)
			# num = v.shape.num_elements()
			#print("La capa tiene", num, "pesos en forma de", v.shape)

			# Inject the specified fault into the randomly chosen values
			if(fiFault == "bitflips"):
				pos = int(fiConf["Bit"])
				if len(v.shape) == 1:
					# Choose the indices for FI
					ind0 = fiConf["Ind0"]
					train_parameter = train_variables_numpy[ind0]

				else:
					# Choose the indices for FI
					ind0 = fiConf["Ind0"]
					ind1 = fiConf["Ind1"]
					ind2 = fiConf["Ind2"]
					ind3 = fiConf["Ind3"]
					train_parameter = train_variables_numpy[ind0, ind1, ind2, ind3]

				if fiFormat == "fp32":
					train_parameter_fi = bitflip(train_parameter, pos)
				else:
					raise("Not implemented yet.")

				if len(v.shape) == 1:
					train_variables_numpy[ind0] = train_parameter_fi
				else:
					train_variables_numpy[ind0, ind1, ind2, ind3] = train_parameter_fi
			else:
				raise("Not implemented yet.")

			if (fiFormat == "fp32"):
				v.assign(train_variables_numpy)
			else:
				None

			logging.info("Completed injections... exiting")

		else:
			raise("Not implemented yet")


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

class other_actions():
	def __init__(
		self, model, confFile, infoFile=None, log_level="ERROR", **kwargs
		):

		# Logging setup
		logging.basicConfig()
		logging.getLogger().setLevel(log_level)
		logging.debug("Logging level set to {0}".format(log_level))

		# Retrieve config params
		fiConf = config.config(confFile)
		self.Model = model # No more passing or using a session variable in TF v2

		# Call the corresponding FI function
		fiFunc = getattr(self, fiConf["Action"])
		fiFunc(model, fiConf, infoFile, **kwargs)

	def completiveness_analysis(self, model, fiConf, infoFile = None, **kwargs):

		#Print number of trainable variables
		num_trainable_variables = len(model.trainable_variables)
		contBias = 0
		contPeso = 0
		info = []

		for layernum in range(num_trainable_variables):
			v = model.trainable_variables[layernum]
			train_variables_numpy = v.numpy()
			lim_sup = fiConf['Lim_sup']
			lim_inf = fiConf['Lim_inf']

			if len(v.shape) == 1:
				for i in range(v.shape[0]):
					float_parameter = train_variables_numpy[i]
					train_parameter_binary = float_to_bin(float_parameter)
					completiveness, pos = check_exponent_bits_completiveness(train_parameter_binary)

					if completiveness == None:
						None
					elif completiveness == 'Ambos':
						status, mantissa_value = check_mantissa_bits_status(train_parameter_binary, lim_sup, lim_inf)
						if status == "Empty" or status == "Full":
							contBias += 1
							#print_valuable_information(float_parameter, train_parameter_binary, pos, mantissa_value, status)
							info.append([layernum, i, None, None, None, pos, float_parameter, status])
					elif completiveness == 'Restar':
						status, mantissa_value = check_mantissa_bits_status(train_parameter_binary, lim_sup, lim_inf)
						if status == "Empty":
							contBias += 1
							#print_valuable_information(float_parameter, train_parameter_binary, pos, mantissa_value, status)
							info.append([layernum, i, None, None, None, pos, float_parameter, status])
					else:
						raise

			else:
				for i in range(v.shape[0]):
					for j in range(v.shape[1]):
						for k in range(v.shape[2]):
							for l in range(v.shape[3]):
								float_parameter = train_variables_numpy[i, j, k, l]
								train_parameter_binary = float_to_bin(float_parameter)
								completiveness, pos = check_exponent_bits_completiveness(train_parameter_binary)

								if completiveness == None:
									None
								elif completiveness == 'Ambos':
									status, mantissa_value = check_mantissa_bits_status(train_parameter_binary, lim_sup, lim_inf)
									if status == "Empty" or status == "Full":
										contPeso += 1
										#print_valuable_information(float_parameter, train_parameter_binary, pos, mantissa_value, status)
										info.append([layernum, i, j, k, l, pos, float_parameter, status])
								elif completiveness == 'Restar':
									status, mantissa_value = check_mantissa_bits_status(train_parameter_binary, lim_sup, lim_inf)
									if status == "Empty":
										contPeso += 1
										#print_valuable_information(float_parameter, train_parameter_binary, pos, mantissa_value, status)
										info.append([layernum, i, j, k, l, pos, float_parameter, status])
								else:
									raise

		print("Bias: ", contBias)
		print("Peso: ", contPeso)
		np.save(infoFile, info)

	def protection(self, model, fiConf=None, infoFile=None, **kwargs):
		#El objetivo de este método es cambiar los valores de ciertos parámetros para protegerlos del fault injection.
		#Debe abrir un fichero .py en el que cada item de la lista contiene información sobre el parámetro a proteger y cómo hacerlo.
		#[layernum, ind0, ind1, ind2, ind3, bit, valor, status]
		#layernum: numero de capa en la que se encuentra el parámetro
		#id0, id1, id2, id3: posición del parámetro dentro de la capa.
		# 	Si id1, id2 e id3 son None, sabemos que el parámetro no es un peso (puede ser bias, gamma o beta)
		#bit: posicion del exponente que se encuentra vacía
		#valor: valor del parámetro
		#status: estado de la mantissa.
		#	Si el estado es "Full" lo que hay que hacer es sumar un bit al exponente y vaciar la mantissa.
		#	Si el estado es "Empty" lo que hay que hacer es quitar un bit al exponente y llenar la mantissa.

		info = np.load(infoFile, allow_pickle=True)
		for i, information in enumerate(info):
			layernum = information[0]
			v = model.trainable_variables[layernum]
			train_variables_numpy = v.numpy()

			ind0 = information[1]
			ind1 = information[2]
			ind2 = information[3]
			ind3 = information[4]
			bit = information[5]
			valor = information[6]
			status = information[7]

			if (ind1 != None) and (ind2 != None) and (ind3 != None):
				float_parameter = train_variables_numpy[ind0, ind1, ind2, ind3]
				train_variables_numpy[ind0, ind1, ind2, ind3] = protect(float_parameter, bit, valor, status)
			else:
				float_parameter = train_variables_numpy[ind0]
				train_variables_numpy[ind0] = protect(float_parameter, bit, valor, status)

			v.assign(train_variables_numpy)