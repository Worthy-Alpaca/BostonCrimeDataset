import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from keras import backend as K
pd.options.mode.chained_assignment = None
tf.get_logger().setLevel('INFO')



class Tensor:
	def __init__(self, sequence_length, n_features, activation="linear", units=256, cell=LSTM, n_layers=2, dropout=0.3,
              loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False) -> Sequential:
		""" Instantiate all options """
		self.sequence_length = sequence_length
		self.n_features = n_features
		self.units = units
		self.cell = cell
		self.n_layers = n_layers
		self.dropout = dropout
		self.loss = loss
		self.optimizer = optimizer
		self.bidirectional = bidirectional
		self.activation = activation

	def model(self):
		""" Returns the instantiated model """
		model = Sequential()
		model.add(Dense(self.units, activation=self.activation))
		for i in range(self.n_layers):
			if i == 0:
				# first layer
				if self.bidirectional:
					model.add(Bidirectional(self.cell(self.units, return_sequences=True), batch_input_shape=(None, self.sequence_length, self.n_features)))
				else:
					model.add(self.cell(self.units, return_sequences=True, batch_input_shape=(None, self.sequence_length, self.n_features)))
			elif i == self.n_layers - 1:
				# last layer
				if self.bidirectional:
					model.add(Bidirectional(self.cell(self.units, return_sequences=False)))
				else:
					model.add(self.cell(self.units, return_sequences=False))
			else:
				# hidden layers
				if self.bidirectional:
					model.add(Bidirectional(self.cell(self.units, return_sequences=True)))
				else:
					model.add(self.cell(self.units, return_sequences=True))
			# add dropout after each layer
			model.add(Dropout(self.dropout))
		model.add(Dense(self.units, activation=self.activation))
		model.add(Dense(1, activation=self.activation))
		model.compile(loss=self.loss, metrics=["mean_absolute_error", self.f1_m, self.precision_m, self.recall_m], optimizer=self.optimizer)
		model.build(input_shape=(None, self.sequence_length, self.n_features))
		return model

	def recall_m(self, y_true, y_pred):
		""" calc the recall """
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall


	def precision_m(self, y_true, y_pred):
		""" calc the precision """
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision


	def f1_m(self, y_true, y_pred):
		""" calc the f1 score """
		precision = self.precision_m(y_true, y_pred)
		recall = self.recall_m(y_true, y_pred)
		return 2*((precision*recall)/(precision+recall+K.epsilon()))


