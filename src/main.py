import os

from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

""" ############### """
from modules import *
from models import *


class Controller:
	def __init__(self, option) -> None:
		""" option can be Plot, PredictNumbers or PredictDistrict """
		if option == 'Plot':
			""" Use this section to create Plots """
			self.data = DataLoader('PredictNumbers').data
			for key, dtype in self.data.dtypes.iteritems():
				print(f"{key} & {dtype} \\\\")
			LocationPlot(self.data).showDistricts()
			TimePlot(self.data).countsPerYear()

		elif option == 'PredictNumbers':
			""" Use this section to Predict crime numbers """
			self.data = DataLoader(option).data
			model = NumberPredictionModel(self.data)
			model.randomForest(plot=True)
			model.decisionTree(plot=False)
			model.naiveBayes(plot=True)
			model.KNeighborsClassifier(plot=True)
			model.linearRegression(plot=True)
			model.SVClinear()
			model.SVCrbf()

		elif option == 'PredictDistrict':
			""" Use this section to predict crime districts """
			self.data = DataLoader(option).data
			model = LocationPredictionModel(self.data)
			model.withLatAndLong()
			model.withOutLatAndLong()
			model.groupDistrict()

		else:
			self.data = DataLoader('PredictDistrict').data
			TimePredictionModel(self.data)



if __name__ == '__main__':
	""" Change option in corresponding section in __init__ """
	Controller('Plot')
	Controller('PredictNumbers')
	Controller('PredictDistrict')
	Controller('else')

	""" Comment this out to access the tensorflow model below """
	exit()

	""" ########## TENSORFLOW MODEL ########## """
	""" The Feature names of the dataset """
	FEATURE_COLUMNS = ['Offense_Code_Group', 'District', 'Year', 'Seasons', 'Hour', 'Shooting', 'DayOfWeek', "Ucr_Parts", "DayofMonth"]

	""" Instantiate the model """
	""" Change variables tweak accuracy """
	model = Tensor(50, len(FEATURE_COLUMNS), activation="tanh", loss="msle", units=256, cell=LSTM, n_layers=5, dropout=0.7, optimizer='adam', bidirectional=False).model()
	
	""" Load the data from the dataset """
	""" If unsure, only put crime.csv into assets folder """
	data = DataLoader().load_data(split_by_date=False, feature_columns=FEATURE_COLUMNS, group_districts=True, n_steps=50, shuffle=False)
	
	model_name = 'DataScienceTest' 
	
	""" Create callbacks for use in the model """
	checkpointer = ModelCheckpoint(os.path.join("tensor_results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
	tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
	
	""" Print a summary of the model """
	print(model.summary())

	""" Train the model """
	""" If you just want to use a previously trained model, comment this part out """
	#"""
	history = model.fit(data['X_train'], data["y_train"],
                     batch_size=64,
                     epochs=3,
                     validation_data=(data["X_test"], data["y_test"]),
                     callbacks=[checkpointer, tensorboard],
                     verbose=1,
					 use_multiprocessing=True)
	#"""

	""" Load the trained model """
	model_path = os.path.join("tensor_results", model_name) + ".h5"
	model.load_weights(model_path)

	""" Now we can predict things """
	y_pred = model.predict(data['X_test'])
	
	""" Grab loss, accuracy, f1_score, precision and recall """
	loss, accuracy, f1_score, precision, recall = model.evaluate(data["X_test"], data["y_test"], verbose=0)

	""" Print Results """
	print(f'Loss: {loss}')
	print(f'Accuracy: {accuracy}')
	print(f'F1 score: {f1_score} ')
	print(f'Precision: {precision} ')
	print(f'Recall: {recall} ')
