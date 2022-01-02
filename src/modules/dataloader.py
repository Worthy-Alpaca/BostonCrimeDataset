import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
from sklearn.model_selection import train_test_split
from collections import deque
from modules import FeatureEncoder
import datetime as dt
pd.options.mode.chained_assignment = None

class DataLoader:
	def __init__(self, option='Tensorflow') -> None:
		""" Loads Data and transforms it when needed """
		PACKAGE_PARENT = '../..'
		SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
		self.PATH = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT, 'assets'))
		content = os.listdir(self.PATH)
		if option == 'Tensorflow' or option == 'PredictDistrict':
			if 'encoded.csv' in content:
				print('Found encoded data set')
				self.data = pd.read_csv(os.path.join(self.PATH, 'encoded.csv'))
			else:
				if 'transformed.csv'in content:
					print('Found transformed.csv')
					print('Starting encoding')
					data = pd.read_csv(os.path.join(self.PATH, 'transformed.csv'))
					encodedData = FeatureEncoder(data).data
					encodedData.to_csv(os.path.join(self.PATH, 'encoded.csv'))
					self.data = encodedData
				else:
					print('Found crime.csv')
					data = pd.read_csv(os.path.join(self.PATH, 'crime.csv'))
					transformed = self.transform(data)
					encodedData = FeatureEncoder(transformed).data
					encodedData.to_csv(os.path.join(self.PATH, 'encoded.csv'))
					self.data = encodedData
		else:
			if option == 'Plot':
				if 'transformed.csv' in content:
					print('Found transformed.csv')
					self.data = pd.read_csv(os.path.join(self.PATH, 'transformed.csv'))
				else:
					print('Found crime.csv')
					data = pd.read_csv(os.path.join(self.PATH, 'crime.csv'))
					self.data = self.transform(data)
			elif option == 'PredictNumbers':
				if 'encoded.csv' in content:
					print('Found encoded data set')
					encodedData = pd.read_csv(os.path.join(self.PATH, 'encoded.csv'))
				else:
					if 'transformed.csv' in content:
						print('Found transformed.csv')
						print('Starting encoding')
						data = pd.read_csv(os.path.join(self.PATH, 'transformed.csv'))
						encodedData = FeatureEncoder(data).data
						encodedData.to_csv(os.path.join(self.PATH, 'encoded.csv'))
					else:
						print('Found crime.csv')
						data = pd.read_csv(os.path.join(self.PATH, 'crime.csv'))
						transformed = self.transform(data)
						encodedData = FeatureEncoder(transformed).data
						encodedData.to_csv(os.path.join(self.PATH, 'encoded.csv'))
				
				self.data = self.prepModel(encodedData)
	
	def transform(self, data):
		""" Needs default Dataset """
		print('starting transform')
		data.rename(columns={"INCIDENT_NUMBER": "Incident_Number",
                       "OFFENSE_CODE": "Offense_Code",
                       "OFFENSE_CODE_GROUP": "Offense_Code_Group",
                       "OFFENSE_DESCRIPTION": "Offense_Description",
                       "DISTRICT": "District",
                       "REPORTING_AREA": "Reporting_Area",
                       "SHOOTING": "Shooting",
                       "OCCURRED_ON_DATE": "Occurred_On_Date",
                       "YEAR": "Year", "MONTH": "Month",
                       "DAY_OF_WEEK": "Day_Of_Week", "HOUR": "Hour",
                       "UCR_PART": "Ucr_Part",
                       "STREET": "Street"
                       }, inplace=True)

		def getSeason(month):
			if (month == 12 or month == 1 or month == 2):
				return "Winter"
			elif(month == 3 or month == 4 or month == 5):
				return "Spring"
			elif(month == 6 or month == 7 or month == 8):
				return "Summer"
			else:
				return "Fall"
		data["Occurred_On_Date"] = data["Occurred_On_Date"].apply(pd.to_datetime, errors='coerce')
		data["Occurred_On_Date"] = data["Occurred_On_Date"].dt.date
		data["Occurred_On_Date"] = data["Occurred_On_Date"].apply(pd.to_datetime, errors='coerce')
		data.Shooting.fillna('N', inplace=True)
		data = data.dropna()
		data['Season'] = data.Month.apply(getSeason)
		print('finished transform')
		data.to_csv(os.path.join(self.PATH, 'transformed.csv'))
		return data

	def prepModel(self, encodedData):
		""" Needs the encoded Dataset """
		dataR = pd.DataFrame(encodedData.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataR["Occurred_On_Date"] = dataR["Occurred_On_Date"].apply(pd.to_datetime, errors='coerce')
		dataR['Day'] = dataR['Occurred_On_Date'].dt.dayofweek
		days = (1, 2, 3, 4, 5, 6, 7)
		dataR['Day'] = dataR['Day'].apply(lambda x: days[x])
		dataR.rename(columns={'Occurred_On_Date': 'OccuredDate', 'Incident_Number': 'CaseCount', 'Day': 'DayOfWeek'}, inplace=True)
		dataR = pd.concat([dataR, pd.get_dummies(dataR['District'], prefix='D')], axis=1)
		dataR.drop(['District'], axis=1, inplace=True)
		dataR["D_A1"] = np.int64(dataR["D_A1"])                      # convert uint8 to int64
		dataR["D_A15"] = np.int64(dataR["D_A15"])
		dataR["D_A7"] = np.int64(dataR["D_A7"])
		dataR["D_B2"] = np.int64(dataR["D_B2"])
		dataR["D_B3"] = np.int64(dataR["D_B3"])
		dataR["D_C11"] = np.int64(dataR["D_C11"])
		dataR["D_C6"] = np.int64(dataR["D_C6"])
		dataR["D_D14"] = np.int64(dataR["D_D14"])
		dataR["D_D4"] = np.int64(dataR["D_D4"])
		dataR["D_E13"] = np.int64(dataR["D_E13"])
		dataR["D_E18"] = np.int64(dataR["D_E18"])
		dataR["D_E5"] = np.int64(dataR["D_E5"])
		dataR['OccuredDate'] = pd.to_datetime(dataR['OccuredDate'])
		dataR['OccuredDate'] = dataR['OccuredDate'].map(dt.datetime.toordinal)
		return dataR
	
	def load_data(self, group_districts=True, n_steps=50, scale=True, shuffle=True, lookup_step=1, split_by_date=True,
              test_size=0.2, feature_columns=['Offense_Code_Group', 'District', 'Year', 'Seasons', 'Hour', 'Shooting', 'DayOfWeek', "Ucr_Parts", "DayofMonth"]):
		""" Prepare data for Tensorflow model.
		Returns dataset ready for use """
		data = self.data
		print('Preparing data for Tensorflow model')
		df = data[['Offense_Code_Group', 'District', 'Year', 'Seasons', 'Hour', 'Shooting', 'DayOfWeek', "Ucr_Parts", "DayofMonth"]]
		le = preprocessing.LabelEncoder()
		if group_districts:
			df['District'] = df["District"].map({

						"E18": 1,
						"E5": 1,
						"E13": 1,
						"B3": 1,

						"D4": 2,
						"B2": 2,
						"C11": 2,
						"C6": 2,

						"A15": 3,
						"A1": 3,
						"A7": 3,
						"D14": 3,

					})
		else:
			df['District'] = le.fit_transform(df['District'])
		
		result = {}
		
		result['df'] = df.copy()
		
		for col in feature_columns:
			assert col in df.columns, f"'{col}' does not exist in the dataframe."
		
		if "DayofMonth" not in df.columns:
			df["DayofMonth"] = df.index
		if scale:
			column_scaler = {}
			for column in feature_columns:
				scaler = preprocessing.MinMaxScaler()
				df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
				column_scaler[column] = scaler

			result["column_scaler"] = column_scaler

		last_sequence = np.array(df[feature_columns].tail(lookup_step))

		df.dropna(inplace=True)
		sequence_data = []
		sequences = deque(maxlen=n_steps)
		for entry, target in zip(df[feature_columns + ["DayofMonth"]].values, df['District']):
			sequences.append(entry)
			if len(sequences) == n_steps:
				sequence_data.append([np.array(sequences), target])

		last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
		last_sequence = np.array(last_sequence).astype(np.float32)

		result['last_sequence'] = last_sequence

		X, y = [], []
		for seq, target in sequence_data:
			X.append(seq)
			y.append(target)
		
		def shuffle_in_unison(a, b):

			state = np.random.get_state()
			np.random.shuffle(a)
			np.random.set_state(state)
			np.random.shuffle(b)

		X = np.array(X)
		y = np.array(y)

		if split_by_date:

			train_samples = int((1 - test_size) * len(X))
			result["X_train"] = X[:train_samples]
			result["y_train"] = y[:train_samples]
			result["X_test"] = X[train_samples:]
			result["y_test"] = y[train_samples:]
			if shuffle:

				shuffle_in_unison(result["X_train"], result["y_train"])
				shuffle_in_unison(result["X_test"], result["y_test"])
		else:

			result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

		result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
		result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
		return result
