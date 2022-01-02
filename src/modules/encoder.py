from sklearn import preprocessing
import pandas as pd
class FeatureEncoder:
	def __init__(self, data) -> None:
		""" Encodes the given data. I.E. converting weekdays into numbers. 
		Requires Transformed data"""
		print('Encoding Data')
		self.data = data
		self.dayOfWeek()
		self.isShooting()
		self.ucrEncode()
		self.seasons()
		self.label()
		self.data = self.data.sort_values(by=['Occurred_On_Date'], ascending=False)

	def dayOfWeek(self):
		""" Encodes the days of the week """
		self.data["DayOfWeek"] = self.data["Day_Of_Week"].map({
                    "Monday": 1,
                    "Tuesday": 2,
                    "Wednesday": 3,
                    "Thursday": 4,
                    "Friday": 5,
                    "Saturday": 6,
                    "Sunday": 7
        		})
		
	def seasons(self):
		le = preprocessing.LabelEncoder()
		self.data['Seasons'] = le.fit_transform(self.data['Season'])

	def label(self):
		le = preprocessing.LabelEncoder()
		self.data['Offense_Code_Group'] = le.fit_transform(self.data['Offense_Code_Group'])

		self.data['Occurred_On_Date'] = pd.to_datetime(self.data['Occurred_On_Date'])

		self.data['DayofMonth'] = self.data['Occurred_On_Date'].dt.day
		self.data['Occurred_On_Date'] = pd.to_datetime(self.data['Occurred_On_Date'])

	def isShooting(self):
		""" encodes the shooting column into 1 if yes and 0 if no """
		self.data['Shooting'] = self.data['Shooting'].apply(lambda x: 1 if x == 'Y' else 0)
		

	def ucrEncode(self):
		self.data['Ucr_Parts'] = self.data['Ucr_Part'].map({
			"Part One": 1,
            "Part Two": 2,
            "Part Three": 3,
            "Other": 0
		})
		self.data = self.data.drop("Ucr_Part", axis=1)
