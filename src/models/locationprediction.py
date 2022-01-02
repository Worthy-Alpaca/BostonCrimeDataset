from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn import preprocessing
from xgboost import plot_importance
import pandas as pd
import seaborn as sn

import matplotlib.pyplot as plt

class LocationPredictionModel:
	def __init__(self, data) -> None:
		""" Requires encoded data """
		self.data = data

	def withLatAndLong(self) -> None:
		""" Predict District with Lat and Long provided """
		data = self.data
		dataD2 = data[['Offense_Code_Group', 'District', 'Year', 'Seasons', 'Hour', 'Shooting', "Ucr_Parts", "Lat", "Long"]]
		le = preprocessing.LabelEncoder()
		dataD2['Offense_Code_Group'] = le.fit_transform(dataD2['Offense_Code_Group'])
		X_train, X_test, y_train, y_test = train_test_split(dataD2.drop(["District"], axis=1), dataD2["District"], test_size=0.20, random_state=42)
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		xgBC = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5, random_state=0)
		xgBC.fit(X_train, y_train)
		y_predXG2 = xgBC.predict(X_test)
		cmXG = confusion_matrix(y_test, y_predXG2)
		print(classification_report(y_test, y_predXG2))
		df_cm = pd.DataFrame(cmXG)
		sn.set(font_scale=1.4)
		sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
		plt.show()

	def withOutLatAndLong(self):
		""" Predict District without Lat and Long provided """
		data = self.data
		data["Occurred_On_Date"] = data["Occurred_On_Date"].apply(pd.to_datetime, errors='coerce')
		data['DayofMonth'] = data['Occurred_On_Date'].dt.day
		dataD = data[['Offense_Code_Group', 'District',   'Year', 'Seasons', 'Hour', 'Shooting', 'DayOfWeek', "Ucr_Parts", "DayofMonth"]]
		le = preprocessing.LabelEncoder()
		dataD['Offense_Code_Group'] = le.fit_transform(dataD['Offense_Code_Group'])
		X_train, X_test, y_train, y_test = train_test_split(dataD.drop(["District"], axis=1), dataD["District"], test_size=0.20, random_state=42)
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		xgBC = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5, random_state=0)
		xgBC.fit(X_train, y_train)
		y_predXG = xgBC.predict(X_test)
		cmXG = confusion_matrix(y_test, y_predXG)
		print(classification_report(y_test, y_predXG))
		plot_importance(xgBC).set_yticklabels(["Offense_Code_Group", "Year", "Seasons", "Hour", "Shooting", "DayOfWeek", "Ucr_Parts", "DayofMonth"])
		df_cm = pd.DataFrame(cmXG)
		sn.set(font_scale=1.4) 
		sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
		plt.title('')
		plt.show()

	def groupDistrict(self):
		""" Predict district after grouping the districts """
		data = self.data
		data["Occurred_On_Date"] = data["Occurred_On_Date"].apply(pd.to_datetime, errors='coerce')
		dataD3 = data[['Offense_Code_Group', 'District', 'Year', 'Seasons', 'Hour', 'Shooting', 'DayOfWeek', "Ucr_Parts"]]
		dataD3['DayofMonth'] = data['Occurred_On_Date'].dt.day
		le = preprocessing.LabelEncoder()
		dataD3['Offense_Code_Group'] = le.fit_transform(dataD3['Offense_Code_Group'])
		dataD3["District"] = dataD3["District"].map({

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
		X_train, X_test, y_train, y_test = train_test_split(dataD3.drop(["District"], axis=1), dataD3["District"], test_size=0.20, random_state=42)
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		xgBC2 = XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=5, random_state=0)
		xgBC2.fit(X_train, y_train)
		y_predXG3 = xgBC2.predict(X_test)
		cmXG3 = confusion_matrix(y_test, y_predXG3)
		print(classification_report(y_test, y_predXG3))
		plot_importance(xgBC2).set_yticklabels(["Year", "Seasons", "Hour", "Shooting", "DayOfWeek", "Ucr_Parts", "DayofMonth", "District"])
		df_cm = pd.DataFrame(cmXG3)
		sn.set(font_scale=1.4)
		sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
		plt.title('')
		plt.show()
