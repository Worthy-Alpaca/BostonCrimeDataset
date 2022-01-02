from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

class TimePredictionModel:
	def __init__(self, data) -> None:
		data['Occurred_On_Date'] = pd.to_datetime(data['Occurred_On_Date'])
		""" Split the data by districts """
		dataD4 = data.loc[data['District'] == "D4"]
		dataD4 = dataD4.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataD4 = pd.DataFrame(dataD4.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataD4.rename(columns={'Incident_Number': 'countD4', 'Occurred_On_Date': "DateD4"}, inplace=True)
		dataD4 = dataD4.drop("District", axis=1)

		dataD14 = data.loc[data['District'] == "D14"]
		dataD14 = dataD14.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataD14 = pd.DataFrame(dataD14.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataD14.rename(columns={'Incident_Number': 'countD14', 'Occurred_On_Date': "DateD14"}, inplace=True)
		dataD14 = dataD14.drop("District", axis=1)

		dataC11 = data.loc[data['District'] == "C11"]
		dataC11 = dataC11.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataC11 = pd.DataFrame(dataC11.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataC11.rename(columns={'Incident_Number': 'countC11', 'Occurred_On_Date': "DateC11"}, inplace=True)
		dataC11 = dataC11.drop("District", axis=1)

		dataB3 = data.loc[data['District'] == "B3"]
		dataB3 = dataB3.loc[:, ["Incident_Number",'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataB3 = pd.DataFrame(dataB3.groupby(["Occurred_On_Date","District"])["Incident_Number"].count()).reset_index()
		dataB3.rename(columns={ 'Incident_Number': 'countB3', 'Occurred_On_Date': "DateB3"}, inplace=True)
		dataB3 = dataB3.drop("District",axis = 1)


		dataB2 = data.loc[data['District'] == "B2"]
		dataB2 = dataB2.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataB2 = pd.DataFrame(dataB2.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataB2.rename(columns={'Incident_Number': 'countB2', 'Occurred_On_Date': "DateB2"}, inplace=True)
		dataB2 = dataB2.drop("District", axis=1)


		dataC6 = data.loc[data['District'] == "C6"]
		dataC6 = dataC6.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataC6 = pd.DataFrame(dataC6.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataC6.rename(columns={'Incident_Number': 'countC6', 'Occurred_On_Date': "DateC6"}, inplace=True)
		dataC6 = dataC6.drop("District", axis=1)


		dataA1 = data.loc[data['District'] == "A1"]
		dataA1 = dataA1.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataA1 = pd.DataFrame(dataA1.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataA1.rename(columns={'Incident_Number': 'countA1', 'Occurred_On_Date': "DateA1"}, inplace=True)
		dataA1 = dataA1.drop("District", axis=1)


		dataE5 = data.loc[data['District'] == "E5"]
		dataE5 = dataE5.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataE5 = pd.DataFrame(dataE5.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataE5.rename(columns={'Incident_Number': 'countE5', 'Occurred_On_Date': "DateE5"}, inplace=True)
		dataE5 = dataE5.drop("District", axis=1)


		dataA7 = data.loc[data['District'] == "A7"]
		dataA7 = dataA7.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataA7 = pd.DataFrame(dataA7.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataA7.rename(columns={'Incident_Number': 'countA7', 'Occurred_On_Date': "DateA7"}, inplace=True)
		dataA7 = dataA7.drop("District", axis=1)


		dataE13 = data.loc[data['District'] == "E13"]
		dataE13 = dataE13.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataE13 = pd.DataFrame(dataE13.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataE13.rename(columns={'Incident_Number': 'countE13', 'Occurred_On_Date': "DateE13"}, inplace=True)
		dataE13 = dataE13.drop("District", axis=1)


		dataE18 = data.loc[data['District'] == "E18"]
		dataE18 = dataE18.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataE18 = pd.DataFrame(dataE18.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataE18.rename(columns={'Incident_Number': 'countE18', 'Occurred_On_Date': "DateE18"}, inplace=True)
		dataE18 = dataE18.drop("District", axis=1)


		dataA15 = data.loc[data['District'] == "A15"]
		dataA15 = dataA15.loc[:, ["Incident_Number", 'Offense_Code_Group', 'District', 'Occurred_On_Date']]
		dataA15 = pd.DataFrame(dataA15.groupby(["Occurred_On_Date", "District"])["Incident_Number"].count()).reset_index()
		dataA15.rename(columns={'Incident_Number': 'countA15', 'Occurred_On_Date': "DateA15"}, inplace=True)
		dataA15 = dataA15.drop("District", axis=1)

		dataUCR = data.loc[data['Ucr_Parts'] == 3]
		dataUCR = dataUCR.loc[:, ["Incident_Number", 'Ucr_Parts', 'Occurred_On_Date']]
		dataUCR = pd.DataFrame(dataUCR.groupby(["Occurred_On_Date","Ucr_Parts"])["Incident_Number"].count()).reset_index()
		dataUCR.rename(columns={ 'Incident_Number': 'countUCR','Occurred_On_Date': "DateUCR"}, inplace=True)
		dataUCR = dataUCR.drop("Ucr_Parts",axis = 1)

		result = pd.concat([dataUCR, dataD14, dataC11, dataD4, dataB3, dataB2, dataC6, dataA1, dataE5, dataA7, dataE13,
                      dataE18, dataA15], axis=1, sort=False)
		
		result = result.drop(["DateUCR", 'DateC11', 'DateD4', 'DateB3',  'DateB2', 'DateC6',  'DateA1', 'DateE5',  'DateA7',  'DateE13', 'DateE18',  'DateA15'], axis=1)
		""" create datetime objects """
		result['DayofMonth'] = result['DateD14'].dt.day
		result['Month'] = result['DateD14'].dt.month
		result['Weekday'] = result['DateD14'].dt.weekday
		result['DateD14'] = pd.to_datetime(result['DateD14'])
		result['DateD14'] = result['DateD14'].map(dt.datetime.toordinal)
		
		""" remove missing values """
		result.dropna(inplace=True)
		
		""" split the data into train and test """
		x_trainUCR, x_testUCR, y_trainUCR, y_testUCR = train_test_split(result.drop(columns=["countUCR"]), result["countUCR"], random_state=42)
		lr = LinearRegression().fit(x_trainUCR, y_trainUCR)
		lr = LinearRegression().fit(x_trainUCR,y_trainUCR)

		y_train_predUCR = lr.predict(x_trainUCR)
		y_test_predUCR = lr.predict(x_testUCR)

		print(lr.score(x_testUCR,y_testUCR))
		ax = sns.regplot(x=y_testUCR, y=y_test_predUCR, color="g")
		plt.show()

