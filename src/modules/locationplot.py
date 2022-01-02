import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

class LocationPlot:
	def __init__(self, data) -> None:
		""" Requires transformed data """
		self.data = data

	def districtPlot(self):
		""" Shows number of crimes by district """
		plt.subplots(figsize=(15, 6))
		sns.countplot('District', palette='BrBG', data=self.data, edgecolor=sns.color_palette('YlGnBu', 20), order=self.data['District'].value_counts().index)
		plt.xticks(rotation=90)
		plt.title('Number Of Crimes By District')
		plt.show()

	def streetPlot(self):
		""" Number Of Crimes By Street """
		plt.figure(figsize=(12, 5))
		crime_street = self.data.groupby('Street')['Incident_Number'].count().nlargest(10)
		crime_street.plot(kind='bar', color="saddlebrown")
		plt.xlabel("Street")
		plt.ylabel("Offense Amount")
		plt.title('Number Of Crimes By Street')
		plt.show()

	def showDistricts(self):
		""" Districts by Lat and Long """
		plt.figure(figsize=(7, 7))
		sp = self.data[(self.data['Lat'] != -1) & (self.data['Long'] != -1)]
		sns.scatterplot(x="Lat", y="Long", hue='District', data=sp)
		plt.show()

	def showGroupedDistricts(self):
		""" Districts by Lat and Long """
		plt.figure(figsize=(7, 7))
		self.data["District"] = self.data["District"].map({

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
		sp = self.data[(self.data['Lat'] != -1) & (self.data['Long'] != -1)]
		sns.scatterplot(x="Lat", y="Long", hue='District', data=sp)
		plt.show()

	def crimePlot(self):
		""" Crimes by numbers """
		plt.subplots(figsize=(15, 6))
		sns.countplot('Offense_Code_Group', palette='BrBG', data=self.data, edgecolor=sns.color_palette('YlGnBu', 20), order=self.data['Offense_Code_Group'].value_counts().index)
		plt.xticks(rotation=90)
		plt.title('Types of serious crimes')
		plt.show()

	def codeByDistrict(self):
		""" Crimes by districts """
		plt.figure(figsize=(20, 10))
		order2 = self.data['Offense_Code_Group'].value_counts().head(6).index
		sns.countplot(data=self.data, x='Offense_Code_Group', hue='District', order=order2, palette='BrBG')
		plt.ylabel("Offense Amount")
		plt.show()

	def heatMapMVAR(self):
		""" Heatmap of Motor Vehicle Accident Response """
		ds = self.data.dropna(subset=['Lat', 'Long', 'District'])
		ds = ds[ds['Offense_Code_Group'] == 'Motor Vehicle Accident Response']
		location = pd.DataFrame(data=(ds.groupby(["Lat", "Long"]).count()[['Incident_Number']]).reset_index().values, columns=["Lat", "Long", "Incident_Number"])
		x, y = location['Long'], location['Lat']
		fig = px.density_mapbox(location, lat="Lat", lon="Long", z="Incident_Number", radius=10, center=dict(lat=42.32475, lon=-71.076), zoom=10, mapbox_style="stamen-terrain", height=500, width=1450)
		fig.show()

	def heatMapLA(self):
		""" Heatmap of Larceny """
		ds = self.data.dropna(subset=['Lat', 'Long', 'District'])
		ds = ds[ds['Offense_Code_Group'] == 'Larceny']
		location = pd.DataFrame(data=(ds.groupby(["Lat", "Long"]).count()[['Incident_Number']]).reset_index().values, columns=["Lat", "Long", "Incident_Number"])
		x, y = location['Long'], location['Lat']
		fig = px.density_mapbox(location, lat="Lat", lon="Long", z="Incident_Number", radius=10, center=dict(lat=42.32475, lon=-71.076), zoom=10, mapbox_style="stamen-terrain", height=500, width=1450)
		fig.show()

	def heatMapMA(self):
		""" Heatmap of Medical Asssistance """
		ds = self.data.dropna(subset=['Lat', 'Long', 'District'])
		ds = ds[ds['Offense_Code_Group'] == 'Medical Assistance']
		location = pd.DataFrame(data=(ds.groupby(["Lat", "Long"]).count()[['Incident_Number']]).reset_index().values, columns=["Lat", "Long", "Incident_Number"])
		x, y = location['Long'], location['Lat']
		fig = px.density_mapbox(location, lat="Lat", lon="Long", z="Incident_Number", radius=10, center=dict(lat=42.32475, lon=-71.076), zoom=10, mapbox_style="stamen-terrain", height=500, width=1450)
		fig.show()
