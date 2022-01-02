import matplotlib.pyplot as plt
import seaborn as sns

class TimePlot:
	def __init__(self, data) -> None:
		""" Requires transformed data """
		self.data = data

	def countsPerYear(self):
		""" Counts per year """
		data = self.data
		year_count = []
		for i in data.Year.unique():
			year_count.append(len(data[data['Year'] == i]))

		plt.figure(figsize=(12, 5))
		sns.pointplot(x=data.Year.unique(), y=year_count, color='blue', alpha=0.8)
		plt.xlabel('Year', fontsize=15)
		plt.xticks(rotation=45)
		plt.ylabel('Crime Count', fontsize=15)
		plt.title('Crime Counts Per Year', fontsize=15)
		plt.grid()
		plt.show()

	def countPlotYear(self):
		""" Crimes per year """
		sns.countplot(data=self.data, x='Year', palette='YlGnBu')
		plt.title('Crimes per year')
		plt.show()

	def monthPlot(self):
		""" Number Of Crimes per Month """
		month_counts = self.data.groupby('Month').count()['Incident_Number'].to_frame().reset_index()
		ax = sns.barplot(x='Month', y="Incident_Number", data=month_counts, palette='YlGnBu')
		plt.title('Number Of Crimes per Month')
		plt.show()

	def dayPlot(self):
		""" Number Of Crimes Each Day_of_Week """
		day_counts = self.data.groupby('Day_Of_Week').count()['Incident_Number'].to_frame().reset_index()
		ax = sns.barplot(x='Day_Of_Week', y="Incident_Number", data=day_counts, palette='YlGnBu')
		plt.title('Number Of Crimes Each Day_of_Week')
		plt.show()

	def hourPlot(self):
		""" Number Of Crimes Each Hour """
		sns.catplot(x='Hour',
             kind='count',
             height=4,
             aspect=3,
             palette='YlGnBu',
             data=self.data)
		plt.title('Number Of Crimes Each Hour')
		plt.xticks(size=10)
		plt.yticks(size=10)
		plt.xlabel('Hour', fontsize=15)
		plt.ylabel('Count', fontsize=15)
		plt.show()

	
