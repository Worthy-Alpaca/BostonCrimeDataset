from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

class NumberPredictionModel:
	def __init__(self, data) -> None:
		""" requires data returned from prepModel """
		self.X_train, self.x_test, self.Y_train, self.y_test = train_test_split(data.drop(columns=["CaseCount"]), data["CaseCount"], random_state=9)

	def linearRegression(self, plot=False):
		""" #Using LinearRegression  """
		print('#Using LinearRegression')
		from sklearn.linear_model import LinearRegression
		log = LinearRegression()
		log.fit(self.X_train, self.Y_train)
		print('[0]Linear Regression Training Accuracy:', log.score(self.X_train, self.Y_train))
		y_test_pred = log.predict(self.x_test)

		print('Score for test data ', log.score(self.x_test, self.y_test))

		if plot:
			cutoff = 0.7
			y_pred_classes = np.zeros_like(y_test_pred)
			y_pred_classes[y_test_pred > cutoff] = 1
			y_test_classes = np.zeros_like(self.y_test)
			y_test_classes[self.y_test > cutoff] = 1
			cmXG3 = confusion_matrix(self.y_test, y_pred_classes)
			df_cm = pd.DataFrame(cmXG3)
			sns.set(font_scale=1.4)
			sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})
			plt.title('')
			plt.show()

	def KNeighborsClassifier(self, plot=False):
		""" #Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm """
		print('#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm')
		from sklearn.neighbors import KNeighborsClassifier
		knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
		knn.fit(self.X_train, self.Y_train)
		print('[1]K Nearest Neighbor Training Accuracy:', knn.score(self.X_train, self.Y_train))
		y_test_pred = knn.predict(self.x_test)

		print('Score for test data ', knn.score(self.x_test, self.y_test))

		if plot:
			cmXG3 = confusion_matrix(self.y_test, y_test_pred)
			df_cm = pd.DataFrame(cmXG3)
			sns.set(font_scale=1.4)
			sns.heatmap(df_cm, annot=True)
			plt.show()

	def SVClinear(self, plot=False):
		""" #Using SVC method of svm class to use Support Vector Machine Algorithm """
		print('#Using SVC method of svm class to use Support Vector Machine Algorithm')
		from sklearn.svm import SVC
		svc_lin = SVC(kernel='linear', random_state=0)
		svc_lin.fit(self.X_train, self.Y_train)
		print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(self.X_train, self.Y_train))
		y_test_pred = svc_lin.predict(self.x_test)

		print('Score for test data ', svc_lin.score(self.x_test, self.y_test))

		if plot:
			ax = sns.regplot(x=self.y_test, y=y_test_pred, color="g")
			plt.show()

	def SVCrbf(self, plot=False):
		""" #Using SVC method of svm class to use Kernel SVM Algorithm """
		print('#Using SVC method of svm class to use Kernel SVM Algorithm')
		from sklearn.svm import SVC
		svc_rbf = SVC(kernel='rbf', random_state=0)
		svc_rbf.fit(self.X_train, self.Y_train)
		print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(self.X_train, self.Y_train))
		y_test_pred = svc_rbf.predict(self.x_test)

		print('Score for test data ', svc_rbf.score(self.x_test, self.y_test))

		if plot:
			ax = sns.regplot(x=self.y_test, y=y_test_pred, color="g")
			plt.show()

	def naiveBayes(self, plot=False):
		""" #Using GaussianNB method of naive_bayes class to use Naive Bayes Algorithm """
		print('#Using GaussianNB method of naive_bayes class to use Naive Bayes Algorithm')
		from sklearn.naive_bayes import GaussianNB
		gauss = GaussianNB()
		gauss.fit(self.X_train, self.Y_train)
		print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(self.X_train, self.Y_train))
		y_test_pred = gauss.predict(self.x_test)

		print('Score for test data ', gauss.score(self.x_test, self.y_test))

		if plot:
			cmXG3 = confusion_matrix(self.y_test, y_test_pred)
			df_cm = pd.DataFrame(cmXG3)
			sns.set(font_scale=1.4)
			sns.heatmap(df_cm, annot=True)
			plt.show()

	def decisionTree(self, plot=False):
		""" #Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm """
		print('#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm')
		from sklearn.tree import DecisionTreeClassifier
		tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
		tree.fit(self.X_train, self.Y_train)
		print('[5]Decision Tree Classifier Training Accuracy:', tree.score(self.X_train, self.Y_train))
		y_test_pred = tree.predict(self.x_test)

		print('Score for test data ', tree.score(self.x_test, self.y_test))

		if plot:
			ax = sns.regplot(x=self.y_test, y=y_test_pred, color="g")
			plt.show()

	def randomForest(self, plot=False):
		""" #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm """
		print('#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm')
		from sklearn.ensemble import RandomForestClassifier
		forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
		forest.fit(self.X_train, self.Y_train)
		print('[6]Random Forest Classifier Training Accuracy:', forest.score(self.X_train, self.Y_train))
		y_test_pred = forest.predict(self.x_test)

		print('Score for test data ', forest.score(self.x_test, self.y_test))
		print(classification_report(self.y_test, y_test_pred))

		if plot:
			cmXG3 = confusion_matrix(self.y_test, y_test_pred)
			df_cm = pd.DataFrame(cmXG3)
			sns.set(font_scale=1.4)
			sns.heatmap(df_cm, annot=True)
			plt.show()
