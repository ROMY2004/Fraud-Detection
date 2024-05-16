import pandas as pd
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

data = pd.read_csv("Final/processedTrain.csv")
test = pd.read_csv("Final/processedTest.csv")


X = data.drop(['is_fraud'],axis=1)
y = data['is_fraud']

X_train = X
X_test = test.drop(['is_fraud'],axis=1)
y_train = y
y_test = test['is_fraud']

model = SVC(kernel='linear')

model.fit(X_train, y_train)

joblib.dump(model, 'Final/Models/SVMTrainedModel.pkl')


