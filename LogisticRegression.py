import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
# from imblearn.over_sampling import SMOTEN


data = pd.read_csv("processedTrain.csv")
test = pd.read_csv("processedTest.csv")



X = data.drop(['is_fraud'],axis=1)
y = data['is_fraud']

X_train = X
X_test = test.drop(['is_fraud'],axis=1)
y_train = y
y_test = test['is_fraud']

# smote = SMOTEN(random_state=42)

# X_train, y_train = smote.fit_resample(X_train, y_train)

# print("Original target distribution:\n", y_train.value_counts())
# print("Resampled target distribution:\n", y_resampled.value_counts())



model = LogisticRegression()

model.fit(X_train, y_train)

joblib.dump(model, 'LogisticRegressionTrainedModel.pkl')

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


print("Accuracy:", accuracy)
print("Classification Report:\n", report)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

