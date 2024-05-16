
import pandas as pd
import xgboost as xgb
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



model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)

model.fit(X_train, y_train)

joblib.dump(model, 'XGBoostTrainedModel.pkl')

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


print("Accuracy:", accuracy)
print("Classification Report:\n", report)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# implementation of the gradient boosted trees algorithm
# supervised learning algorithm
# combining the estimates of a set of simpler, weaker models.
