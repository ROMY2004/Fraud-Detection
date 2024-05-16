import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


model = RandomForestClassifier(n_estimators=100, random_state=42)
print(X_train.info())
model.fit(X_train, y_train)

joblib.dump(model, 'RandomForestTrainedModel.pkl')

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


print("Accuracy:", accuracy)
print("Classification Report:\n", report)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


#  #   Column               Non-Null Count    Dtype  
# ---  ------               --------------    -----  
#  0   Unnamed: 0           1002103 non-null  int64  
#  1   category             1002103 non-null  int64  
#  2   Amount               1002103 non-null  float64
#  3   month                1002103 non-null  int64  
#  4   hour                 1002103 non-null  int64  
#  5   time_diff            1002103 non-null  int64  
#  6   countName            1002103 non-null  int64  
#  7   countNamehour        1002103 non-null  int64  
#  8   countNamesinglehour  1002103 non-null  int64  
#  9   countNamecard        1002103 non-null  int64  
#  10  countcardhour        1002103 non-null  int64  
#  11  dis                  1002103 non-null  float64


#  0   category             1 non-null      int32  
#  1   Amount               1 non-null      float64
#  2   month                1 non-null      int32  
#  3   hour                 1 non-null      int32  
#  4   time_diff            1 non-null      int64  
#  5   countName            1 non-null      int64  
#  6   countNamehour        1 non-null      int64  
#  7   countNamesinglehour  1 non-null      int64  
#  8   countNamecard        1 non-null      int64  
#  9   countcardhour        1 non-null      int64  
#  10  dis                  1 non-null      float64

