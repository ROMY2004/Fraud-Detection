from tkinter import *
from tkinter import ttk
import pandas
import tkcalendar as tc
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from   sklearn.metrics import f1_score, precision_score, recall_score

data = pd.read_csv(r"D:\FraudDetection\fraudTrain.csv")
testP = pd.read_csv("processedTest.csv")

data['Time'] = pd.to_datetime(data['Time'])


data['year'] = data['Time'].dt.year
data['day'] = data['Time'].dt.day
data['month'] = data['Time'].dt.month
data['hour'] = data['Time'].dt.hour

data['fullName'] = data['firstName'] + data['lastName']
data['fullNamehour'] = data['fullName'] + data['year'].astype(str) + data['month'].astype(str) + data['day'].astype(str) + data['hour'].astype(str)
data['fullNamesinglehour'] = data['fullName'] + data['hour'].astype(str)
data['fullNamecard'] = data['fullName'] + data['Card Number'].astype(str)

root = Tk()
root.geometry("800x550")
root.geometry("+450+100")

root.title("Fraud detection")

title = Label(root,text="\t\t Fraud detection",font="Calibre 20 bold")
title.grid(row = 0,column=0,pady=10,columnspan=4,ipady=10)

fnameLable = Label(root,text="first name")
fnameLable.grid(row=1,column =0,padx=10,pady=10)
first_name = ttk.Entry(root,width=20)
first_name.grid(row=1,column =1)


lname = Label(root,text="last name")
lname.grid(row=1,column =3,padx=10,pady=20)
last_name = ttk.Entry(root,width=20)
last_name.grid(row=1,column =4)


Category = Label(root,text="Category")
Category.grid(row=2,column =0,padx=10,pady=10)
categoryCombo = ttk.Combobox(root,values = ['misc_net' , 'grocery_pos' , 'entertainment', 'gas_transport' ,'misc_pos'
 ,'grocery_net' ,'shopping_net' ,'shopping_pos' ,'food_dining', 'personal_care'
 ,'health_fitness', 'travel', 'kids_pets', 'home'],width=17)
categoryCombo.grid(row=2,column =1)


mer = Label(root,text="Merchant")
mer.grid(row=2,column =3,padx=10,pady=10)
Merchant = ttk.Entry(root,width=20)
Merchant.grid(row= 2,column=4)


card = Label(root,text="Card Number")
card.grid(row=3,column =0,padx=10,pady=10)
Card_number = Spinbox(root,width=19,from_=0,to=99999999999)
Card_number.grid(row=3,column=1)


AmountLable = Label(root,text="Amount")
AmountLable.grid(row=3,column =3,padx=10,pady=10)
Amount = Spinbox(root,width=19,from_=0,to=999999999)
Amount.grid(row=3,column=4)


d = Label(root,text="Date")
d.grid(row=4,column=0,padx=10,pady=10)
date = tc.DateEntry(root,width=18)
date.grid(row=4,column=1)
date.set_date(date=datetime.datetime.strptime("1/1/20","%m/%d/%y"))


timeLable = Label(root,text="Time (HH : MM)")
timeLable.grid(row=4,column =3,padx=10,pady=10)
time = ttk.Entry(root,width=20)
time.grid(row=4,column =4)
time.insert(0,"1:0")


algoLabel = Label(root,text="Algorithm",font="Calibre 10 bold")
algoLabel.grid(row=6,column=0,padx=20,pady=10,sticky=W)
algo = StringVar(value="Random Forest")

# SVM = Radiobutton(root,text="SVM",variable=algo,value = "SVM")
# SVM.grid(row=10,column =0,padx=40,pady=10,sticky=W)

randomForest = Radiobutton(root,text="Random Forest",variable=algo,value="Random Forest")
randomForest.grid(row=7,column =0,padx=40,pady=10,sticky=W)

LR = Radiobutton(root,text="Logistic Regression",variable=algo , value = "Logistic Regression")
LR.grid(row=9,column =0,padx=40,pady=10,sticky=W)

XGBoost = Radiobutton(root,text="XGBoost",variable=algo , value ="XGBoost")
XGBoost.grid(row=8,column =0,padx=40,pady=10,sticky=W)



# model = joblib.load('RandomForestTrainedModel.pkl')



def datapre():
    data['year'] = data['Time'].dt.year
    data['day'] = data['Time'].dt.day
    data['month'] = data['Time'].dt.month
    data['hour'] = data['Time'].dt.hour
    data['fullName'] = data['firstName'] + data['lastName']
    data['fullNamehour'] = data['fullName'] + data['year'].astype(str) + data['month'].astype(str) + data['day'].astype(str) + data['hour'].astype(str)
    data['fullNamesinglehour'] = data['fullName'] + data['hour'].astype(str)
    data['fullNamecard'] = data['fullName'] + data['Card Number'].astype(str)
    data['cardhour'] = data['Card Number'].astype(str) + data['year'].astype(str) + data['month'].astype(str) + data['day'].astype(str) + data['hour'].astype(str)
    data['fullNameday'] = data['fullName'] + data['year'].astype(str) + data['month'].astype(str) + data['day'].astype(str)


def buttonClicked():
    datapre()
    firstName = first_name.get()
    lastName = last_name.get()
    category = categoryCombo.get()
    merchant = Merchant.get()
    cardNumber = float(Card_number.get())
    amount = float(Amount.get())
    Date = date.get()
    Time = time.get()
    wholeTime = datetime.datetime.strptime(Date +' '+Time,"%m/%d/%y %I:%M")
    user_input = {'firstName' : firstName,'lastName':lastName ,'category': category, 'Card Number':cardNumber ,'Amount': amount,'Merchant':merchant, 'Time':wholeTime , 'ID':0 , 'trans_num':"abc"}
    user_df = pandas.DataFrame([user_input])
    
    user_df['year'] = user_df['Time'].dt.year
    user_df['day'] = user_df['Time'].dt.day
    user_df['month'] = user_df['Time'].dt.month
    user_df['hour'] = user_df['Time'].dt.hour
    user_df['fullName'] = user_df['firstName'] + user_df['lastName']
    user_df['fullNamehour'] = user_df['fullName'] + user_df['year'].astype(str) + user_df['month'].astype(str) + user_df['day'].astype(str) + user_df['hour'].astype(str)
    user_df['fullNamesinglehour'] = user_df['fullName'] + user_df['hour'].astype(str)
    user_df['fullNamecard'] = user_df['fullName'] + user_df['Card Number'].astype(str)
    user_df['cardhour'] = user_df['Card Number'].astype(str) + user_df['year'].astype(str) + user_df['month'].astype(str) + user_df['day'].astype(str) + user_df['hour'].astype(str)
    user_df['fullNameday'] = user_df['fullName'] + user_df['year'].astype(str) + user_df['month'].astype(str) + user_df['day'].astype(str)
    # user_df['time_diff'] = user_df.groupby('Card Number')['Time'].diff().fillna(pd.Timedelta(seconds=0))

    
        
    # countuser = user_df['fullName'].value_counts()
    # user_df['countName'] = user_df['fullName'].map(countuser)
    user_df['countName'] = (data['fullName'] == user_df['fullName'].values[0]).sum()


    # countuser = user_df['fullNamehour'].value_counts()
    # user_df['countNamehour'] = user_df['fullNamehour'].map(countuser)
    user_df['countNamehour'] = (data['fullNamehour'] == user_df['fullNamehour'].values[0]).sum()


    # countuser = user_df['fullNamesinglehour'].value_counts()
    # user_df['countNamesinglehour'] = user_df['fullNamesinglehour'].map(countuser)
    user_df['countNamesinglehour'] = (data['fullNamesinglehour'] == user_df['fullNamesinglehour'].values[0]).sum()


    # countuser = user_df['fullNamecard'].value_counts()
    # user_df['countNamecard'] = user_df['fullNamecard'].map(countuser)
    user_df['countNamecard'] = (data['fullNamecard'] == user_df['fullNamecard'].values[0]).sum()

    
    # countuser = user_df['cardhour'].value_counts()
    # user_df['countcardhour'] = user_df['cardhour'].map(countuser)
    # user_df['countcardhour'] = (data['cardhour'] == user_df['cardhour'].values[0]).sum()

    user_df['dis'] = user_df['Amount'] - user_df['Amount'].mean()

    label_encoder = LabelEncoder()

    user_df['fullName'] = label_encoder.fit_transform(user_df['fullName'])
    user_df['fullNamehour'] = label_encoder.fit_transform(user_df['fullNamehour'])
    user_df['fullNamesinglehour'] = label_encoder.fit_transform(user_df['fullNamesinglehour'])
    user_df['fullNamecard'] = label_encoder.fit_transform(user_df['fullNamecard'])
    # user_df['cardhour'] = label_encoder.fit_transform(user_df['cardhour'])
    user_df['fullNameday'] = label_encoder.fit_transform(user_df['fullNameday'])
    user_df['category'] = label_encoder.fit_transform(user_df['category'])
    # user_df['time_diff'] = label_encoder.fit_transform(user_df['time_diff'])

    if (algo.get() == "XGBoost"):
        model = joblib.load('XGBoostTrainedModel.pkl')
    elif (algo.get() == "Logistic Regression"):
        model = joblib.load('LogisticRegressionTrainedModel.pkl')
    else:
        model = joblib.load('RandomForestTrainedModel.pkl')



    user_df.drop(['Card Number','day','year','fullName','fullNamehour','fullNamesinglehour','fullNamecard','cardhour','fullNameday'],axis=1,inplace=True)
    user_df.drop(['Time','ID','Merchant','trans_num','firstName','lastName'],axis=1,inplace=True)
    train(user_df,model)

    #data.drop(['is_fraud'],axis=1,inplace=True)
    
def train(user_df,model):
    user_df.info()
    user_pred = model.predict(user_df)
    print("Predicted fraud status:", user_pred)
    acc(user_df,user_pred,model)

def acc(user_df,user_pred,model):
    X_test = testP.drop(['is_fraud'],axis=1)
    y_test = testP['is_fraud']
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)


    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    mess = Tk()
    mess.title("Prediction Info")
    mess.geometry("300x300")
    mess.geometry("+550+300")
    txt = "Accuracy : " + str(accuracy) + "\n\n"

    txt += "Precision : " + str(round(precision_score(y_test, y_pred, average='binary'),2)) + "\n\n"

    txt += "Recall: " + str(round(recall_score(y_test, y_pred, average='binary'),2)) + "\n\n"

    txt += "F1-score: " + str(round(f1_score(y_test, y_pred, average='binary'),2)) + "\n\n"

    txt += "=============\n\n"

    txt += "Prediction: is" + ("" if user_pred else "n't")+" Fraud" 
    Label(mess,text=txt).pack(pady=20)
    ttk.Button(mess,text="ok",command=mess.destroy).pack(pady=20)
    mess.mainloop()

predict = ttk.Button(root,text="Predict",command=buttonClicked)
predict.grid(row=11,column= 2,padx=35,pady=40,ipadx=20,ipady=10)


root.mainloop()
