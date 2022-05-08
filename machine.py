import pandas
import sklearn

from sklearn.model_selection import train_test_split
def machine():
       table = pandas.read_csv("lungcanser.csv")

       features=table.columns
       print(features)
       table = table.rename(columns={'FATIGUE ': 'FATIGUE'})
       table = table.rename(columns={'ALLERGY ': 'ALLERGY'})


       target=features[-1]
       features=features[:-1]
       table=pandas.get_dummies(data=table,columns=['LUNG_CANCER'],drop_first=True)#1=TRUE CANCER
       table=pandas.get_dummies(data=table,columns=['GENDER'],drop_first=True)#1=MALE

       scaler = sklearn.preprocessing.StandardScaler()
       scaler.fit(table.drop('LUNG_CANCER_YES',axis = 1))
       scaled_features = scaler.transform(table.drop('LUNG_CANCER_YES',axis = 1))
       table_feat = pandas.DataFrame(scaled_features,columns = ['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
              'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
              'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
              'SWALLOWING DIFFICULTY', 'CHEST PAIN',  'GENDER'])

       X = table_feat
       y = table['LUNG_CANCER_YES']
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=8)


       from xgboost import XGBClassifier
       xgb = XGBClassifier(booster = 'gblinear', learning_rate = 1, n_estimators = 100)
       xgb.fit(X_train, y_train)
       print("X_TEST",X_test)
       y_pred = xgb.predict(X_test)
       print(y_pred)
       xgb_train_acc = sklearn.metrics.accuracy_score(y_train, xgb.predict(X_train))
       xgb_test_acc = sklearn.metrics.accuracy_score(y_test, y_pred)


       print(f"Training Accuracy of XGB Model is {xgb_train_acc}")
       print(f"Test Accuracy of XGB Model is {xgb_test_acc}")
       output=f"Training Accuracy of XGB Model is {xgb_train_acc}"

       return output





