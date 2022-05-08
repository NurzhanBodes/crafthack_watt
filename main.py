from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.button import ButtonBehavior
from kivy.uix.image import Image
import pandas
import sklearn

from sklearn.model_selection import train_test_split

class HomeScreen(Screen):
    pass
class StartScreen(Screen):
    pass
class SmokeScreen(Screen):
    pass
class AnxiousScreen(Screen):
    pass
class FingersScreen(Screen):
    pass
class TiredScreen(Screen):
    pass
class WheezeScreen(Screen):
    pass
class SwallowScreen(Screen):
    pass
class AncestorScreen(Screen):
    pass
class PositiveScreen(Screen):
    pass
class NegativeScreen(Screen):
    pass

class ImageButton(ButtonBehavior, Image):
    pass


GUI=Builder.load_file("main.kv")

map={
            'gender':0,
            'age':0,
             'smoke':1,
             'alcohol':1,
             'anxious':1,
             'pressure':1,
             'yellow':1,
             'chronic':1,
             'tired':1,
             'allergy':1,
             'wheeze':1,
             'shortness':1,
             'swallowing':1,
             'pain':1,
             'family':1}

class MainApp(App):
    def build(self):
        return GUI
    def change_screen(self,screen_name):
        screen_manager=self.root.ids['screen_manager']
        screen_manager.current=screen_name
    def input(self,key,input):
        map[key]=input

    def machine(self):
        table = pandas.read_csv("lungcanser.csv")

        features = table.columns
        print(features)
        table = table.rename(columns={'FATIGUE ': 'FATIGUE'})
        table = table.rename(columns={'ALLERGY ': 'ALLERGY'})

        table = pandas.get_dummies(data=table, columns=['LUNG_CANCER'], drop_first=True)  # 1=TRUE CANCER
        table = pandas.get_dummies(data=table, columns=['GENDER'], drop_first=True)  # 1=MALE

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(table.drop('LUNG_CANCER_YES', axis=1))
        scaled_features = scaler.transform(table.drop('LUNG_CANCER_YES', axis=1))
        table_feat = pandas.DataFrame(scaled_features,
                                      columns=['AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                                               'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
                                               'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
                                               'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'GENDER'])
        keys_list=list(map.keys())
        state=0
        for keys in keys_list:
            if(map[keys]==1):
                state=1

        if(state==1):self.change_screen("negative_screen")
        else:self.change_screen("positive_screen")

        X = table_feat
        y = table['LUNG_CANCER_YES']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

        from xgboost import XGBClassifier
        xgb = XGBClassifier(booster='gblinear', learning_rate=1, n_estimators=100)
        xgb.fit(X_train, y_train)
        print("X_TEST", X_test)
        y_pred = xgb.predict(X_test)
        print(y_pred)
        xgb_train_acc = sklearn.metrics.accuracy_score(y_train, xgb.predict(X_train))
        xgb_test_acc = sklearn.metrics.accuracy_score(y_test, y_pred)

        print(f"Training Accuracy of XGB Model is {xgb_train_acc}")
        print(f"Test Accuracy of XGB Model is {xgb_test_acc}")


MainApp().run()
print(map)
