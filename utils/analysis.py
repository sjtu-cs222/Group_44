import sys
sys.path.append(".//utils//")
import pandas as pd
import parsing_utils
from parsing_utils import parse_databases
import importlib
importlib.reload(parsing_utils)
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def drop_empty_cols(matches, minscale = .2):
    matches["time"] = MinMaxScaler((minscale,1)).fit_transform(matches.loc[:,["date"]])
    return matches
    
    
def make_feats(player_feats, roles=["Top","Jungle","Middle","ADC","Support"]):
    blue_feats = {}
    red_feats = {}
    feats = {}
    all_feats = []
    if roles != None:
        for ii in roles:
            blue_feats[ii] = ["blue"+"_"+ii+"_"+x for x in player_feats]
            red_feats[ii] = ["red"+"_"+ii+"_"+x for x in player_feats]
            all_feats += blue_feats[ii]
            all_feats += red_feats[ii]
        feats = {"Red":red_feats,"Blue":blue_feats}
    else:
        blue_feats = ["blue"+"_"+x for x in player_feats]
        red_feats = ["red"+"_"+x for x in player_feats]
        all_feats += blue_feats
        all_feats += red_feats
        feats = {"Red":red_feats,"Blue":blue_feats}
    return all_feats, feats
    print(len(feats))


def split(match_data, split = True):
    data = match_data.dropna(how = "any", axis=0).drop("league",axis=1, errors = "ignore")
    data_init = pd.get_dummies(data)
    cols = data_init.drop("blue_win",axis=1).columns.values
    y = data_init["blue_win"].values
    scaler = StandardScaler().fit(data_init.drop("blue_win",axis=1))
    data = scaler.transform(data_init.drop("blue_win",axis=1))
    if split:
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=21, shuffle=True, stratify = y)
        return y, cols, X_train, X_test, y_train, y_test, scaler
    else:
        return y, cols, data, data_init, scaler
		
def run_classifier(X_train, X_test, y_train, y_test, params = {"max_depth":1, "n_estimators":100, "learning_rate":.1, "colsample_bytree": .5, "subsample": .5, "gamma":0}, cv = True):
    cvresult = None
    clf = xgb.XGBClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], colsample_bytree = params["colsample_bytree"], subsample=params["subsample"], gamma = params["gamma"])
    if cv:
        cvresult = cross_val_score(clf,X_train,y_train,cv = 5)
    clf.fit(X_train, y_train)
	
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print("Training accuracy: {:10.3f}".format(train_acc))
    
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print("Testing accuracy: {:10.3f}".format(test_acc))

    return clf, cvresult
    

def main():
    player_feats = ['gamelength', 'result', 'k', 'd', 'a', 'teamkills', 'teamdeaths', 'doubles', 'triples', 
                'quadras', 'pentas', 'fb', 'fbassist', 'fbvictim', 'fbtime', 'kpm', 'okpm', 'ckpm', 'fd', 
                'fdtime', 'teamdragkills', 'oppdragkills', 'herald', 'ft', 'fttime', 'firstmidouter', 
                'firsttothreetowers', 'teamtowerkills', 'opptowerkills', 'fbaron', 'fbarontime', 'teambaronkills', 
                'oppbaronkills', 'dmgtochamps', 'dmgtochampsperminute', 'dmgshare', 'earnedgoldshare', 'wards', 
                'wpm', 'wardshare', 'wardkills', 'wcpm', 'visionwards', 'visionwardbuys', 'visiblewardclearrate', 
                'invisiblewardclearrate', 'totalgold', 'earnedgpm', 'goldspent', 'gspd', 'minionkills', 
                'monsterkills', 'monsterkillsownjungle', 'monsterkillsenemyjungle', 'cspm', 'goldat10', 
                'oppgoldat10', 'gdat10', 'goldat15', 'oppgoldat15', 'gdat15', 'xpat10', 'oppxpat10', 'xpdat10']	
                
    parser = parse_databases()
    db_dir =  r"C:\Users\pc\Desktop\LoL-match-analytics-master\databases"
    match,random = parser.get_dbs(db_dir)
    #query = """SELECT * FROM '2016matchdata';"""
    #matches_16 = drop_empty_cols(pd.read_sql_query(query,match))
    #query = """SELECT * FROM '2017matchdata';"""
    #matches_17 = drop_empty_cols(pd.read_sql_query(query,match))
    query = """SELECT * FROM '2018matchdata';"""
    matches_18 = drop_empty_cols(pd.read_sql_query(query,match), .4) 
              
    all_feats, feats = make_feats(player_feats)   
    to_use = matches_18.copy()
    match_data = pd.DataFrame(columns = all_feats+["league"], index = to_use["gameid"].unique()).fillna(0.0)
    match_data["blue_win"] = 0
    X_train, X_test, y_train, y_test = split(match_data)
    clf, cvresult = run_classifier(X_train, X_test, y_train, y_test)
    
if __name__ == '__main__':
    main()