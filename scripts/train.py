import numpy as np, pandas as pd, json, joblib, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
D=pd.read_csv('bee_training_final.csv')
FEAT=['elevation_m','slope_deg','plan_curvature','profile_curvature','dist_streams_m','dist_roads_m','dist_faults_m','erodibility_k','northness','eastness',
      'geol_1','geol_2','geol_3','geol_4','lulc_tree','lulc_grass','lulc_bare','lulc_other']
tr=D[D.split=='train']; X,y=tr[FEAT].values,tr.label.values
cv=StratifiedKFold(5,shuffle=True,random_state=42)
models={
 'RF':(RandomForestClassifier(random_state=42,n_jobs=-1),{'n_estimators':[200,500],'max_depth':[None,6,10],'min_samples_leaf':[1,3,5],'max_features':['sqrt',0.5]}),
 'XGBoost':(XGBClassifier(random_state=42,eval_metric='logloss',n_jobs=4),{'n_estimators':[100,300],'max_depth':[2,3,5],'learning_rate':[0.03,0.1],'subsample':[0.8,1.0],'colsample_bytree':[0.7,1.0]}),
 'SVM':(make_pipeline(StandardScaler(),SVC(probability=True,random_state=42)),{'svc__C':[0.1,1,10,100],'svc__gamma':['scale',0.01,0.05,0.1]}),
 'MLP':(make_pipeline(StandardScaler(),MLPClassifier(max_iter=3000,early_stopping=False,random_state=42)),{'mlpclassifier__hidden_layer_sizes':[(16,),(32,),(32,16),(64,32)],'mlpclassifier__alpha':[1e-4,1e-3,1e-2,1e-1],'mlpclassifier__learning_rate_init':[1e-3,1e-2]}),
}
best={}
for name,(est,grid) in models.items():
    gs=GridSearchCV(est,grid,scoring='roc_auc',cv=cv,n_jobs=-1,refit=True).fit(X,y)
    best[name]={'params':{k.split('__')[-1]:v for k,v in gs.best_params_.items()},'cv_auc':round(gs.best_score_,4),'cv_auc_sd':round(gs.cv_results_['std_test_score'][gs.best_index_],4)}
    joblib.dump(gs.best_estimator_,f'{name}.joblib'); print(name,best[name])
json.dump({'features':FEAT,'models':best},open('best_params.json','w'),indent=1,default=str)
