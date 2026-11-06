import numpy as np, pandas as pd, json, joblib, warnings
warnings.filterwarnings('ignore')
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
from sklearn.inspection import permutation_importance
D=pd.read_csv('bee_training_final.csv'); B=json.load(open('best_params.json')); FEAT=B['features']
tr,te=D[D.split=='train'],D[D.split=='test']
names=['RF','XGBoost','SVM','MLP']; M={n:joblib.load(f'{n}.joblib') for n in names}
def metrics(y,p):
    yh=(p>=0.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,yh).ravel()
    return dict(AUC=roc_auc_score(y,p),Accuracy=accuracy_score(y,yh),Precision=precision_score(y,yh,zero_division=0),Recall=recall_score(y,yh),
                Specificity=tn/(tn+fp),F1=f1_score(y,yh),Kappa=cohen_kappa_score(y,yh),TP=tp,FP=fp,TN=tn,FN=fn)
rows=[]; roc={}
for n in names:
    p=M[n].predict_proba(te[FEAT].values)[:,1]; rows.append({'model':n,**metrics(te.label.values,p)}); roc[n]=p
test=pd.DataFrame(rows).round(4); print('== TEST SET (n=90)'); print(test.to_string(index=False))
pd.DataFrame({'label':te.label.values,**roc}).to_csv('test_predictions.csv',index=False)
# spatial block CV with tuned hyper-parameters, all 300 points
sp=[]
for n in names:
    for f in sorted(D.spatial_fold.unique()):
        a,b=D[D.spatial_fold!=f],D[D.spatial_fold==f]
        m=clone(M[n]).fit(a[FEAT].values,a.label.values); p=m.predict_proba(b[FEAT].values)[:,1]
        sp.append({'model':n,'fold':f,**metrics(b.label.values,p)})
sp=pd.DataFrame(sp); spm=sp.groupby('model')[['AUC','Accuracy','F1','Kappa']].agg(['mean','std']).round(3).reindex(names)
print('== SPATIAL 5-FOLD CV'); print(spm.to_string())
sp.round(4).to_csv('spatial_cv_by_fold.csv',index=False)
# permutation importance on test set (AUC drop)
imp={}
for n in names:
    r=permutation_importance(M[n],te[FEAT].values,te.label.values,scoring='roc_auc',n_repeats=30,random_state=42)
    imp[n]=r.importances_mean
imp=pd.DataFrame(imp,index=FEAT).round(4); print('== PERMUTATION IMPORTANCE (AUC drop)'); print(imp.sort_values('RF',ascending=False).to_string())
imp.to_csv('permutation_importance.csv'); test.to_csv('test_metrics.csv',index=False); spm.to_csv('spatial_cv_summary.csv')
