import numpy as np, pandas as pd, joblib, json, shap, warnings, time
warnings.filterwarnings('ignore')
S=pd.read_csv('bee_training_final.csv'); FEAT=json.load(open('best_params.json'))['features']
X=S[FEAT].astype(float); tr=S[S.split=='train'][FEAT].astype(float)
M={n:joblib.load(f'{n}.joblib') for n in ['RF','XGBoost','SVM','MLP']}
bg=shap.kmeans(tr.values,25)
out={}
t=time.time()
e=shap.TreeExplainer(M['RF']); v=e.shap_values(X.values)
v=v[...,1] if np.ndim(v)==3 else (v[1] if isinstance(v,list) else v); out['RF']=v; print('RF',time.time()-t)
e=shap.TreeExplainer(M['XGBoost'],data=tr.sample(100,random_state=0).values,model_output='probability',feature_perturbation='interventional'); out['XGBoost']=e.shap_values(X.values); print('XGB',time.time()-t)
for n in ['SVM','MLP']:
    f=lambda A,m=M[n]: m.predict_proba(A)[:,1]
    e=shap.KernelExplainer(f,bg); out[n]=e.shap_values(X.values,nsamples=400,silent=True); print(n,time.time()-t)
np.savez('shap_values.npz',**out,X=X.values,feat=np.array(FEAT))
mabs=pd.DataFrame({n:np.abs(v).mean(0) for n,v in out.items()},index=FEAT); mabs['mean']=mabs.mean(1)
mabs.sort_values('mean',ascending=False).round(4).to_csv('shap_mean_abs.csv'); print(mabs.sort_values('mean',ascending=False).round(4).head(10))
# additivity check: base + sum = prediction
p=M['RF'].predict_proba(X.values)[:,1]; print('RF additivity max err', np.abs(out['RF'].sum(1)+np.mean(p)-p).max())
