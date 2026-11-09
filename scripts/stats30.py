import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
S3=np.load('stack30.npz'); m=np.load('mask30.npy')
print('cells',m.sum(),'km2',round(m.sum()*0.0009,1))
for k in ['elevation_m','slope_deg','plan_curvature','profile_curvature','ls_factor','rainfall_mm','dist_streams_m','dist_roads_m','dist_faults_m','erodibility_k']:
    a=S3[k][m]; print(k,round(float(np.nanmin(a)),3),round(float(np.nanmax(a)),3),'mean',round(float(np.nanmean(a)),1))
lc=S3['lulc_code'][m]; v,c=np.unique(lc[~np.isnan(lc)],return_counts=True); print('lulc 30m',dict(zip(v.astype(int),(c/c.sum()*100).round(1))))
for k in ['lulc_tree','lulc_grass','lulc_bare','lulc_other']: print(k,'mean frac',round(float(np.nanmean(S3[k][m]))*100,1))
e=S3['eastness'][m]; print('area eastness>0',round((e>0).mean()*100,1))
g=S3['geology_unit'][m]; sl=S3['slope_deg'][m]; el=S3['elevation_m'][m]
for u in [1,2,3,4]: s=g==u; print('geol',u,'area%',round(s.mean()*100,1),'med slope',round(float(np.median(sl[s])),1),'med elev',int(np.median(el[s])))
D=pd.read_csv('bee_training_final.csv'); P=D[D.label==1]; A=D[D.label==0]
print('pres eastness>0',round((P.eastness>0).mean()*100,1),'abs',round((A.eastness>0).mean()*100,1))
d,_=cKDTree(P[['x','y']].values).query(P[['x','y']].values,k=2); print('presence NN median',int(np.median(d[:,1])),'min',int(d[:,1].min()))
print('corr', D[['slope_deg','ls_factor']].corr().iloc[0,1].round(2), D[['elevation_m','rainfall_mm']].corr().iloc[0,1].round(2))
NUM=['elevation_m','slope_deg','plan_curvature','profile_curvature','dist_streams_m','dist_roads_m','dist_faults_m','erodibility_k','northness','eastness']
C=D[NUM+['ls_factor','rainfall_mm']].corr(); print('pairs |r|>0.7:',[(a,b,round(C.loc[a,b],2)) for i,a in enumerate(C) for b in C.columns[i+1:] if abs(C.loc[a,b])>0.7])
print('slope P quantiles',P.slope_deg.quantile([.1,.5,.9]).round(1).tolist(),'A',A.slope_deg.quantile([.1,.5,.9]).round(1).tolist())
print(D[D.label==0].groupby('geology_unit').size())
