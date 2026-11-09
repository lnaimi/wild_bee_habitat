import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option('display.width',200)
XLL,YLL,CS,NR,NC=266900.0,3535978.0,30.0,2460,4150; TOP=YLL+NR*CS
S3=np.load('stack30.npz'); L={k:S3[k] for k in S3.files}; mask=np.load('mask30.npy')
P=pd.read_csv('../points30_exact.csv')
old=pd.read_csv('../v2/bee_training_final.csv')[['cell','split','spatial_fold']]
P=P.merge(old,on='cell',how='left'); assert P.split.notna().all()
P['row']=((TOP-P.y)//CS).astype(int); P['col']=((P.x-XLL)//CS).astype(int); P['pix']=P.row*NC+P.col+1
assert P.pix.is_unique
ok=mask[P.row,P.col]; print('points on valid 30 m pixels',ok.sum(),'/',len(P))
for k,v in L.items(): P[k]=v[P.row,P.col]
P['source']=np.where(P.label==1,'verified presence (field survey + shepherd interviews)','verified absence (field survey + shepherd interviews)')
NUM=['elevation_m','slope_deg','plan_curvature','profile_curvature','dist_streams_m','dist_roads_m','dist_faults_m','erodibility_k','northness','eastness']
X=(P[NUM]-P[NUM].mean())/P[NUM].std(); X.insert(0,'c',1)
print('VIF',{k:round(variance_inflation_factor(X.values,i+1),2) for i,k in enumerate(NUM)})
print('r elev-rain',round(P.elevation_m.corr(P.rainfall_mm),2),'r slope-LS',round(P.slope_deg.corr(P.ls_factor),2))
print(P.groupby('label')[NUM+['rainfall_mm','ls_factor','lulc_tree','lulc_grass','lulc_bare']].median().T.round(3))
print('single-variable AUC:',{k:round(max(roc_auc_score(P.label,P[k]),1-roc_auc_score(P.label,P[k])),3) for k in NUM})
pr=P[P.label==1]; ab=P[P.label==0]
d,_=cKDTree(pr[['x','y']].values).query(ab[['x','y']].values); print('absence->presence min',int(d.min()),'median',int(np.median(d)),'n<1km',int((d<1000).sum()))
print(pd.crosstab(P.geology_unit,P.label)); print(pd.crosstab(P.spatial_fold,P.label)); print(P.groupby(['split','label']).size().to_dict())
inv=Transformer.from_crs(32630,4326,always_xy=True)
FEAT=NUM+['geol_1','geol_2','geol_3','geol_4','lulc_tree','lulc_grass','lulc_bare','lulc_other']
P=P.rename(columns={'cell':'cell300','pix':'cell'})
P[['cell','cell300','id','label','source','x','y','lon','lat','split','spatial_fold','aspect_deg','ls_factor','rainfall_mm','geology_unit','lulc_code']+FEAT].to_csv('bee_training_final.csv',index=False)
# full grid table for mapping
r,c=np.nonzero(mask)
G=pd.DataFrame({'id':(r*NC+c+1).astype('int32'),'row':r.astype('int32'),'col':c.astype('int32'),'x':(XLL+(c+.5)*CS).astype('float32'),'y':(YLL+(NR-r-.5)*CS).astype('float32')})
for k in FEAT+['ls_factor','rainfall_mm','geology_unit','lulc_code','aspect_deg']: G[k]=L[k][r,c]
lo,la=inv.transform(G.x.values.astype(float),G.y.values.astype(float)); G['lon']=lo.astype('float32'); G['lat']=la.astype('float32')
G.to_pickle('grid_features.pkl'); print('grid rows',len(G))
