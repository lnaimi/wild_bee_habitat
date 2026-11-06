import json, glob, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option('display.width',200)
rows=[json.load(open(f)) for f in glob.glob('../db_v4/points/*.json')]
D=pd.DataFrame([r.get('data',r) for r in rows]); print('points',len(D),D.label.value_counts().to_dict(),D.groupby('label').source.value_counts().to_dict())
G=pd.read_pickle('../final/grid_features.pkl')
old=pd.read_csv('../final/bee_training_final.csv')
for l,nm in [(1,'presences'),(0,'absences')]:
    a=set(D[D.label==l].cell); b=set(old[old.label==l].cell); print(nm,': kept',len(a&b),'new',len(a-b),'removed',len(b-a))
assert D.cell.is_unique
S=D[['cell','label']].merge(G.rename(columns={'id':'cell'}),on='cell',how='left'); print('missing',int(S.isna().any(axis=1).sum()))
S['source']=np.where(S.label==1,'field presence','background (adjusted)')
NUM=['elevation_m','slope_deg','plan_curvature','profile_curvature','dist_streams_m','dist_roads_m','dist_faults_m','erodibility_k','northness','eastness']
X=(S[NUM]-S[NUM].mean())/S[NUM].std(); X.insert(0,'c',1)
print('VIF',{k:round(variance_inflation_factor(X.values,i+1),2) for i,k in enumerate(NUM)})
print('r elev-rain',round(S.elevation_m.corr(S.rainfall_mm),2),'slope-ls',round(S.slope_deg.corr(S.ls_factor),2))
print(S.groupby('label')[NUM+['rainfall_mm']].median().T.round(3))
print('single-variable AUC (circularity check):',{k:round(max(roc_auc_score(S.label,S[k]),1-roc_auc_score(S.label,S[k])),3) for k in NUM})
P=S[S.label==1]; B=S[S.label==0]
t=cKDTree(P[['x','y']].values); d,_=t.query(B[['x','y']].values); print('absence->nearest presence: min',int(d.min()),'median',int(np.median(d)),'n<1km',int((d<1000).sum()))
dp,_=cKDTree(P[['x','y']].values).query(P[['x','y']].values,k=2); print('presence NN median',int(np.median(dp[:,1])))
print(pd.crosstab(S.geology_unit,S.label)); print(pd.crosstab(S.lulc_code,S.label))
tr,te=train_test_split(S.index,test_size=0.3,stratify=S.label,random_state=42); S['split']='train'; S.loc[te,'split']='test'
blk=(((S.x-266900)//4000).astype(int)*1000+((S.y-3535978)//4000).astype(int))
g=S.groupby(blk).label.agg(['sum','count']); Pp=g['sum'].values; Nn=(g['count']-g['sum']).values
rng=np.random.default_rng(0); best=None; nP,nB=len(P)/5,len(B)/5
for it in range(20000):
    a=rng.integers(0,5,len(g)); fp=np.bincount(a,Pp,5); fn=np.bincount(a,Nn,5); s=max(np.abs(fp-nP).max(),np.abs(fn-nB).max())
    if best is None or s<best[0]: best=(s,a)
S['spatial_fold']=blk.map(dict(zip(g.index,best[1]+1))).values
print(pd.crosstab(S.spatial_fold,S.label)); print(S.groupby(['split','label']).size().to_dict())
FEAT=NUM+['geol_1','geol_2','geol_3','geol_4','lulc_tree','lulc_grass','lulc_bare','lulc_other']
S[['cell','label','source','x','y','lon','lat','split','spatial_fold','aspect_deg','ls_factor','rainfall_mm','geology_unit','lulc_code']+FEAT].to_csv('bee_training_final.csv',index=False)
