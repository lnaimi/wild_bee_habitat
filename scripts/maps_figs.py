import numpy as np, pandas as pd, joblib, json, rasterio, warnings
warnings.filterwarnings('ignore')
from rasterio.transform import from_origin
from rasterio.crs import CRS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import roc_curve, roc_auc_score
NR,NC,CS,XLL,YLL=2460,4150,30.0,266900.0,3535978.0
G=pd.read_pickle('grid_features.pkl'); S=pd.read_csv('bee_training_final.csv'); FEAT=json.load(open('best_params.json'))['features']
names=['RF','XGBoost','SVM','MLP']; LBL={'RF':'Random forest','XGBoost':'XGBoost','SVM':'SVM','MLP':'Neural network (MLP)'}
COL={'RF':'#2a78d6','XGBoost':'#eb6834','SVM':'#1baf7a','MLP':'#eda100'}
M={n:joblib.load(f'{n}.joblib') for n in names}
import time
XA=G[FEAT].values.astype(np.float64); CH=200000
for n in names:
    t=time.time(); G['p_'+n]=np.concatenate([M[n].predict_proba(XA[i:i+CH])[:,1] for i in range(0,len(XA),CH)]).astype(np.float32); print(n,'mapped',round(time.time()-t),'s',flush=True)
del XA
G['p_ensemble']=G[['p_'+n for n in names]].mean(axis=1).astype(np.float32)
G['p_trees']=G[['p_RF','p_XGBoost']].mean(axis=1).astype(np.float32)
# rasters
tf=from_origin(XLL,YLL+NR*CS,CS,CS); out={}
for k in names+['ensemble','trees']:
    a=np.full(NR*NC,-9999,np.float32); a[G.id.values-1]=G['p_'+k].values; a=a.reshape(NR,NC); out[k]=a
    with rasterio.open(f'suitability_{k}.tif','w',driver='GTiff',height=NR,width=NC,count=1,dtype='float32',crs=CRS.from_epsg(32630),transform=tf,nodata=-9999,compress='deflate') as r: r.write(a,1)

# classes
bins=[0,.2,.4,.6,.8,1.0001]; cls=['Very low','Low','Moderate','High','Very high']
rows=[]
pres_cells=S[S.label==1].cell.values
for k in names+['ensemble','trees']:
    c=pd.cut(G['p_'+k],bins,labels=cls,right=False); cnt=c.value_counts().reindex(cls)
    pc=pd.cut(G.set_index('id').loc[pres_cells,'p_'+k],bins,labels=cls,right=False).value_counts().reindex(cls)
    for cl in cls: rows.append({'model':k,'class':cl,'cells':int(cnt[cl]),'area_km2':round(cnt[cl]*0.0009,1),'area_pct':round(cnt[cl]/len(G)*100,1),'presences':int(pc[cl]),'presence_pct':round(pc[cl]/len(pres_cells)*100,1)})
CT=pd.DataFrame(rows); CT.to_csv('suitability_classes.csv',index=False); print(CT[CT.model.isin(['ensemble','trees'])].to_string(index=False))
# ---- figure style
INK,INK2,GRID='#0b0b0b','#52514e','#e4e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False})
# ROC
te=S[S.split=='test']
fig,ax=plt.subplots(figsize=(6,5.4),dpi=200)
ax.plot([0,1],[0,1],color='#b5b4ad',lw=1,ls=(0,(4,3)),zorder=1)
ends=[]
for n in names:
    p=M[n].predict_proba(te[FEAT].values)[:,1]; f,t,_=roc_curve(te.label,p); a=roc_auc_score(te.label,p)
    ax.plot(f,t,color=COL[n],lw=2,label=f'{LBL[n]} (AUC = {a:.3f})',zorder=3)
ax.set_xlabel('False positive rate (1 − specificity)'); ax.set_ylabel('True positive rate (sensitivity)')
ax.set_xlim(0,1); ax.set_ylim(0,1.01); ax.grid(color=GRID,lw=.6); ax.set_axisbelow(True)
ax.legend(loc='lower right',frameon=False,fontsize=9,labelcolor=INK); ax.set_title('ROC curves on the test set (n = 90)',loc='left',fontsize=11,color=INK)
fig.tight_layout(); fig.savefig('fig_roc_test.png'); plt.close()
# importance dot plot
imp=pd.read_csv('permutation_importance.csv',index_col=0)
lab={'slope_deg':'Slope','dist_roads_m':'Distance to roads','dist_faults_m':'Distance to faults','elevation_m':'Elevation','geol_2':'Geology unit 2','eastness':'Eastness (aspect)','northness':'Northness (aspect)','erodibility_k':'Soil erodibility (K)','plan_curvature':'Plan curvature','profile_curvature':'Profile curvature','dist_streams_m':'Distance to streams','lulc_grass':'Land cover: grassland','lulc_bare':'Land cover: bare/sparse','lulc_tree':'Land cover: tree cover','lulc_other':'Land cover: other','geol_1':'Geology unit 1','geol_3':'Geology unit 3','geol_4':'Geology unit 4'}
top=imp.mean(axis=1).sort_values(ascending=False).head(10).index[::-1]
fig,ax=plt.subplots(figsize=(7,5),dpi=200)
off=dict(zip(names,[-.24,-.08,.08,.24]))
for n in names: ax.scatter(imp.loc[top,n].clip(lower=0),np.arange(len(top))+off[n],s=34,color=COL[n],edgecolor='white',linewidth=1,label=LBL[n],zorder=3)
ax.set_yticks(range(len(top))); ax.set_yticklabels([lab[t] for t in top]); ax.grid(axis='x',color=GRID,lw=.6); ax.set_axisbelow(True)
ax.set_xlabel('Permutation importance (drop in test AUC)'); ax.set_title('Variable importance, top 10 (mean across models)',loc='left',fontsize=11,color=INK)
ax.legend(frameon=False,fontsize=8.5,loc='lower right',labelcolor=INK); fig.tight_layout(); fig.savefig('fig_importance.png'); plt.close()
# maps
RAMP=['#86b6ef','#5598e7','#2a78d6','#1c5cab','#104281']
cmap=ListedColormap(RAMP); cmap.set_bad('#ffffff'); norm=BoundaryNorm([0,.2,.4,.6,.8,1.0001],5)
ext=[XLL/1000,(XLL+NC*CS)/1000,YLL/1000,(YLL+NR*CS)/1000]
P1=S[S.label==1]
def draw(ax,k,title,pts=True):
    a=np.ma.masked_equal(out[k],-9999); im=ax.imshow(a,cmap=cmap,norm=norm,extent=ext,interpolation='nearest')
    if pts: ax.scatter(P1.x/1000,P1.y/1000,s=7,color='#eb6834',edgecolor='white',linewidth=.5,zorder=3)
    ax.set_title(title,loc='left',fontsize=10,color=INK); ax.set_aspect('equal'); ax.tick_params(labelsize=7)
    for s in ax.spines.values(): s.set_visible(False)
    return im
fig,axs=plt.subplots(2,2,figsize=(10,6.6),dpi=200)
for ax,n in zip(axs.ravel(),names): im=draw(ax,n,LBL[n])
for ax in axs[:,0]: ax.set_ylabel('UTM N (km)',fontsize=8)
for ax in axs[1,:]: ax.set_xlabel('UTM E (km)',fontsize=8)
cb=fig.colorbar(im,ax=axs,ticks=[.1,.3,.5,.7,.9],fraction=.025,pad=.02); cb.ax.set_yticklabels(cls,fontsize=8); cb.outline.set_visible(False)
fig.suptitle('Wild bee habitat suitability by model (orange dots: presences)',x=.02,ha='left',fontsize=11,color=INK)
fig.savefig('fig_maps_4models.png',bbox_inches='tight'); plt.close()
fig,ax=plt.subplots(figsize=(9,5.6),dpi=200); im=draw(ax,'ensemble','Wild bee habitat suitability, ensemble mean of 4 models')
ax.set_xlabel('UTM E (km), zone 30N',fontsize=8); ax.set_ylabel('UTM N (km)',fontsize=8)
cb=fig.colorbar(im,ax=ax,ticks=[.1,.3,.5,.7,.9],fraction=.03,pad=.02); cb.ax.set_yticklabels([f'{c} ({lo:.1f}–{lo+.2:.1f})' for c,lo in zip(cls,[0,.2,.4,.6,.8])],fontsize=8); cb.outline.set_visible(False)
ax.scatter([],[],s=14,color='#eb6834',edgecolor='white',label='Bee presence (n = 150)'); ax.legend(loc='lower left',frameon=False,fontsize=8)
fig.savefig('fig_map_ensemble.png',bbox_inches='tight'); plt.close()
print('corr between model maps'); print(G[['p_'+n for n in names]].corr().round(2))
