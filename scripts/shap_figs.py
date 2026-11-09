import numpy as np, pandas as pd, joblib, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
Z=np.load('shap_values.npz',allow_pickle=True); FEAT=list(Z['feat']); X=pd.DataFrame(Z['X'],columns=FEAT)
S=pd.read_csv('bee_training_final.csv')
names=['RF','XGBoost','SVM','MLP']; LBL={'RF':'Random forest','XGBoost':'XGBoost','SVM':'SVM','MLP':'Neural network (MLP)'}
COL={'RF':'#2a78d6','XGBoost':'#eb6834','SVM':'#1baf7a','MLP':'#eda100'}
lab={'slope_deg':'Slope','dist_roads_m':'Distance to roads','dist_faults_m':'Distance to faults','elevation_m':'Elevation','geol_2':'Geology unit 2','eastness':'Eastness','northness':'Northness','erodibility_k':'Soil erodibility (K)','plan_curvature':'Plan curvature','profile_curvature':'Profile curvature','dist_streams_m':'Distance to streams','lulc_grass':'Grassland share','lulc_bare':'Bare/sparse share','lulc_tree':'Tree cover share','lulc_other':'Other land cover share','geol_1':'Geology unit 1','geol_3':'Geology unit 3','geol_4':'Geology unit 4'}
INK,INK2,GRID='#0b0b0b','#52514e','#e4e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False})
M=pd.read_csv('shap_mean_abs.csv',index_col=0)
# A: mean |SHAP| dot plot
top=M.head(10).index[::-1]
fig,ax=plt.subplots(figsize=(7,4.8),dpi=250); off=dict(zip(names,[-.24,-.08,.08,.24]))
for n in names: ax.scatter(M.loc[top,n],np.arange(len(top))+off[n],s=34,color=COL[n],edgecolor='white',linewidth=1,label=LBL[n],zorder=3)
ax.set_yticks(range(len(top))); ax.set_yticklabels([lab[t] for t in top]); ax.grid(axis='x',color=GRID,lw=.6); ax.set_axisbelow(True)
ax.set_xlabel('Mean |SHAP value| (change in predicted probability)'); ax.legend(frameon=False,fontsize=8,loc='lower right',labelcolor=INK)
ax.set_title('Global SHAP importance, top 10 features',loc='left',fontsize=10.5,color=INK); fig.tight_layout(); fig.savefig('fig_shap_global.png'); plt.close()
# B: beeswarm RF
V=Z['RF']; cmap=LinearSegmentedColormap.from_list('d',['#2a78d6','#d9d8d4','#e34948'])
order=M['RF'].sort_values(ascending=False).head(10).index[::-1]
fig,ax=plt.subplots(figsize=(7,5),dpi=250); rng=np.random.default_rng(0)
for k,f in enumerate(order):
    j=FEAT.index(f); sv=V[:,j]; xv=X[f].values; q=(xv-np.percentile(xv,2))/(np.percentile(xv,98)-np.percentile(xv,2)+1e-12)
    # simple beeswarm jitter by density
    h,edges=np.histogram(sv,bins=40); idx=np.clip(np.digitize(sv,edges)-1,0,39); dens=h[idx]/h.max()
    y=k+rng.uniform(-1,1,len(sv))*0.38*dens
    sc=ax.scatter(sv,y,c=np.clip(q,0,1),cmap=cmap,s=9,linewidth=0,alpha=.9,vmin=0,vmax=1)
ax.axvline(0,color=INK2,lw=.8); ax.set_yticks(range(len(order))); ax.set_yticklabels([lab[t] for t in order])
ax.set_xlabel('SHAP value (impact on predicted probability of presence)'); ax.grid(axis='x',color=GRID,lw=.6); ax.set_axisbelow(True)
cb=fig.colorbar(sc,ax=ax,fraction=.03,pad=.02,ticks=[0,1]); cb.ax.set_yticklabels(['Low','High']); cb.set_label('Feature value',fontsize=8); cb.outline.set_visible(False)
ax.set_title('SHAP summary (beeswarm), random forest, n = 300',loc='left',fontsize=10.5,color=INK); fig.tight_layout(); fig.savefig('fig_shap_beeswarm_rf.png'); plt.close()
# C: dependence plots RF
pan=[('slope_deg','Slope (°)',1,'elevation_m','Elevation (m)'),('dist_roads_m','Distance to roads (km)',1000,'slope_deg','Slope (°)'),('dist_faults_m','Distance to faults (km)',1000,'slope_deg','Slope (°)'),('elevation_m','Elevation (m)',1,'slope_deg','Slope (°)')]
seq=LinearSegmentedColormap.from_list('s',['#cde2fb','#5598e7','#1c5cab','#0d366b'])
fig,axs=plt.subplots(1,4,figsize=(13,3.4),dpi=250,sharey=True)
for ax,(f,xl,sc_,cf,cl) in zip(axs,pan):
    j=FEAT.index(f); s=ax.scatter(X[f]/sc_,V[:,j],c=X[cf],cmap=seq,s=10,linewidth=.3,edgecolor='white')
    ax.axhline(0,color=INK2,lw=.7); ax.set_xlabel(xl); ax.grid(color=GRID,lw=.5); ax.set_axisbelow(True)
    cb=fig.colorbar(s,ax=ax,fraction=.05,pad=.02); cb.set_label(cl,fontsize=7); cb.ax.tick_params(labelsize=6.5); cb.outline.set_visible(False)
axs[0].set_ylabel('SHAP value')
fig.suptitle('SHAP dependence plots, random forest (colour: interacting feature)',x=.01,ha='left',fontsize=10.5,color=INK); fig.tight_layout(); fig.savefig('fig_shap_dependence_rf.png'); plt.close()
# D: local explanations (RF): highest-prob test presence and lowest-prob test background
import joblib; rf=joblib.load('RF.joblib'); p=rf.predict_proba(X.values)[:,1]; base=float(np.mean(p))
te=S.split.values=='test'; ip=np.where(te&(S.label.values==1))[0]; ib=np.where(te&(S.label.values==0))[0]
cases=[(ip[np.argmax(p[ip])],'Test presence'),(ib[np.argmin(p[ib])],'Test absence')]
fig,axs=plt.subplots(1,2,figsize=(11,3.8),dpi=250)
for ax,(i,t) in zip(axs,cases):
    sv=V[i]; o=np.argsort(-np.abs(sv))[:7][::-1]
    cols=['#e34948' if sv[j]>0 else '#2a78d6' for j in o]
    def fmt(f,v):
        return f'{v:.0f}' if f.endswith('_m') else (f'{v:.1f}' if f=='slope_deg' else (f'{v:.2f}' if not f.startswith('geol') else ('yes' if v>0.5 else 'no')) if not f.startswith('lulc') else f'{v*100:.0f}%')
    ax.barh(range(len(o)),sv[o],color=cols,height=.6); ax.axvline(0,color=INK2,lw=.8)
    ax.set_yticks(range(len(o))); ax.set_yticklabels([f'{lab[FEAT[j]]} = {fmt(FEAT[j],X.iloc[i,j])}' for j in o],fontsize=8)
    ax.set_xlabel('SHAP value'); ax.grid(axis='x',color=GRID,lw=.5); ax.set_axisbelow(True)
    ax.set_title(f'{t}: p = {p[i]:.2f} (base = {base:.2f})',loc='left',fontsize=9.5,color=INK)
fig.tight_layout(); fig.savefig('fig_shap_local_rf.png'); plt.close()
print('cases',[(int(S.cell.iloc[i]),round(p[i],3)) for i,_ in cases],'base',round(base,3))
# numbers for text: slope SHAP sign change
j=FEAT.index('slope_deg'); sl=X.slope_deg.values; sv=V[:,j]
print('RF slope SHAP <0 median slope',np.median(sl[sv<0]).round(1),' >0',np.median(sl[sv>0]).round(1))
for f in ['slope_deg','dist_roads_m','dist_faults_m','elevation_m']:
    jj=FEAT.index(f); r=np.corrcoef(X[f],V[:,jj])[0,1]; print(f,'corr value-SHAP',round(r,2))
# crossing point estimate for slope
o=np.argsort(sl); sm=pd.Series(sv[o]).rolling(25,center=True).mean().values; xs=sl[o]; k=np.where(np.diff(np.sign(sm[~np.isnan(sm)]))!=0)[0]; print('slope crossing ~',xs[~np.isnan(sm)][k][:3].round(1))
o=np.argsort(X.dist_roads_m.values); jj=FEAT.index('dist_roads_m'); sm=pd.Series(V[o,jj]).rolling(25,center=True).mean().values; xs=X.dist_roads_m.values[o]; k=np.where(np.diff(np.sign(sm[~np.isnan(sm)]))!=0)[0]; print('roads crossing ~',xs[~np.isnan(sm)][k][:3].round(0))
