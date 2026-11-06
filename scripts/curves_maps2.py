import numpy as np, pandas as pd, joblib, json, rasterio, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.inspection import partial_dependence
S=pd.read_csv('bee_training_final.csv'); FEAT=json.load(open('best_params.json'))['features']
tr=S[S.split=='train']; X=tr[FEAT]
names=['RF','XGBoost','SVM','MLP']; LBL={'RF':'Random forest','XGBoost':'XGBoost','SVM':'SVM','MLP':'Neural network (MLP)'}
COL={'RF':'#2a78d6','XGBoost':'#eb6834','SVM':'#1baf7a','MLP':'#eda100'}
M={n:joblib.load(f'{n}.joblib') for n in names}
INK,INK2,GRID='#0b0b0b','#52514e','#e4e3df'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,'axes.spines.top':False,'axes.spines.right':False})
VARS=[('slope_deg','Slope (°)',1),('dist_roads_m','Distance to roads (km)',1000),('dist_faults_m','Distance to faults (km)',1000),('elevation_m','Elevation (m)',1),('northness','Northness (−1 south … +1 north)',1),('eastness','Eastness (−1 west … +1 east)',1)]
rows=[]
fig,axs=plt.subplots(2,3,figsize=(11,6.6),dpi=200,sharey=True)
for ax,(v,lab,sc) in zip(axs.ravel(),VARS):
    j=FEAT.index(v); lo,hi=np.percentile(X[v],[2,98]); grid=np.linspace(lo,hi,40)
    for n in names:
        pd_=partial_dependence(M[n],X,[j],grid_resolution=40,kind='average',custom_values={j:grid}) if 'custom_values' in partial_dependence.__code__.co_varnames else partial_dependence(M[n],X,[j],grid_resolution=40,percentiles=(0.02,0.98),kind='average')
        xs=pd_['grid_values'][0]; ys=pd_['average'][0]
        ax.plot(xs/sc,ys,color=COL[n],lw=2,label=LBL[n])
        for a,b in zip(xs,ys): rows.append({'variable':v,'model':n,'value':a,'probability':b})
    yv=X[v][tr.label==1]/sc; yb=X[v][tr.label==0]/sc
    ax.plot(yv,np.full(len(yv),0.015),'|',color=INK,ms=6,alpha=.6); ax.plot(yb,np.full(len(yb),-0.02),'|',color='#9a9992',ms=6,alpha=.6)
    ax.set_xlabel(lab); ax.grid(color=GRID,lw=.6); ax.set_axisbelow(True); ax.set_ylim(-0.04,1)
for ax in axs[:,0]: ax.set_ylabel('Predicted probability of presence')
h,l=axs[0,0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(0.5,0.955),labelcolor=INK)
fig.suptitle('Response curves (partial dependence). Ticks along the bottom: presences (dark), background (grey)',x=.01,ha='left',y=.995,fontsize=10.5,color=INK)
fig.tight_layout(rect=(0,0,1,.93)); fig.savefig('fig_response_curves.png'); plt.close()
pd.DataFrame(rows).round(4).to_csv('response_curves.csv',index=False)
# categorical responses: mean predicted prob when the cell is set to each class (others at observed)
cat=[]
for grp,cols,labs in [('Geology',['geol_1','geol_2','geol_3','geol_4'],['Unit 1','Unit 2','Unit 3','Unit 4']),('Land cover',['lulc_tree','lulc_grass','lulc_bare','lulc_other'],['Tree cover','Grassland','Bare / sparse','Other'])]:
    for c,lb in zip(cols,labs):
        Xc=X.copy(); Xc[cols]=0; Xc[c]=1
        for n in names: cat.append({'group':grp,'class':lb,'model':n,'probability':M[n].predict_proba(Xc.values)[:,1].mean()})
cat=pd.DataFrame(cat); cat.round(4).to_csv('response_categorical.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(10,3.6),dpi=200,sharey=True)
for ax,g in zip(axs,['Geology','Land cover']):
    d=cat[cat.group==g]; cls_=list(dict.fromkeys(d['class'])); w=.19
    for k,n in enumerate(names):
        vals=d[d.model==n].set_index('class').loc[cls_,'probability']
        ax.bar(np.arange(len(cls_))+(k-1.5)*w,vals,width=w-.02,color=COL[n],label=LBL[n],edgecolor='white',linewidth=.8)
    ax.set_xticks(range(len(cls_))); ax.set_xticklabels(cls_); ax.set_title(g,loc='left',fontsize=10,color=INK); ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True); ax.set_ylim(0,1)
axs[0].set_ylabel('Mean predicted probability'); axs[1].legend(frameon=False,fontsize=8,loc='upper right',labelcolor=INK)
fig.tight_layout(); fig.savefig('fig_response_categorical.png'); plt.close()
print(cat.pivot_table(index=['group','class'],columns='model',values='probability').round(2))
# ---- maps with distinct class colours
NR,NC,CS,XLL,YLL=246,415,300.0,266900.0,3535978.0
CLS=['#2a78d6','#1baf7a','#eda100','#e05a2b','#9b1d4a']; names_cls=['Very low','Low','Moderate','High','Very high']
cmap=ListedColormap(CLS); cmap.set_bad('#ffffff'); norm=BoundaryNorm([0,.2,.4,.6,.8,1.0001],5)
ext=[XLL/1000,(XLL+NC*CS)/1000,YLL/1000,(YLL+NR*CS)/1000]; P1=S[S.label==1]
R={k:rasterio.open(f'suitability_{k}.tif').read(1) for k in names+['ensemble']}
def draw(ax,k,title):
    im=ax.imshow(np.ma.masked_equal(R[k],-9999),cmap=cmap,norm=norm,extent=ext,interpolation='nearest')
    ax.scatter(P1.x/1000,P1.y/1000,s=7,color=INK,edgecolor='white',linewidth=.5,zorder=3)
    ax.set_title(title,loc='left',fontsize=10,color=INK); ax.set_aspect('equal'); ax.tick_params(labelsize=7)
    for s in ax.spines.values(): s.set_visible(False)
    return im
fig,axs=plt.subplots(2,2,figsize=(10,6.2),dpi=200)
for ax,n in zip(axs.ravel(),names): im=draw(ax,n,LBL[n])
for ax in axs[:,0]: ax.set_ylabel('UTM N (km)',fontsize=8)
for ax in axs[1,:]: ax.set_xlabel('UTM E (km)',fontsize=8)
cb=fig.colorbar(im,ax=axs,ticks=[.1,.3,.5,.7,.9],fraction=.025,pad=.02); cb.ax.set_yticklabels(names_cls,fontsize=8); cb.outline.set_visible(False)
fig.suptitle('Wild bee habitat suitability by model (black dots: presences)',x=.02,y=.97,ha='left',fontsize=11,color=INK)
fig.savefig('fig_maps_4models.png',bbox_inches='tight'); plt.close()
fig,ax=plt.subplots(figsize=(9,5.6),dpi=200); im=draw(ax,'ensemble','Wild bee habitat suitability, ensemble mean of 4 models')
ax.set_xlabel('UTM E (km), zone 30N',fontsize=8); ax.set_ylabel('UTM N (km)',fontsize=8)
cb=fig.colorbar(im,ax=ax,ticks=[.1,.3,.5,.7,.9],fraction=.03,pad=.02); cb.ax.set_yticklabels([f'{c} ({lo:.1f}–{lo+.2:.1f})' for c,lo in zip(names_cls,[0,.2,.4,.6,.8])],fontsize=8); cb.outline.set_visible(False)
ax.scatter([],[],s=14,color=INK,edgecolor='white',label='Bee presence (n = 150)'); ax.legend(loc='lower left',frameon=False,fontsize=8)
fig.savefig('fig_map_ensemble.png',bbox_inches='tight'); plt.close()
# ArcGIS/QGIS colour hint: classified ensemble raster (1-5) + colour table
cl=np.full(R['ensemble'].shape,0,np.uint8); m=R['ensemble']!=-9999; cl[m]=np.digitize(R['ensemble'][m],[.2,.4,.6,.8])+1
with rasterio.open('suitability_ensemble.tif') as src: prof=src.profile
prof.update(dtype='uint8',nodata=0)
with rasterio.open('suitability_ensemble_classes.tif','w',**prof) as d:
    d.write(cl,1); d.write_colormap(1,{0:(255,255,255,0),**{i+1:tuple(int(c[k:k+2],16) for k in (1,3,5))+(255,) for i,c in enumerate(CLS)}})
