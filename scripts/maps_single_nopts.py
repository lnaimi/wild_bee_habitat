import numpy as np, pandas as pd, rasterio, json, joblib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.metrics import roc_auc_score
NR,NC,CS,XLL,YLL=246,415,300.0,266900.0,3535978.0
S=pd.read_csv('bee_training_final.csv'); FEAT=json.load(open('best_params.json'))['features']; te=S[S.split=='test']; P1=S[S.label==1]
LBL={'RF':'Random forest','XGBoost':'XGBoost','SVM':'Support vector machine (SVM)','MLP':'Neural network (MLP)'}
CLS=['#2a78d6','#1baf7a','#eda100','#e05a2b','#9b1d4a']; names_cls=['Very low','Low','Moderate','High','Very high']
cmap=ListedColormap(CLS); cmap.set_bad('#ffffff'); norm=BoundaryNorm([0,.2,.4,.6,.8,1.0001],5)
ext=[XLL/1000,(XLL+NC*CS)/1000,YLL/1000,(YLL+NR*CS)/1000]; INK='#0b0b0b'
plt.rcParams.update({'font.family':'DejaVu Sans','xtick.color':'#52514e','ytick.color':'#52514e'})
CT=pd.read_csv('suitability_classes.csv')
for i,(k,lab) in enumerate(LBL.items(),start=1):
    a=rasterio.open(f'suitability_{k}.tif').read(1)
    auc=roc_auc_score(te.label,joblib.load(f'{k}.joblib').predict_proba(te[FEAT].values)[:,1])
    fig,ax=plt.subplots(figsize=(9,5.6),dpi=250)
    im=ax.imshow(np.ma.masked_equal(a,-9999),cmap=cmap,norm=norm,extent=ext,interpolation='nearest')
    ax.set_title(f'Wild bee habitat suitability: {lab}  (test AUC = {auc:.3f})',loc='left',fontsize=11,color=INK)
    ax.set_xlabel('UTM E (km), zone 30N',fontsize=8); ax.set_ylabel('UTM N (km)',fontsize=8); ax.tick_params(labelsize=7); ax.set_aspect('equal')
    for s in ax.spines.values(): s.set_visible(False)
    c=CT[CT.model==k].set_index('class')
    cb=fig.colorbar(im,ax=ax,ticks=[.1,.3,.5,.7,.9],fraction=.03,pad=.02); cb.outline.set_visible(False)
    cb.ax.set_yticklabels([f'{n} ({lo:.1f}–{lo+.2:.1f})  ·  {c.loc[n,"area_pct"]:.0f}% of area' for n,lo in zip(names_cls,[0,.2,.4,.6,.8])],fontsize=7.5)
    # north arrow + scale bar
    ax.annotate('N',xy=(388,3606),xytext=(388,3598),ha='center',fontsize=10,fontweight='bold',arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.2))
    ax.plot([272,292],[3603,3603],color=INK,lw=2.5); ax.text(282,3604.5,'20 km',ha='center',fontsize=7.5)
    fig.savefig(f'fig_map_{i}_{k}.png',bbox_inches='tight'); plt.close()
    print(k, round(auc,3), c[['area_pct','presence_pct']].to_dict())
