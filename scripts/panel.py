import numpy as np, rasterio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
NR,NC,CS,XLL,YLL=2460,4150,30.0,266900.0,3535978.0
CLS=['#2a78d6','#1baf7a','#eda100','#e05a2b','#9b1d4a']
cmap=ListedColormap(CLS); cmap.set_bad('#ffffff'); norm=BoundaryNorm([0,.2,.4,.6,.8,1.0001],5)
ext=[XLL/1000,(XLL+NC*CS)/1000,YLL/1000,(YLL+NR*CS)/1000]; INK='#0b0b0b'
plt.rcParams.update({'font.family':'DejaVu Sans','xtick.color':'#52514e','ytick.color':'#52514e'})
L=[('RF','(a) Random forest'),('XGBoost','(b) XGBoost'),('SVM','(c) Support vector machine'),('MLP','(d) Multilayer perceptron')]
fig,axs=plt.subplots(2,2,figsize=(12,8),dpi=250)
for ax,(k,t) in zip(axs.ravel(),L):
    a=rasterio.open(f'suitability_{k}.tif').read(1)
    ax.imshow(np.ma.masked_equal(a,-9999),cmap=cmap,norm=norm,extent=ext,interpolation='nearest')
    ax.set_title(t,loc='left',fontsize=12,color=INK); ax.tick_params(labelsize=7); ax.set_aspect('equal')
    for s in ax.spines.values(): s.set_visible(False)
for ax in axs[:,0]: ax.set_ylabel('UTM N (km)',fontsize=8)
for ax in axs[1]: ax.set_xlabel('UTM E (km)',fontsize=8)
ax=axs[0,0]
ax.annotate('N',xy=(388,3606),xytext=(388,3598),ha='center',fontsize=10,fontweight='bold',arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.2))
ax.plot([272,292],[3603,3603],color=INK,lw=2.5); ax.text(282,3604.5,'20 km',ha='center',fontsize=7.5)
names=['Very low (0.0–0.2)','Low (0.2–0.4)','Moderate (0.4–0.6)','High (0.6–0.8)','Very high (0.8–1.0)']
fig.legend([Patch(color=c) for c in CLS],names,loc='lower center',ncol=5,frameon=False,fontsize=9)
fig.tight_layout(rect=[0,.04,1,1]); fig.savefig('fig_maps_panel.png',bbox_inches='tight')
