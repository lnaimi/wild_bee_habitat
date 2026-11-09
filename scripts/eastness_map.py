import numpy as np, pandas as pd, rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
NR,NC,CS,XLL,YLL=2460,4150,30.0,266900.0,3535978.0; TOP=YLL+NR*CS
G=pd.read_pickle('grid_features.pkl')
e=np.full(NR*NC,np.nan,np.float32); e[G.id.values-1]=np.sin(np.radians(G.aspect_deg.values)); e=e.reshape(NR,NC)
# GeoTIFF
a=np.where(np.isnan(e),-9999,e).astype(np.float32)
with rasterio.open('eastness_300m.tif','w',driver='GTiff',height=NR,width=NC,count=1,dtype='float32',crs=CRS.from_epsg(32630),transform=from_origin(XLL,TOP,CS,CS),nodata=-9999,compress='deflate') as r: r.write(a,1)
v=e[~np.isnan(e)]
print('cells',v.size,'min',v.min().round(3),'max',v.max().round(3),'mean',v.mean().round(3),
      'east-facing (>0.5)',round((v>0.5).mean()*100,1),'% west-facing (<-0.5)',round((v<-0.5).mean()*100,1),'% N/S (|e|<=0.5)',round((np.abs(v)<=0.5).mean()*100,1),'%')
# figure
INK='#0b0b0b'
cmap=LinearSegmentedColormap.from_list('div',['#104281','#2a78d6','#86b6ef','#f0efec','#f2a19f','#e34948','#8f1d1d']); cmap.set_bad('#ffffff')
plt.rcParams.update({'font.family':'DejaVu Sans','xtick.color':'#52514e','ytick.color':'#52514e'})
ext=[XLL/1000,(XLL+NC*CS)/1000,YLL/1000,TOP/1000]
fig,ax=plt.subplots(figsize=(9,5.6),dpi=250)
im=ax.imshow(np.ma.masked_invalid(e),cmap=cmap,vmin=-1,vmax=1,extent=ext,interpolation='nearest')
ax.set_title('Eastness (sin of aspect), 300 m grid',loc='left',fontsize=11,color=INK)
ax.set_xlabel('UTM E (km), zone 30N',fontsize=8); ax.set_ylabel('UTM N (km)',fontsize=8); ax.tick_params(labelsize=7); ax.set_aspect('equal')
for s in ax.spines.values(): s.set_visible(False)
ax.annotate('N',xy=(388,3606),xytext=(388,3598),ha='center',fontsize=10,fontweight='bold',arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.2))
ax.plot([272,292],[3603,3603],color=INK,lw=2.5); ax.text(282,3604.5,'20 km',ha='center',fontsize=7.5)
cb=fig.colorbar(im,ax=ax,fraction=.03,pad=.02,ticks=[-1,-0.5,0,0.5,1]); cb.outline.set_visible(False)
cb.ax.set_yticklabels(['−1  West-facing','−0.5','0  North / south-facing','0.5','+1  East-facing'],fontsize=7.5)
fig.savefig('fig_eastness_map.png',bbox_inches='tight'); plt.close()
