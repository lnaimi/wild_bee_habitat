import numpy as np, pandas as pd, rasterio
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
NR,NC,CS,XLL,YLL=246,415,300.0,266900.0,3535978.0
S=pd.read_csv('bee_training_final.csv'); INK='#0b0b0b'
ext=[XLL/1000,(XLL+NC*CS)/1000,YLL/1000,(YLL+NR*CS)/1000]
plt.rcParams.update({'font.family':'DejaVu Sans','xtick.color':'#52514e','ytick.color':'#52514e'})
def deco(ax):
    ax.set_xlabel('UTM E (km), zone 30N',fontsize=8); ax.set_ylabel('UTM N (km)',fontsize=8); ax.tick_params(labelsize=7); ax.set_aspect('equal')
    for s in ax.spines.values(): s.set_visible(False)
    ax.annotate('N',xy=(388,3606),xytext=(388,3598),ha='center',fontsize=10,fontweight='bold',arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.2))
    ax.plot([272,292],[3603,3603],color=INK,lw=2.5); ax.text(282,3604.5,'20 km',ha='center',fontsize=7.5)
# ensemble with test presences only
CLS=['#2a78d6','#1baf7a','#eda100','#e05a2b','#9b1d4a']; nm=['Very low','Low','Moderate','High','Very high']
cmap=ListedColormap(CLS); cmap.set_bad('#ffffff'); norm=BoundaryNorm([0,.2,.4,.6,.8,1.0001],5)
CT=pd.read_csv('suitability_classes.csv'); c=CT[CT.model=='ensemble'].set_index('class')
a=rasterio.open('suitability_ensemble.tif').read(1); T=S[(S.split=='test')&(S.label==1)]
fig,ax=plt.subplots(figsize=(9,5.6),dpi=250)
im=ax.imshow(np.ma.masked_equal(a,-9999),cmap=cmap,norm=norm,extent=ext,interpolation='nearest')
ax.scatter(T.x/1000,T.y/1000,s=10,color=INK,edgecolor='white',linewidth=.6,zorder=3,label=f'Test presences (n = {len(T)}, not used in training)')
ax.set_title('Wild bee habitat suitability: ensemble mean of 4 models',loc='left',fontsize=11,color=INK); deco(ax)
cb=fig.colorbar(im,ax=ax,ticks=[.1,.3,.5,.7,.9],fraction=.03,pad=.02); cb.outline.set_visible(False)
cb.ax.set_yticklabels([f'{n} ({lo:.1f}–{lo+.2:.1f})  ·  {c.loc[n,"area_pct"]:.0f}% of area' for n,lo in zip(nm,[0,.2,.4,.6,.8])],fontsize=7.5)
ax.legend(loc='lower left',frameon=False,fontsize=8); fig.savefig('fig_map_ensemble.png',bbox_inches='tight'); plt.close()
tp=rasterio.open('suitability_ensemble.tif').read(1).ravel()[(T.cell.values-1)]
print('test presences in high/very high:',round((tp>=.6).mean()*100,1),'%')
# study area & sampling figure over elevation (neutral grey ramp)
G=pd.read_pickle('grid_features.pkl'); e=np.full(NR*NC,np.nan); e[G.id.values-1]=G.elevation_m.values; e=e.reshape(NR,NC)
gcm=LinearSegmentedColormap.from_list('g',['#f2f1ed','#b9b8b1','#6f6e68','#3a3936']); gcm.set_bad('#ffffff')
fig,ax=plt.subplots(figsize=(9,5.6),dpi=250)
im=ax.imshow(np.ma.masked_invalid(e),cmap=gcm,extent=ext,interpolation='nearest')
P=S[S.label==1]; A=S[S.label==0]
ax.scatter(A.x/1000,A.y/1000,s=13,marker='s',color='#2a78d6',edgecolor='white',linewidth=.6,zorder=3,label=f'Verified absences (n = {len(A)})')
ax.scatter(P.x/1000,P.y/1000,s=16,marker='o',color='#eb6834',edgecolor='white',linewidth=.6,zorder=4,label='Verified bee colony presences (n = 150)')
ax.set_title('Study area and sampling points',loc='left',fontsize=11,color=INK); deco(ax)
cb=fig.colorbar(im,ax=ax,fraction=.03,pad=.02); cb.set_label('Elevation (m)',fontsize=8); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
ax.legend(loc='lower left',frameon=False,fontsize=7.5)
# Morocco locator inset (Natural Earth 1:10m, Morocco point-of-view boundaries)
import json
from matplotlib.patches import Polygon as MPoly
from pyproj import Transformer
NE=json.load(open('ne_10m_admin_0_countries_mar.geojson'))
ia=ax.inset_axes([0.0,0.58,0.24,0.42])
def rings(g):
    polys=g['coordinates'] if g['type']=='MultiPolygon' else [g['coordinates']]
    for poly in polys: yield np.array(poly[0])
for f in NE['features']:
    a3=f['properties'].get('ADM0_A3'); g=f['geometry']
    if g is None: continue
    for r in rings(g):
        if r[:,0].max()<-20 or r[:,0].min()>6 or r[:,1].max()<19 or r[:,1].min()>38: continue
        ia.add_patch(MPoly(r,closed=True,fc='#e8e7e2' if a3=='MAR' else '#f7f7f5',ec='#6f6e68' if a3=='MAR' else '#cfcec8',lw=.5 if a3=='MAR' else .3))
tr=Transformer.from_crs(32630,4326,always_xy=True)
bx=[XLL,XLL+NC*CS,XLL+NC*CS,XLL,XLL]; by=[YLL,YLL,YLL+NR*CS,YLL+NR*CS,YLL]
lo,la=tr.transform(bx,by)
valid=~np.isnan(e); rr,cc=np.where(valid)
lon,lat=tr.transform(XLL+(cc+.5)*CS,YLL+(NR-rr-.5)*CS)
ia.scatter(lon[::7],lat[::7],s=.4,color='#e05a2b',linewidths=0,zorder=3)
ia.plot(lo,la,color='#9b1d4a',lw=.8,zorder=4)
ia.text(-7.2,31.3,'Morocco',fontsize=6.5,color='#3a3936',ha='center')
ia.text(-1.5,33.6,'Algeria',fontsize=5.5,color='#8a8983',ha='center')
ia.text(-12.6,32.5,'Atlantic\nOcean',fontsize=5,color='#8a8983',ha='center',style='italic')
ia.set_xlim(-17.5,0.5); ia.set_ylim(20.5,36.3); ia.set_aspect('equal'); ia.set_xticks([]); ia.set_yticks([])
ia.set_facecolor('#ffffff')
for ln in ax.lines[:1]: ln.remove()
for t in [t for t in ax.texts if t.get_text()=='20 km']: t.remove()
ax.plot([368,388],[3539.5,3539.5],color=INK,lw=2.5); ax.text(378,3541,'20 km',ha='center',fontsize=7.5)
for sp in ia.spines.values(): sp.set_edgecolor('#b9b8b1'); sp.set_linewidth(.6)
fig.savefig('fig_study_area_sampling.png',bbox_inches='tight'); plt.close()
