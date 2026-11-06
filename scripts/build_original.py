import numpy as np, pandas as pd, rasterio, shapefile, pyproj
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt
U='./raw_data/'  # folder with the original rasters (see paper, predictor table)
UTM=CRS.from_epsg(32630)
NR,NC,CS,XLL,YLL=246,415,300.0,266900.0,3535978.0
TOP=YLL+NR*CS
T300=from_origin(XLL,TOP,CS,CS)
T30=from_origin(XLL,TOP,30,30); NR30,NC30=NR*10,NC*10

old=pd.read_csv('./raw_data/bee_habitat_parameters_300m.csv')
mask=np.zeros(NR*NC,bool); mask[old['id'].values-1]=True; mask=mask.reshape(NR,NC)

def to_grid(path, resampling, dst_t=T300, shape=(NR,NC), src_nodata='file', band=1, src_crs=None, window=None):
    with rasterio.open(path) as r:
        src=r.read(band, window=window).astype('float64') if window is not None else r.read(band).astype('float64')
        st=r.window_transform(window) if window is not None else r.transform
        nd=r.nodata if src_nodata=='file' else src_nodata
        if nd is not None: src[src==nd]=np.nan
        src[~np.isfinite(src)]=np.nan; src[np.abs(src)>1e30]=np.nan
        dst=np.full(shape,np.nan)
        reproject(src,dst,src_transform=st,src_crs=src_crs or r.crs or UTM,dst_transform=dst_t,dst_crs=UTM,
                  resampling=resampling,src_nodata=np.nan,dst_nodata=np.nan)
    return dst
L={}
# --- DEM at 30 m (aligned) for derivatives
dem30=to_grid(U+'bee/data/original/dem_map',Resampling.nearest,T30,(NR30,NC30))
print('dem30 range',np.nanmin(dem30),np.nanmax(dem30))
L['elevation']=to_grid(U+'bee/data/original/dem_map',Resampling.average)
L['slope']=to_grid(U+'bee/data/original/slope_map.tif',Resampling.average)
# aspect + curvature on 30 m DEM (ArcGIS formulas)
Z=np.pad(dem30,1,mode='edge'); c=30.0
a,b,cc=Z[:-2,:-2],Z[:-2,1:-1],Z[:-2,2:]
d,e,f=Z[1:-1,:-2],Z[1:-1,1:-1],Z[1:-1,2:]
g,h,i=Z[2:,:-2],Z[2:,1:-1],Z[2:,2:]
dzdx=((cc+2*f+i)-(a+2*d+g))/(8*c); dzdy=((g+2*h+i)-(a+2*b+cc))/(8*c)
asp=np.degrees(np.arctan2(dzdy,-dzdx)); comp=np.where(asp<0,90-asp,np.where(asp>90,360-asp+90,90-asp))
flat=(dzdx==0)&(dzdy==0)
D=((d+f)/2-e)/c**2; E=((b+h)/2-e)/c**2; F=(-a+cc+g-i)/(4*c**2); G=(-d+f)/(2*c); H=(b-h)/(2*c)
den=G**2+H**2
with np.errstate(invalid='ignore',divide='ignore'):
    prof=np.where(den>0,-2*(D*G**2+E*H**2+F*G*H)/den*100,0.0)
    plan=np.where(den>0,2*(D*H**2+E*G**2-F*G*H)/den*100,0.0)
for arr in (prof,plan,comp): arr[np.isnan(dem30)]=np.nan
def agg30(x,fn=np.nanmean):
    with np.errstate(all='ignore'):
        return fn(x.reshape(NR,10,NC,10),axis=(1,3))
L['plancurv']=agg30(plan); L['profcurv']=agg30(prof)
rad=np.radians(comp); w=np.where(flat|np.isnan(comp),np.nan,1.0)
sn=agg30(np.sin(rad)*w); cn=agg30(np.cos(rad)*w)
aspect=(np.degrees(np.arctan2(sn,cn))+360)%360
aspect[np.hypot(sn,cn)<1e-6]=-1
L['aspect']=aspect
L['ls']=to_grid(U+'e/WaTEM_inputs/run_1936_1_LSfactor.rst',Resampling.average,src_nodata=None,src_crs=UTM)
L['dtof']=to_grid(U+'e/MANAOUCH/Landslides_fuzzy/fault_eucdis1',Resampling.average)
L['dtos']=to_grid(U+'e/MANAOUCH/Landslides_fuzzy/Streams_euclidist.tif',Resampling.average)
# roads: rasterize at 10 m then EDT, average to 300 m
tr=pyproj.Transformer.from_crs(4326,32630,always_xy=True)
sf=shapefile.Reader(U+'e/MANAOUCH/Landslides_fuzzy/roadsnrr.shp'); shapes=[]
for s in sf.shapes():
    pts=s.points; parts=list(s.parts)+[len(pts)]
    for p0,p1 in zip(parts[:-1],parts[1:]):
        xs,ys=tr.transform([p[0] for p in pts[p0:p1]],[p[1] for p in pts[p0:p1]])
        if len(xs)>1: shapes.append(({'type':'LineString','coordinates':list(zip(xs,ys))},1))
print('road segments',len(shapes), sf.shapeType)
T10=from_origin(XLL,TOP,10,10)
rr=rasterize(shapes,out_shape=(NR*30,NC*30),transform=T10,fill=0,all_touched=True,dtype='uint8')
dist10=distance_transform_edt(rr==0)*10.0
L['dtor']=dist10.reshape(NR,30,NC,30).mean(axis=(1,3)); del dist10,rr
L['erodibility']=to_grid(U+'bee/data/original/Kajuste_map.tif',Resampling.average)
# CHELSA: read window, bilinear to 300 m
with rasterio.open(U+'bee/data/original/CHELSA_bio12_1981-2010_V.2.1.tif') as r:
    win=rasterio.windows.from_bounds(-5.7,31.8,-3.95,32.8,r.transform).round_offsets().round_lengths()
L['rainfall']=to_grid(U+'bee/data/original/CHELSA_bio12_1981-2010_V.2.1.tif',Resampling.bilinear,window=win)
# WorldCover: window then mode
with rasterio.open(U+'bee/data/original/ESA_WorldCover_10m_2021_v200_N30W006_Map.tif') as r:
    win=rasterio.windows.from_bounds(-5.6,31.85,-4.05,32.7,r.transform).round_offsets().round_lengths()
    src=r.read(1,window=win); st=r.window_transform(win)
lc=np.zeros((NR,NC),dtype='uint8')
reproject(src,lc,src_transform=st,src_crs=CRS.from_epsg(4326),dst_transform=T300,dst_crs=UTM,resampling=Resampling.mode,src_nodata=0,dst_nodata=0)
L['lulc']=np.where(lc==0,np.nan,lc.astype(float))
# geology from old normalized (categorical codes)
gv=old['geology'].values; codes={v:k+1 for k,v in enumerate(sorted(np.unique(gv)))}; print('geology codes',codes)
geo=np.full(NR*NC,np.nan); geo[old['id'].values-1]=[codes[v] for v in gv]; L['geology']=geo.reshape(NR,NC)

order=['elevation','slope','aspect','plancurv','profcurv','ls','rainfall','dtos','dtor','dtof','geology','lulc','erodibility']
r,cidx=np.meshgrid(np.arange(NR),np.arange(NC),indexing='ij')
df=pd.DataFrame({'id':(r*NC+cidx+1).ravel(),'row':r.ravel(),'col':cidx.ravel(),
   'x':(XLL+(cidx+.5)*CS).ravel(),'y':(YLL+(NR-r-.5)*CS).ravel(),**{k:L[k].ravel() for k in order}})
df=df[mask.ravel()]
print('cells in study mask',len(df)); print('missing per layer:'); print(df[order].isna().sum().to_string())
np.save('layers.npy',{k:L[k] for k in order},allow_pickle=True)
df.to_pickle('df.pkl')
