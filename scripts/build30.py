# Build the 30 m predictor stack (3 x 3 = 90 m focal window) for the wild bee habitat model
import numpy as np, pandas as pd, rasterio, shapefile, pyproj, time
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt, uniform_filter
t0=time.time()
U='/mnt/user-data/uploads/'; UTM=CRS.from_epsg(32630)
NR3,NC3,XLL,YLL=246,415,266900.0,3535978.0; TOP=YLL+NR3*300
CS=30.0; NR,NC=NR3*10,NC3*10; T30=from_origin(XLL,TOP,CS,CS)
def to30(path,res,src_crs=None,window=None,nodata='file'):
    with rasterio.open(path) as r:
        src=(r.read(1,window=window) if window is not None else r.read(1)).astype('float32')
        st=r.window_transform(window) if window is not None else r.transform
        nd=r.nodata if nodata=='file' else nodata
        if nd is not None: src[src==nd]=np.nan
        src[~np.isfinite(src)]=np.nan; src[np.abs(src)>1e30]=np.nan
        dst=np.full((NR,NC),np.nan,'float32')
        reproject(src,dst,src_transform=st,src_crs=src_crs or r.crs or UTM,dst_transform=T30,dst_crs=UTM,resampling=res,src_nodata=np.nan,dst_nodata=np.nan)
    return dst
def focal(a,k=3):
    v=np.isfinite(a); s=uniform_filter(np.where(v,a,0).astype('float64'),k,mode='nearest'); n=uniform_filter(v.astype('float64'),k,mode='nearest')
    with np.errstate(invalid='ignore',divide='ignore'): out=(s/n).astype('float32')
    out[~v]=np.nan; return out
L={}
dem=to30(U+'bee/data/original/dem_map',Resampling.nearest)
L['elevation_m']=focal(dem)
L['slope_deg']=focal(to30(U+'bee/data/original/slope_map.tif',Resampling.nearest))
# aspect and curvature from the 30 m DEM (ArcGIS / Zevenbergen-Thorne formulas)
Z=np.pad(dem,1,mode='edge'); c=30.0
a,b,cc=Z[:-2,:-2],Z[:-2,1:-1],Z[:-2,2:]; d,e,f=Z[1:-1,:-2],Z[1:-1,1:-1],Z[1:-1,2:]; g,h,i=Z[2:,:-2],Z[2:,1:-1],Z[2:,2:]
dzdx=((cc+2*f+i)-(a+2*d+g))/(8*c); dzdy=((g+2*h+i)-(a+2*b+cc))/(8*c)
asp=np.degrees(np.arctan2(dzdy,-dzdx)); comp=np.where(asp<0,90-asp,np.where(asp>90,360-asp+90,90-asp)); flat=(dzdx==0)&(dzdy==0)
D=((d+f)/2-e)/c**2; E=((b+h)/2-e)/c**2; F=(-a+cc+g-i)/(4*c**2); G=(-d+f)/(2*c); H=(b-h)/(2*c); den=G**2+H**2
with np.errstate(invalid='ignore',divide='ignore'):
    prof=np.where(den>0,-2*(D*G**2+E*H**2+F*G*H)/den*100,0.0).astype('float32'); plan=np.where(den>0,2*(D*H**2+E*G**2-F*G*H)/den*100,0.0).astype('float32')
for arr in (prof,plan): arr[np.isnan(dem)]=np.nan
L['plan_curvature']=focal(plan); L['profile_curvature']=focal(prof)
del a,b,cc,d,e,f,g,h,i,D,E,F,G,H,den,prof,plan,Z
rad=np.radians(comp); w=np.where(flat|np.isnan(comp),0.0,1.0)
sn=uniform_filter(np.nan_to_num(np.sin(rad))*w,3,mode='nearest'); cn=uniform_filter(np.nan_to_num(np.cos(rad))*w,3,mode='nearest')
mag=np.hypot(sn,cn); asp90=(np.degrees(np.arctan2(sn,cn))+360)%360
with np.errstate(invalid='ignore',divide='ignore'):
    L['northness']=np.where(mag>1e-6,cn/mag,0).astype('float32'); L['eastness']=np.where(mag>1e-6,sn/mag,0).astype('float32')
L['aspect_deg']=np.where(mag>1e-6,asp90,-1).astype('float32')
for k in ('northness','eastness','aspect_deg'): L[k][np.isnan(dem)]=np.nan
del comp,rad,w,sn,cn,mag,asp90,dzdx,dzdy,asp,flat
print('terrain done',round(time.time()-t0))
L['ls_factor']=focal(to30(U+'e/WaTEM_inputs/run_1936_1_LSfactor.rst',Resampling.nearest,src_crs=UTM,nodata=None))
L['dist_faults_m']=to30(U+'e/MANAOUCH/Landslides_fuzzy/fault_eucdis1',Resampling.nearest)
L['dist_streams_m']=to30(U+'e/MANAOUCH/Landslides_fuzzy/Streams_euclidist.tif',Resampling.nearest)
L['erodibility_k']=focal(to30(U+'bee/data/original/Kajuste_map.tif',Resampling.nearest))
tr=pyproj.Transformer.from_crs(4326,32630,always_xy=True); sf=shapefile.Reader(U+'e/MANAOUCH/Landslides_fuzzy/roadsnrr.shp'); shapes=[]
for s in sf.shapes():
    pts=s.points; parts=list(s.parts)+[len(pts)]
    for p0,p1 in zip(parts[:-1],parts[1:]):
        xs,ys=tr.transform([p[0] for p in pts[p0:p1]],[p[1] for p in pts[p0:p1]])
        if len(xs)>1: shapes.append(({'type':'LineString','coordinates':list(zip(xs,ys))},1))
rr=rasterize(shapes,out_shape=(NR,NC),transform=T30,fill=0,all_touched=True,dtype='uint8')
L['dist_roads_m']=(distance_transform_edt(rr==0)*CS).astype('float32'); del rr
with rasterio.open(U+'bee/data/original/CHELSA_bio12_1981-2010_V.2.1.tif') as r:
    win=rasterio.windows.from_bounds(-5.7,31.8,-3.95,32.8,r.transform).round_offsets().round_lengths()
L['rainfall_mm']=to30(U+'bee/data/original/CHELSA_bio12_1981-2010_V.2.1.tif',Resampling.bilinear,window=win)
print('distances/climate done',round(time.time()-t0))
# land cover: WorldCover 10 m -> class fractions within the 90 m window
with rasterio.open(U+'bee/data/original/ESA_WorldCover_10m_2021_v200_N30W006_Map.tif') as r:
    win=rasterio.windows.from_bounds(-5.6,31.85,-4.05,32.7,r.transform).round_offsets().round_lengths(); src=r.read(1,window=win); st=r.window_transform(win)
T10=from_origin(XLL,TOP,10,10); lc10=np.zeros((NR*3,NC*3),'uint8')
reproject(src,lc10,src_transform=st,src_crs=CRS.from_epsg(4326),dst_transform=T10,dst_crs=UTM,resampling=Resampling.nearest,src_nodata=0,dst_nodata=0); del src
def frac(mask): return uniform_filter(mask.reshape(NR,3,NC,3).mean(axis=(1,3)).astype('float32'),3,mode='nearest')
valid_lc=lc10>0
fr={k:frac(m) for k,m in {'tree':lc10==10,'grass':lc10==30,'bare':lc10==60,'valid':valid_lc}.items()}
with np.errstate(invalid='ignore',divide='ignore'):
    for k in ('tree','grass','bare'): L['lulc_'+k]=np.where(fr['valid']>0,fr[k]/fr['valid'],np.nan).astype('float32')
L['lulc_other']=(1-L['lulc_tree']-L['lulc_grass']-L['lulc_bare']).clip(0,1).astype('float32')
# dominant class at the 30 m pixel itself (for reporting)
lcm=lc10.reshape(NR,3,NC,3)[:,1,:,1].astype('float32'); lcm[lcm==0]=np.nan; L['lulc_code']=lcm; del lc10,fr
# geology: only available on the 300 m grid -> nearest neighbour
G3=pd.read_pickle('../v2/grid_features.pkl'); geo3=np.full(NR3*NC3,np.nan,'float32'); geo3[G3.id.values-1]=G3.geology_unit.values
geo=np.repeat(np.repeat(geo3.reshape(NR3,NC3),10,0),10,1); L['geology_unit']=geo
for u in (1,2,3,4): L[f'geol_{u}']=np.where(np.isnan(geo),np.nan,(geo==u).astype('float32'))
# study-area mask: the 300 m study mask, and every layer present
m3=np.zeros(NR3*NC3,bool); m3[G3.id.values-1]=True; mask=np.repeat(np.repeat(m3.reshape(NR3,NC3),10,0),10,1)
for k,v in L.items(): mask&=np.isfinite(v)
print('valid 30 m cells',int(mask.sum()),'area km2',round(mask.sum()*0.0009,1))
np.save('mask30.npy',mask)
np.savez('stack30.npz',**L)
print('saved',round(time.time()-t0),'s')
