"""Focused synthetic tests; no real GOES data required."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS

import preprocess_detection as p

ATTRS=dict(grid_mapping_name='geostationary',perspective_point_height=35786023.,semi_major_axis=6378137.,semi_minor_axis=6356752.31414,latitude_of_projection_origin=0.,longitude_of_projection_origin=-75.,sweep_angle_axis='x')


def grid():
    h=ATTRS['perspective_point_height']; x=np.linspace(-.151,.151,96)*h
    return p.Grid(x,x[::-1].copy(),CRS.from_cf(ATTRS),h)


def test_projection_and_box():
    g=grid(); x,y=g.pixel(-75.,0.)
    assert np.allclose([x,y],[48,48])
    lon,lat=g.geographic(x,y); assert np.allclose([lon,lat],[-75,0])
    obs=dict(sid='A',lon=-75.,lat=0.,nature='TS',match='exact',offset_s=0.)
    box=p.box_for(g,obs,300)
    labels=p.label_for_tile([box],32,32,32)
    assert labels and np.allclose(labels[0]['values'][4:],[.5,.5])
    assert p.label_for_tile([dict(box,nature='NR')],32,32,32) is None
    assert p.label_for_tile([],0,0,32)==[]


def test_match_dateline_and_gap():
    track=('A',np.array([0,21600]),np.array([0.,0.]),np.array([179.,-179.]),np.array(['TS','TS']))
    assert abs(abs(p.match_track(track,10800)['lon'])-180)<.001
    assert p.match_track(track,0)['match']=='exact'
    assert p.match_track(track,-1000) is None
    assert p.match_track(track,10800,max_gap=100)['unknown']
    transition=(*track[:-1],np.array(['TS','ET']))
    assert p.match_track(transition,300)['nature']=='AMBIGUOUS'


def test_encoding():
    a=np.array([[200.,np.nan]],dtype=np.float32); b=np.array([[190.,200.]],dtype=np.float32)
    rgb,valid=p.encode(a,b)
    assert rgb.shape==(1,2,3) and rgb.dtype==np.uint8
    assert valid.tolist()==[[True,False]] and (rgb[0,1]==0).all()


def make_scene(root, when):
    folder=root/when.strftime('%Y%m%d_%H%M')
    start=when.strftime('%Y%j%H%M%S')+'0'; end=(when+timedelta(minutes=10)).strftime('%Y%j%H%M%S')+'0'
    g=grid()
    for band,sub,temp in [(13,'IR',250),(8,'WV',225)]:
        path=folder/sub; path.mkdir(parents=True)
        ds=xr.Dataset({'CMI':(('y','x'),np.full(g.shape,temp,dtype=np.float32),{'units':'K'}),'DQF':(('y','x'),np.zeros(g.shape,dtype=np.int16)), 'band_id':band,'goes_imager_projection':((),0,ATTRS)},coords={'x':('x',g.x/g.h,{'units':'rad'}),'y':('y',g.y/g.h,{'units':'rad'})})
        ds.to_netcdf(path/f'OR_ABI-L2-CMIPF-M6C{band:02d}_G16_s{start}_e{end}_c{end}.nc',engine='scipy')


def test_end_to_end(tmp_path):
    root=tmp_path/'goes'; root.mkdir(); rows=[]
    start=datetime(2022,1,1,tzinfo=timezone.utc)
    for i in range(12):
        when=start+timedelta(days=i*9)
        make_scene(root,when)
        for hours in (0,6):
            rows.append(dict(SID=f'S{i}',ISO_TIME=(when+timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S'),LAT=0,LON=-75,NATURE='TS'))
    csv=tmp_path/'ibtracs.csv'; pd.DataFrame(rows).to_csv(csv,index=False)
    out=tmp_path/'out'; script=Path(p.__file__)
    cmd=[sys.executable,str(script),'--goes',str(root),'--ibtracs',str(csv),'--output',str(out),'--tile-size','32','--stride','16','--image-size','32','--workers','2']
    subprocess.run(cmd+['--plan-only'],check=True,capture_output=True,text=True)
    assert not out.exists() and (tmp_path/'out.work'/'scene_plan.json').exists()
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    summary=json.loads((out/'summary.json').read_text())
    assert summary['storm_leakage'] is False and summary['scene_leakage'] is False
    assert all(summary['splits'][s]['positive'] and summary['splits'][s]['negative'] for s in p.SPLITS)
    assert not summary['uncovered_storm_observations']
    assert (out/'data.yaml').exists()
    assert all(len(x.read_text().split()) in (0,8) for x in (out/'labels').rglob('*.txt'))
