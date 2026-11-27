from types import SimpleNamespace
import json
import numpy as np
from pyproj import CRS
import pytest
import train_detection as t


def target(sid='A',lon=-75.,lat=0.):
    return dict(sid=sid,lon=lon,lat=lat,nature='TS',match='exact',offset_s=0.)


def pred(scene='S',image='a',index=0,lon=-75.,lat=0.,score=.9,valid=True,x0=0):
    return dict(scene=scene,image=image,index=index,lon=lon if valid else None,lat=lat if valid else None,score=score,valid=valid,x0=x0,y0=0,tile_size=1024)


def test_projection_pixel_edge():
    crs=CRS.from_proj4('+proj=geos +h=35786023 +lon_0=-75 +sweep=x +ellps=WGS84 +units=m')
    p=t.Projection(dict(crs_wkt=crs.to_wkt(),x0_m=-999000,y0_m=999000,dx_m=2000,dy_m=-2000))
    lon,lat,col,row=p.geographic(dict(x0=0,y0=0,tile_size=1000),.5,.5)
    assert np.allclose([lon,lat,col,row],[-75,0,500,500])


def test_dedup_and_invalid():
    raw=[pred(),pred(image='b',lon=-75.1,score=.8,x0=512),pred(index=1,score=.7),pred(image='c',valid=False,score=.6),pred(image='d',score=.5,x0=2048)]
    kept=t.deduplicate(raw,100)
    assert len(kept)==4 and all(p['image']!='b' for p in kept)
    m,*_=t.evaluate_at({'S':kept},{'S':[target()]},.25,150)
    assert (m['tp'],m['fp'],m['fn'])==(1,3,0)


def test_matching_maximum_cardinality():
    # First prediction can match either storm, second can match only storm A.
    p=[pred(lon=.4),pred(lon=-.6,index=1)]; truth=[target(lon=0),target('B',lon=1)]
    assert len(t.match(p,truth,100))==2


def test_ap_and_zero_denominators():
    assert t.centre_ap({'S':[pred()]},{'S':[target(),target('B',lon=-65)]},150)==.5
    assert t.centre_ap({'S':[]},{'S':[target()]},150)==0
    assert t.error_stats([])['mean'] is None
    assert t.binary_counts([(True,False),(False,False)])['accuracy']==.5


def test_report_and_locked_threshold(tmp_path):
    samples={'a':dict(scene='S',objects=[target()]),'b':dict(scene='S',objects=[target()]),'c':dict(scene='N',objects=[])}
    plans={'S':dict(positive_ids=['A','B']),'N':dict(positive_ids=[]),'U':dict(positive_ids=['C'])}
    raw={'S':[pred(),pred(image='b',score=.8)],'N':[pred(scene='N',image='c',score=.2)]}
    args=SimpleNamespace(match_km=150,dedup_km=100,keypoint_conf=.25)
    m=t.geographic_report(raw,{'a':.9,'b':.8,'c':.2},samples,plans,args,tmp_path)
    assert m['tp']==1 and m['fp']==0 and m['coverage']['planned_target_observations']==3
    assert m['coverage']['recall_including_uncovered_as_misses']==pytest.approx(1/3)
    assert m['centre_error_km']['mean']==pytest.approx(0)
    assert (tmp_path/'centre_matches.csv').exists()
    m2=t.geographic_report(raw,{'a':.9,'b':.8,'c':.2},samples,plans,args,tmp_path,.1)
    assert m2['threshold']==.1 and m2['fp']==1


def test_dateline():
    assert t.distances([pred(lon=179.9)],[target(lon=-179.9)])[0,0]<25


def test_v2_split_counts():
    counts = {s: {'images': n} for s, n in zip(t.SPLITS, (80, 10, 10))}
    summary = dict(exact_image_split=True, image_split_fractions=dict(train=.8,val=.1,test=.1), planned_storm_observation_coverage=.95)
    assert t.validate_split_counts(counts, summary) == summary['image_split_fractions']
    with pytest.raises(ValueError):
        t.validate_split_counts(counts, dict(summary, exact_image_split=False))
    counts['train']['images'] = 60
    summary.update(exact_image_split=False, image_split_fractions=dict(train=.75,val=.125,test=.125))
    with pytest.raises(ValueError, match='Expected exact'):
        t.validate_split_counts(counts, summary)
    assert t.validate_split_counts(counts, summary, True)['train'] == .75


def test_training_options():
    args=SimpleNamespace(epochs=50,imgsz=640,batch=32,device='0',workers=4,seed=42,patience=10,no_amp=False,lr0=.0003,weight_decay=.001,warmup_epochs=3,warmup_bias_lr=.01,freeze=0,degrees=10,translate=.05,scale=.1,save_period=2)
    k=t.training_kwargs(args,'data.yaml','run')
    assert (k['lr0'],k['weight_decay'],k['degrees'],k['translate'],k['scale'])==(.0003,.001,10,.05,.1)
    assert all(k[v]==0 for v in ('hsv_h','hsv_s','hsv_v','bgr','fliplr','flipud','mosaic','mixup','erasing','cutmix','copy_paste'))
    assert k['batch']==32 and k['patience']==10


def test_diversity_and_geographic_selection():
    samples={'a':dict(scene='S',objects=[target()]),'b':dict(scene='S',objects=[target()]),'c':dict(scene='N',objects=[])}
    plans={'S':dict(positive_ids=['A']),'N':dict(positive_ids=[]),'U':dict(positive_ids=['B'])}
    report=t.diversity_report({s:samples for s in t.SPLITS},{s:plans for s in t.SPLITS})['train']
    assert report['unique_storms']==1 and report['tile_annotations']==2 and report['unique_storm_observations']==1
    args=SimpleNamespace(match_km=100,dedup_km=100,confidence=None)
    good=t.selection_metrics({'S':[pred()],'N':[pred(scene='N',score=.2)]},samples,plans,args)
    bad=t.selection_metrics({'S':[pred(lon=-70.)]},samples,plans,args)
    assert good['f1']==1 and good['recall_including_uncovered']==.5
    assert t.checkpoint_rank(good)>t.checkpoint_rank(bad)
    args.confidence=.1
    assert t.selection_metrics({'S':[pred()],'N':[pred(scene='N',score=.2)]},samples,plans,args)['fp']==1


def test_checkpoint_selection(tmp_path,monkeypatch):
    weights=tmp_path/'weights'; weights.mkdir()
    for name in ('best.pt','epoch2.pt','last.pt'):
        (weights/name).write_text(name)
    samples={'a':dict(scene='S',objects=[target()])}
    plans={'S':dict(positive_ids=['A'])}
    args=SimpleNamespace(selection='geographic',match_km=100,dedup_km=100,confidence=None)
    def infer_fake(model,*unused):
        return {'S':[pred(lon=-75 if model.endswith('epoch2.pt') else -70)]},{}
    monkeypatch.setattr(t,'infer',infer_fake)
    best,threshold=t.select_checkpoint(str,weights/'best.pt',tmp_path,samples,plans,None,args,tmp_path)
    assert best.read_text()=='epoch2.pt' and threshold<=.9
    assert json.loads((tmp_path/'checkpoint_selection.json').read_text())['split']=='val'


def test_recall_weighted_selection():
    import cyclone_detection_diagnostics as d
    a=dict(tp=7,fp=0,fn=3,recall=.7,threshold=.8)
    b=dict(tp=9,fp=5,fn=1,recall=.9,threshold=.3)
    assert d.operating_rank(a,SimpleNamespace(beta=1))>d.operating_rank(b,SimpleNamespace(beta=1))
    assert d.operating_rank(b,SimpleNamespace(beta=2))>d.operating_rank(a,SimpleNamespace(beta=2))
    assert d.fbeta(0,0,0) is None


def test_cluster_ci_and_binary_metrics():
    import cyclone_detection_diagnostics as d
    counts={'A':dict(matched=2,observations=4),'B':dict(matched=3,observations=3)}
    assert d.cluster_ci(counts,100,42)==d.cluster_ci(counts,100,42)
    assert d.cluster_ci({'A':counts['A']}) is None
    b=d.binary_extra(t.binary_counts([(True,True),(False,False)]))
    assert b['mcc']==1 and b['balanced_accuracy']==1


def test_training_plots(tmp_path):
    import cyclone_detection_diagnostics as d
    p=tmp_path/'results.csv'
    p.write_text('epoch,train/box_loss,val/box_loss,metrics/mAP50(B),lr/pg0\n1,2,3,.5,.001\n2,1,4,.4,.0005\n')
    d.training_plots(tmp_path,p)
    assert (tmp_path/'training_diagnostics'/'box_loss.png').is_file()
    assert (tmp_path/'training_diagnostics'/'validation_metrics.png').is_file()


def test_precision_constraint():
    args=SimpleNamespace(beta=2,min_precision=.85)
    good=dict(tp=7,fp=1,fn=3,precision=.875,recall=.7,threshold=.6)
    bad=dict(tp=10,fp=5,fn=0,precision=2/3,recall=1.,threshold=.2)
    assert t.choose_operating_point([good,bad],args)==good
    assert not t.precision_ok(t.choose_operating_point([bad],args),args)


def test_mining_and_sampling(tmp_path):
    import cyclone_hard_negatives as h
    samples={str(i):dict(split='train',objects=[target()] if i<3 else []) for i in range(10)}
    scores={k:.9 if k=='3' else .1 for k in samples}
    mined=h.candidates(samples,scores,.3)
    assert mined==[('3',.9)]
    p=tmp_path/'review.csv'; p.write_text('image,approved\n3,yes\n')
    hard=h.approved(p,mined)
    selected=h.sample_list(samples,hard,42,True,.30)
    assert len(selected)==10 and selected.count('3')==3
    assert selected==h.sample_list(samples,hard,42,True,.30)
    samples['0']['split']='val'
    with pytest.raises(ValueError): h.candidates(samples,scores,.3)


def test_cached_mining_and_approval_errors(tmp_path):
    import cyclone_hard_negatives as h
    folder=tmp_path/'mining'; folder.mkdir(); ck=tmp_path/'model.pt'; ck.write_bytes(b'weights')
    samples={'a':dict(split='train',objects=[],scene='S'),'b':dict(split='train',objects=[],scene='S')}
    (tmp_path/'run_config.json').write_text(json.dumps(dict(dataset_fingerprint='abc',arguments=dict(mine_confidence=.3,imgsz=640,keypoint_conf=.25))))
    (folder/'summary.json').write_text(json.dumps(dict(checkpoint=str(ck),negative_tiles=2,candidates=1)))
    (folder/'tile_predictions.jsonl').write_text(json.dumps(pred(image='a'))+'\n')
    c=folder/'candidates.csv'; c.write_text('image,score,approved\na,0.9,\n')
    with pytest.raises(ValueError,match='weight hash'): h.load_cache(folder,samples,'abc',ck,.3,640,.25)
    mined,raw=h.load_cache(folder,samples,'abc',ck,.3,640,.25,True)
    assert mined==[('a',.9)] and len(raw['a'])==1
    with pytest.raises(ValueError,match='No approved'): h.approved(c,mined)
    c.write_text('image,score,approved\na,0.9,yes\n'); assert h.approved(c,mined)=={'a'}
    with pytest.raises(ValueError,match='fingerprint'): h.load_cache(folder,samples,'changed',ck,.3,640,.25,True)
    with pytest.raises(ValueError,match='mine-confidence'): h.load_cache(folder,samples,'abc',ck,.35,640,.25,True)
    (folder/'cache_identity.json').write_text(json.dumps(dict(checkpoint_sha256=h.sha256(ck),predictions_sha256=h.sha256(folder/'tile_predictions.jsonl'))))
    assert h.load_cache(folder,samples,'abc',ck,.3,640,.25)[0]==mined
    ck.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'): h.load_cache(folder,samples,'abc',ck,.3,640,.25)


def test_screen_and_cap():
    import cyclone_hard_negatives as h
    samples={str(i):dict(split='train',objects=[target()] if i<30 else [],scene='S') for i in range(100)}
    mined=[('30',.9),('31',.8),('32',.7)]
    raw={'30':[pred(image='30',lon=-75)],'31':[pred(image='31',lon=-55)],'32':[pred(image='32',valid=False)]}
    rows=h.screen(mined,raw,samples,{'S':{'objects':[target()]}},.3)
    assert [r['screen_status'] for r in rows]==['near_catalogued_storm','clear_of_catalogued_centres','invalid_keypoint']
    selected=h.sample_list(samples,{'31'},42,True,.30,3,{'30','32'})
    assert len(selected)==100 and selected.count('31')==3 and '30' not in selected and '32' not in selected


def test_baseline_guard(tmp_path,monkeypatch):
    weights=tmp_path/'weights'; weights.mkdir()
    baseline=tmp_path/'prior.pt'; baseline.write_text('prior')
    best=weights/'best.pt'; best.write_text('new')
    samples={'a':dict(scene='S',objects=[target()])}
    plans={'S':dict(positive_ids=['A'])}
    args=SimpleNamespace(selection='geographic',match_km=100,dedup_km=100,confidence=None,baseline=baseline)
    def fake(model,*unused):
        return {'S':[pred(lon=-75 if model.endswith('prior.pt') else -70)]},{}
    monkeypatch.setattr(t,'infer',fake)
    chosen,_=t.select_checkpoint(str,best,tmp_path,samples,plans,None,args,tmp_path)
    report=json.loads((tmp_path/'checkpoint_selection.json').read_text())
    assert chosen.read_text()=='prior' and baseline.read_text()=='prior' and report['retained_baseline']
    assert report['f1_delta']==0
    monkeypatch.setattr(t,'infer',lambda model,*unused: ({'S':[pred(lon=-75 if model.endswith('best.pt') else -70)]},{}))
    chosen,_=t.select_checkpoint(str,best,tmp_path,samples,plans,None,args,tmp_path)
    assert chosen.read_text()=='new'
    assert not json.loads((tmp_path/'checkpoint_selection.json').read_text())['retained_baseline']


def test_filtered_box_metrics():
    samples={'a':dict(scene='S',objects=[dict(values=[.5,.5,.5,.5,.5,.5])]),'b':dict(scene='S',objects=[])}
    a=dict(image='a',score=.9,box_xyxy=[160,160,480,480])
    duplicate=dict(a,score=.01)
    false=dict(a,image='b',score=.6)
    raw={'S':[a,duplicate,false]}
    low,_=t.filtered_box_counts(raw,samples,.001,640)
    high,rows=t.filtered_box_counts(raw,samples,.3,640)
    assert (low['tp'],low['fp'],low['fn'])==(1,2,0)
    assert (high['tp'],high['fp'],high['fn'])==(1,1,0)
    assert rows[1]['fp']==1


def test_same_tile_dedup_optional():
    raw=[pred(),pred(index=1,lon=-75.1,score=.8),pred(index=2,lon=-76.,score=.7)]
    assert len(t.deduplicate(raw,100))==3
    assert len(t.deduplicate(raw,100,50))==2


def test_preserve_all_positives():
    import cyclone_hard_negatives as h
    samples={str(i):dict(split='train',objects=[target('A' if i<25 else 'B')] if i<40 else []) for i in range(100)}
    selected=h.sample_list(samples,{'40','41'},42,True,.15,3)
    assert len(selected)==100 and all(str(i) in selected for i in range(40))
    assert selected.count('40')<=3 and selected.count('41')<=3
