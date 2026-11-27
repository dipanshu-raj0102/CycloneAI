#!/usr/bin/env python3
"""Finalize an interrupted V2 build IN PLACE from successful journals; no raw reads."""
import argparse
import json
import os
from collections import Counter
from pathlib import Path
import preprocess_detection as p


def load(path):
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('data/processed/detection_pose_v2'))
    ap.add_argument('--allow-approximate-split', action='store_true')
    args = ap.parse_args()
    src = args.source.resolve()
    if not src.is_dir():
        raise FileNotFoundError(src)
    lock = src / 'BUILD.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    try:
        recover(src, args.allow_approximate_split)
    finally:
        lock.unlink(missing_ok=True)


def recover(src, approximate):
    stage = src
    build = load(src/'build.json'); cfg = dict(build['config'])
    cfg['allow_approximate_split'] = approximate
    inventory = load(src/'inventory.json')
    original = load(src/'original_scene_plan.json' if (src/'original_scene_plan.json').exists() else src/'scene_plan.json')
    failures = load(src/'excluded_scene_failures.json' if (src/'excluded_scene_failures.json').exists() else src/'processing_failures.json')
    mapping = load(src/'storm_splits.json')
    if not failures:
        raise ValueError('No recorded failures to recover; check dataset with training --check-only')
    selection = load(src/'image_balance.json') if (src/'image_balance.json').exists() else None
    drop = set(selection['drop_images']) if selection else set()
    if selection and not selection['exact'] and not approximate:
        raise ValueError('Existing approximate selection requires --allow-approximate-split')
    failed = {f['scene'] for f in failures}
    ids = {q['scene']['id'] for q in original}
    if len(ids) != len(original) or not failed or not failed <= ids:
        raise ValueError('Invalid scene plan or failure list')
    plans = [q for q in original if q['scene']['id'] not in failed]
    missing = ids - failed - {f.stem for f in (src/'records').glob('*.json')}
    if missing:
        raise ValueError(f'Unrecorded nonfailed scenes: {sorted(missing)}')
    expected = sum(len(q['positive_ids']) for q in original)
    lost = sum(len(q['positive_ids']) for q in original if q['scene']['id'] in failed)
    counts = {s: Counter() for s in p.SPLITS}; covered = 0
    for q in plans:
        sid = q['scene']['id']; split = q['split']; rec = load(src/'records'/f'{sid}.json')
        if rec['scene'] != sid or rec['split'] != split or any(mapping.get(i) != split for i in q['ids']):
            raise ValueError(f'Inconsistent scene or storm assignment: {sid}')
        present = set()
        for r in rec['samples']:
            if r['scene'] != sid or r['split'] != split:
                raise ValueError(f'Invalid sample in {sid}')
            im, lab = Path(r['image']), Path(r['label'])
            if im.parent != Path('images')/split or lab != Path('labels')/split/(im.stem+'.txt') or not im.name.startswith(sid+'_'):
                raise ValueError(f'Invalid sample path: {im}')
            if r['image'] in drop and r['objects']:
                raise ValueError(f'Balance selection attempts to delete a positive: {im}')
            if r['image'] in drop:
                continue
            if not (src/im).is_file() or not (src/lab).is_file():
                raise FileNotFoundError(im if not (src/im).is_file() else lab)
            present.update(o['sid'] for o in r['objects'])
            counts[split]['positive' if r['objects'] else 'negative'] += 1
        if not present <= set(q['positive_ids']) or set(rec['uncovered_sids']) != set(q['positive_ids'])-present:
            raise ValueError(f'Inconsistent coverage in {sid}')
        covered += len(present)
    coverage = covered / max(expected, 1)
    print('Retained scene counts:', dict(Counter(q['split'] for q in plans)))
    print('Pre-balance image counts:', counts)
    print(f'Coverage including failed scenes: {coverage:.2%}; failed positive observations: {lost}')
    if coverage < cfg['min_storm_coverage']:
        raise ValueError('Original planned storm coverage below configured minimum')
    factors = dict(zip(p.SPLITS,(8,1,1)))
    if any(not counts[s]['positive'] or not counts[s]['negative'] for s in p.SPLITS):
        raise ValueError('Every split must contain positives and negatives')
    low=max((counts[s]['positive']+factors[s])//factors[s] for s in p.SPLITS)
    high=min(sum(counts[s].values())//factors[s] for s in p.SPLITS)
    if high < low and not approximate:
        raise ValueError('Exact split infeasible without deleting positives; inspect counts, or explicitly use --allow-approximate-split')
    # Validate all existing labels/images before any destructive balancing.
    # On resume, the persisted selection may have already removed some files.
    if selection is None:
        p.validate(stage, plans, {**cfg, 'allow_approximate_split': True})
    # Preserve original metadata in this same directory, before changing it.
    if not (stage/'original_inventory.json').exists():
        p.dump(stage/'original_inventory.json', inventory)
    inventory = load(stage/'original_inventory.json')
    p.dump(stage/'original_scene_plan.json',original)
    p.dump(stage/'excluded_scene_failures.json',failures)
    inv=dict(inventory); inv.update(original_planned_scenes=len(original),planned_scenes=len(plans),scene_split_counts=dict(Counter(q['split'] for q in plans)),positive_scene_split_counts=dict(Counter(q['split'] for q in plans if q['positive_ids'])),estimated_positive_tiles_per_split={s:sum(q['estimated_positive_tiles'] for q in plans if q['split']==s) for s in p.SPLITS},positive_observations_after_split=expected-lost,excluded_failed_scenes=len(failed))
    build['config']=cfg; build['recovery']={'source':str(src),'source_signature':build['signature'],'excluded_scenes':sorted(failed),'method':'successful scene journals only'}
    p.balance_images(stage,plans,cfg)
    summary=p.validate(stage,plans,cfg)
    summary.update(original_planned_storm_observations=expected,excluded_failed_positive_observations=lost,coverage_including_failed_scenes=coverage,excluded_failed_scenes=len(failed),positive_observations_before_split=inventory['positive_observations_before_split'],coverage_relative_to_pre_split_plan=summary['covered_storm_observations']/max(inventory['positive_observations_before_split'],1))
    p.dump(stage/'scene_plan.json',plans)
    p.dump(stage/'inventory.json',inv)
    p.dump(stage/'build.json',build)
    p.dump(stage/'summary.json',summary)
    # All retained scenes succeeded; original failures remain separately auditable.
    p.write_previews(stage)
    p.atomic_text(stage/'data.yaml',f'path: {json.dumps(str(src))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cyclone\nkpt_shape: [1, 3]\nflip_idx: [0]\n')
    p.dump(stage/'processing_failures.json',[])
    print(json.dumps(summary['splits'],indent=2)); print('Recovered in place:',src)

if __name__=='__main__':
    main()
