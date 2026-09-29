"""CPU-only, source-faithful cosine reconstruction; no behavioral-vector writes."""
import argparse
import ast
from collections import Counter
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import resource
import shutil
import sys
import time
import traceback
import warnings
import zipfile

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['S1_pain', 'S2_pain', 'Fear', 'NegEmotion', 'NegWorld', 'BodySens', 'Arousal', 'Random', 'Numb', 'Sadness']
PAIN = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E']
SETS = ['S1_1P', 'S2_1P', 'Arousal_1P', 'Random_1P']
NAMES = ['paper_available', 'common_neutral', 'common_controls']
SOURCE = ROOT / 'references/author/02_build_control_vectors.py'
# Compile only the named pure functions; importing the full author script would
# execute its path setup. Their AST bodies and all arithmetic remain unchanged.
fn_names = {'clean_mean', 'denoise_basis', 'project_out', 'compute_pain_vector'}
tree = ast.parse(SOURCE.read_text())
ns = dict(np=np, PCA=PCA, DENOISE_VARIANCE=0.5, PAIN_CATEGORIES=PAIN, CONTROL_CATEGORIES=CONTROL)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in fn_names], type_ignores=[]), str(SOURCE), 'exec'), ns)
clean_mean, denoise_basis, project_out, compute_pain_vector = [ns[k] for k in ['clean_mean', 'denoise_basis', 'project_out', 'compute_pain_vector']]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def csvout(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def matrix_write(path, mat):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow([''] + ORDER)
        for name, row in zip(ORDER, mat):
            w.writerow([name] + [float(x) if np.isfinite(x) else '' for x in row])


def matrix_read(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0][1:] == ORDER and [r[0] for r in rows[1:]] == ORDER
    return np.array([[float(v) if v else np.nan for v in row[1:]] for row in rows[1:]])


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or not (np.isfinite(na) and np.isfinite(nb)):
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def matrix(vectors):
    m = np.full((10, 10), np.nan)
    for i, a in enumerate(ORDER[:8]):
        for j, b in enumerate(ORDER[:8]):
            m[i, j] = cosine(vectors[a], vectors[b])
    return m


def check_matrix(m):
    assert m.shape == (10, 10)
    assert np.allclose(m, m.T, atol=1e-7, equal_nan=True)
    assert np.isnan(m[8:, :]).all() and np.isnan(m[:, 8:]).all()
    assert np.all(np.abs(m[np.isfinite(m)]) <= 1 + 2e-6)
    d = np.diag(m)
    assert np.allclose(d[np.isfinite(d)], 1, atol=2e-6)


def refs():
    result = {}
    for p in sorted((ROOT / 'references/raw_matrices').glob('similarity_*_L*.csv')):
        match = re.fullmatch(r'similarity_(.+)_L(\d+)\.csv', p.name)
        assert match
        model, layer = match.groups()
        result[model] = (int(layer), p)
    assert len(result) == 25
    return result


def structural(x, depth=0):
    if isinstance(x, torch.Tensor):
        return {'shape': list(x.shape), 'dtype': str(x.dtype)}
    if isinstance(x, dict) and depth < 4:
        return {str(k): structural(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return {'type': type(x).__name__, 'length': len(x)}
    return str(x)[:160]


def load(path, model, expected_layer):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    assert isinstance(obj, dict), structural(obj)
    # Only documented conventional wrappers are accepted. A recorded layer is
    # mandatory; it is never inferred from the reference or silently changed.
    layer_values = [obj[k] for k in ('layer', 'extraction_layer') if k in obj]
    assert layer_values, ('No recorded layer', structural(obj))
    assert all(int(v) == expected_layer for v in layer_values), ('layer mismatch', layer_values, expected_layer)
    for key in ('model_name', 'model'):
        if key in obj:
            assert obj[key] == model, ('model mismatch', obj[key], model)
    acts = obj.get('activations', obj.get('final_token'))
    assert isinstance(acts, dict), structural(obj)
    if 'final_token' in acts:
        acts = acts['final_token']
    meta = obj.get('metadata')
    assert isinstance(meta, dict), structural(obj)
    arrays, cats, reviews = {}, {}, []
    dims = set()
    for ds, value in acts.items():
        if isinstance(value, dict):
            assert expected_layer in value or str(expected_layer) in value, (ds, list(value))
            value = value.get(expected_layer, value.get(str(expected_layer)))
        assert isinstance(value, torch.Tensor) and value.ndim == 2, (ds, structural(value))
        assert value.dtype == torch.float16, (ds, str(value.dtype))
        n, d = value.shape
        assert n > 0 and d > 0
        dims.add(d)
        assert ds in meta and isinstance(meta[ds], dict)
        for k, v in meta[ds].items():
            if isinstance(v, (list, tuple)):
                assert len(v) == n, (ds, k, len(v), n)
        a = value.float().numpy()
        categories = meta[ds].get('categories')
        if ds in ('S1_1P', 'S2_1P'):
            assert categories is not None and len(categories) == n
            assert set(PAIN + CONTROL).issubset(categories), (ds, Counter(categories))
        if categories is not None:
            assert len(categories) == n
        ex = {k: v[:2] for k, v in meta[ds].items() if isinstance(v, (list, tuple))}
        reviews.append({'model': model, 'layer': expected_layer, 'set': ds, 'rows': n, 'width': d,
                        'dtype': str(value.dtype), 'category_counts': dict(Counter(categories or [])),
                        'nan': int(np.isnan(a).sum()), 'posinf': int(np.isposinf(a).sum()),
                        'neginf': int(np.isneginf(a).sum()), 'used': ds in SETS,
                        'metadata_sha256': hashlib.sha256(json.dumps(meta[ds], sort_keys=True).encode()).hexdigest(),
                        'examples': ex})
        if ds in SETS:
            arrays[ds] = a
            cats[ds] = categories
    assert len(dims) == 1 and all(ds in arrays for ds in SETS)
    assert not {'ControlSupplement_1P', 'Numb_1P', 'SD_sadness_1P'}.intersection(acts), 'Unexpected supplementary inputs require scope review'
    return arrays, cats, reviews, structural(obj)


def means_and_clouds(arr, cats):
    def rows(ds, labels):
        return arr[ds][np.isin(cats[ds], labels)]
    means = {key: np.nanmean(rows(ds, PAIN), axis=0) for key, ds in [('S1_pain', 'S1_1P'), ('S2_pain', 'S2_1P')]}
    for key, cat in [('Fear', 'B'), ('NegEmotion', 'C1'), ('NegWorld', 'C2'), ('BodySens', 'E')]:
        means[key] = clean_mean(np.concatenate([rows(ds, [cat]) for ds in SETS[:2]]))
    means['Arousal'] = clean_mean(arr['Arousal_1P'])
    means['Random'] = clean_mean(arr['Random_1P'])
    neutral = np.concatenate([rows(ds, ['D']) for ds in SETS[:2]])
    controls = np.concatenate([rows(ds, CONTROL) for ds in SETS[:2]])
    return means, neutral, controls


def construct(arr, cats, construction, model, layer):
    means, neutral, controls = means_and_clouds(arr, cats)
    cloud = neutral if construction != 'common_controls' else controls
    baseline = clean_mean(cloud)
    basis = denoise_basis(cloud, baseline)
    vectors = {k: project_out(np.nan_to_num(v - baseline, nan=0., posinf=0., neginf=0.), basis) for k, v in means.items()}
    dims = [{'model': model, 'layer': layer, 'construction': construction, 'cloud': 'pooled_neutral' if construction != 'common_controls' else 'pooled_controls', 'rows': len(cloud), 'retained_components': len(basis)}]
    if construction == 'paper_available':
        for key, ds in [('S1_pain', 'S1_1P'), ('S2_pain', 'S2_1P')]:
            vectors[key] = compute_pain_vector(arr[ds], cats[ds])
            c = arr[ds][np.isin(cats[ds], CONTROL)]
            pca = PCA().fit(c - np.nanmean(c, axis=0))
            n = min(int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .5)) + 1, len(pca.components_))
            dims.append({'model': model, 'layer': layer, 'construction': construction, 'cloud': ds + '_controls', 'rows': len(c), 'retained_components': n})
    mat = matrix(vectors)
    check_matrix(mat)
    return mat, dims, {k: float(np.linalg.norm(v)) for k, v in vectors.items()}


def aggregate(mats):
    stack = np.stack(mats)
    n = np.isfinite(stack).sum(0)
    mean = np.divide(np.nansum(stack, axis=0), n, out=np.full((10, 10), np.nan), where=n > 0)
    return mean, n


def progress(i, total, phase):
    print(json.dumps({'phase': phase, 'step': i, 'total_steps': total}), flush=True)
    try:
        from silico.slurm_telemetry import report_progress
        report_progress(step=i, total_steps=total, phase=phase)
    except ImportError:
        pass


def preflight(out):
    start = time.monotonic()
    rng = np.random.default_rng(42)
    a = rng.normal(size=(40, 16)).astype(np.float16)
    path = out / 'toy.pt'
    torch.save({'layer': 2, 'activations': {'S1_1P': torch.tensor(a)}}, path)
    restored = torch.load(path, map_location='cpu', weights_only=True)['activations']['S1_1P'].float().numpy()
    assert np.array_equal(restored, a.astype(np.float32))
    cats = (PAIN + CONTROL) * 4
    v = compute_pain_vector(restored, cats)
    assert v.dtype == np.float32 and np.isfinite(v).all()
    assert abs(cosine(v, v) - 1) < 1e-6
    assert np.isnan(cosine(np.zeros(2), np.ones(2)))
    b = denoise_basis(restored, clean_mean(restored))
    q = project_out(v, b)
    assert np.max(np.abs(b @ q)) < 1e-5
    assert np.allclose(clean_mean(np.array([[1, np.inf], [3, 4]], dtype=np.float32)), [2, 4])
    assert len(refs()) == 25
    r, _ = aggregate([matrix_read(p) for _, p in refs().values()])
    assert abs(r[0, 1] - .6098) < 1e-4
    result = {'status': 'passed', 'checks': ['safe tensor roundtrip and fp16-to-fp32', 'source pain-vector finite float32', 'self cosine', 'zero norm missing', 'sequential projection', 'infinity-aware clean mean', '25 raw references', 'released equal-model S1-S2 anchor'], 'expected_S1_S2': .6098, 'observed_released_mean': float(r[0, 1]), 'seconds': time.monotonic() - start, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'environment': environment(), 'source_sha256': sha(SOURCE)}
    dump(out / 'preflight.json', result)
    path.unlink()
    print(json.dumps(result), flush=True)


def environment():
    return {'python': sys.version, 'platform': platform.platform(), 'packages': {k: importlib.metadata.version(k) for k in ['torch', 'numpy', 'scipy', 'scikit-learn']}, 'threads': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}, 'seed': 42}


def run(archive, out):
    start = time.monotonic()
    np.random.seed(42)
    reference = refs()
    inp = out / 'inputs'
    inp.mkdir()
    assert archive.stat().st_size == 421518920, 'archive size mismatch'
    shutil.copy2(archive, inp / archive.name)
    manifest = {'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size, 'source_revision': '8d1649c03a63a39c9aa092532c376800cc4a3863', 'source_sha256': sha(SOURCE), 'environment': environment(), 'reference_hashes': {p.name: sha(p) for _, p in reference.values()}, 'members': []}
    paths = {}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'ZIP CRC failure'
        files = [info for info in z.infolist() if not info.is_dir()]
        assert len(files) == 25, len(files)
        for info in files:
            p = PurePosixPath(info.filename)
            assert not p.is_absolute() and '..' not in p.parts and len(p.parts) == 2 and p.parts[0] == 'pain_axis_layer_slices'
            assert not ((info.external_attr >> 16) & 0o170000) == 0o120000, 'symlink forbidden'
            suffix = '_final_token_extraction_layer.pt'
            assert p.name.endswith(suffix)
            model = p.name[:-len(suffix)]
            assert model in reference and model not in paths, model
            z.extract(info, inp)
            path = inp / info.filename
            paths[model] = path
            manifest['members'].append({'model': model, 'member': info.filename, 'bytes': info.file_size, 'sha256': sha(path), 'expected_layer': reference[model][0]})
    assert set(paths) == set(reference)
    dump(out / 'input_manifest.json', manifest)
    failures, review, dimensions, norm_rows, structures = [], [], [], [], {}
    results = {name: {} for name in NAMES}
    usable = []
    for i, (model, path) in enumerate(sorted(paths.items()), 1):
        layer = reference[model][0]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                arr, cats, audit, structure = load(path, model, layer)
                structures[model] = structure
                m, dims, norms = construct(arr, cats, 'paper_available', model, layer)
            review.extend(audit)
            dimensions.extend(dims)
            norm_rows.append({'model': model, 'construction': 'paper_available', 'norms': norms, 'warnings': [str(w.message) for w in caught]})
            results['paper_available'][model] = m
            matrix_write(out / 'matrices/paper_available' / f'{model}_L{layer}.csv', m)
            usable.append(model)
            del arr, cats
        except Exception as exc:
            failures.append({'model': model, 'phase': 'paper_available', 'error': repr(exc), 'traceback': traceback.format_exc()})
        progress(i, 25, 'input_and_reproduction')
        gc.collect()
    dump(out / 'input_review.json', review)
    dump(out / 'input_structures.json', structures)
    dump(out / 'failure_manifest.json', failures)
    dump(out / 'vector_norms.json', norm_rows)
    discrepancies = []
    for i in range(8):
        for j in range(i + 1, 8):
            models = [m for m in usable if np.isfinite(results['paper_available'][m][i, j]) and np.isfinite(matrix_read(reference[m][1])[i, j])]
            observed = float(np.mean([results['paper_available'][m][i, j] for m in models])) if models else None
            expected = float(np.mean([matrix_read(reference[m][1])[i, j] for m in models])) if models else None
            delta = observed - expected if models else None
            discrepancies.append({'direction_a': ORDER[i], 'direction_b': ORDER[j], 'n_models': len(models), 'observed': observed, 'reference': expected, 'delta': delta, 'flag_gt_003': delta is not None and abs(delta) > .03, 'involves_neutral_pool': not (i == 0 and j == 1), 'models': '|'.join(models)})
    csvout(out / 'reproduction_discrepancies.csv', discrepancies)
    anchor = discrepancies[0]
    passed = bool(anchor['n_models'] and abs(anchor['delta']) <= .01)
    reproduction = {'passed': passed, 'stopping_rule': 'abs matched equal-model S1-S2 mean difference <= 0.01', 'anchor': anchor, 'usable_models': usable, 'failed_files': len(failures), 'n_discrepancies_gt_003': sum(r['flag_gt_003'] for r in discrepancies), 'all_flagged_involve_neutral': all(r['involves_neutral_pool'] for r in discrepancies if r['flag_gt_003']), 'missing': ['ControlSupplement_1P', 'Numb_1P', 'SD_sadness_1P'], 'seconds': time.monotonic() - start}
    dump(out / 'reproduction.json', reproduction)
    print('REPRODUCTION ' + json.dumps(reproduction), flush=True)
    # No alternative construction has been evaluated before this stopping check.
    if passed:
        for i, model in enumerate(usable, 1):
            layer = reference[model][0]
            arr, cats, _, _ = load(paths[model], model, layer)
            for construction in NAMES[1:]:
                try:
                    m, dims, norms = construct(arr, cats, construction, model, layer)
                    results[construction][model] = m
                    dimensions.extend(dims)
                    norm_rows.append({'model': model, 'construction': construction, 'norms': norms, 'warnings': []})
                    matrix_write(out / 'matrices' / construction / f'{model}_L{layer}.csv', m)
                except Exception as exc:
                    failures.append({'model': model, 'phase': construction, 'error': repr(exc), 'traceback': traceback.format_exc()})
            del arr, cats
            gc.collect()
            progress(i, len(usable), 'symmetric_comparisons')
    csvout(out / 'pca_dimensions.csv', dimensions, ['model', 'layer', 'construction', 'cloud', 'rows', 'retained_components'])
    dump(out / 'failure_manifest.json', failures)
    dump(out / 'vector_norms.json', norm_rows)
    comparison, cell_rows, paired = [], [], []
    plot = {'directions': ORDER, 'constructions': {}}
    for construction, models in results.items():
        if not models:
            continue
        avg, counts = aggregate(list(models.values()))
        check_matrix(avg)
        matrix_write(out / f'mean_{construction}.csv', avg)
        matrix_write(out / f'counts_{construction}.csv', counts)
        plot['constructions'][construction] = {'mean': [[float(v) if np.isfinite(v) else None for v in row] for row in avg], 'counts': counts.tolist(), 'models': list(models)}
        for i in range(10):
            for j in range(10):
                cell_rows.append({'construction': construction, 'direction_a': ORDER[i], 'direction_b': ORDER[j], 'mean': float(avg[i, j]) if np.isfinite(avg[i, j]) else '', 'n_models': int(counts[i, j])})
        for i, j in [(1, 2), (1, 3), (2, 3), (0, 1)]:
            values = [float(m[i, j]) for m in models.values() if np.isfinite(m[i, j])]
            comparison.append({'construction': construction, 'direction_a': ORDER[i], 'direction_b': ORDER[j], 'n_models': len(values), 'mean': float(np.mean(values)) if values else '', 'minimum': min(values) if values else '', 'median': float(np.median(values)) if values else '', 'maximum': max(values) if values else ''})
    if passed:
        for i in range(8):
            for j in range(i + 1, 8):
                common = [m for m in usable if all(m in results[c] and np.isfinite(results[c][m][i, j]) for c in NAMES)]
                row = {'direction_a': ORDER[i], 'direction_b': ORDER[j], 'n_common': len(common), 'models': '|'.join(common)}
                for c in NAMES:
                    row[c] = float(np.mean([results[c][m][i, j] for m in common])) if common else ''
                paired.append(row)
        csvout(out / 'common_model_comparison.csv', paired)
    csvout(out / 'comparison.csv', comparison, ['construction', 'direction_a', 'direction_b', 'n_models', 'mean', 'minimum', 'median', 'maximum'])
    csvout(out / 'mean_cells.csv', cell_rows, ['construction', 'direction_a', 'direction_b', 'mean', 'n_models'])
    dump(out / 'plot_data.json', plot)
    checks = []
    for construction, models in results.items():
        for model, original in models.items():
            restored = matrix_read(out / 'matrices' / construction / f'{model}_L{reference[model][0]}.csv')
            check_matrix(restored)
            assert np.array_equal(original, restored, equal_nan=True)
        if models:
            recomputed, n = aggregate([matrix_read(out / 'matrices' / construction / f'{m}_L{reference[m][0]}.csv') for m in models])
            assert np.array_equal(recomputed, matrix_read(out / f'mean_{construction}.csv'), equal_nan=True)
            assert np.array_equal(n, matrix_read(out / f'counts_{construction}.csv'))
            checks.append({'construction': construction, 'n_models': len(models), 'csv_roundtrip': True, 'symmetry_bounds_defined_diagonals_missingness': True, 'aggregate_recomputed_from_csv': True, 'min_available_cell_count': int(n[:8, :8].min()), 'max_available_cell_count': int(n[:8, :8].max())})
    summary = {'reproduction_passed': passed, 'checks': checks, 'failed_files_or_constructions': len(failures), 'runtime_seconds': time.monotonic() - start, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'comparison': comparison, 'nonfinite_elements': {k: sum(r[k] for r in review) for k in ['nan', 'posinf', 'neginf']}, 'notes': ['No held-out split: descriptive geometry of the fixed uploaded rows.', 'Source compute_pain_vector uses sklearn PCA; neutral and symmetric pooled clouds use source NumPy SVD.', 'FP16 stored rows converted to FP32; sequential projection and 50% threshold unchanged.', 'Source pain calculation does not sanitize control cloud before PCA; such failures remain file failures.', 'Other means map infinity to NaN; centered SVD cloud and final differences map nonfinite values to zero.']}
    dump(out / 'verification.json', summary)
    print('COMPLETE ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.preflight:
        preflight(args.out)
    else:
        assert args.archive
        run(args.archive, args.out)
