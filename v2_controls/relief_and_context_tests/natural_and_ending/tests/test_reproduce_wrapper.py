"""The portable wrapper must not qualify already-qualified intervals twice."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch


def test_preserve_analyzer_output():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('portable_natural_reproduce', root / 'reproduce.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {}
    for name in ('rates', 'contrasts'):
        rows = json.loads((root / 'results/analysis_v2' / (name + '.json')).read_text())
        expected[name] = json.dumps(rows).encode()
    calls = []
    def analyze(rows, output, repetitions, seed, sources):
        calls.append((repetitions, seed, sources))
        output.mkdir()
        for name, data in expected.items():
            (output / (name + '.json')).write_bytes(data)
    fake = types.SimpleNamespace(load_raw=lambda _: ([], ['saved-source']), analyze=analyze)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / 'new'
        with patch.object(module, 'identities', return_value=0), patch.dict(sys.modules, {'analyze_natural': fake}), patch.object(sys, 'argv', ['reproduce.py', '--data-dir', tmp, '--output', str(output)]):
            module.main()
        assert calls == [(10000, 42, ['saved-source'])]
        for name, data in expected.items():
            assert (output / (name + '.json')).read_bytes() == data
