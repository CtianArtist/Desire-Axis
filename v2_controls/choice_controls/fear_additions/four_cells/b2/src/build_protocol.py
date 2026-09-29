"""Deterministic binding-only adaptation of the hash-pinned successful sampler."""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'inputs/reference/b2_protocol.py'
OUTPUT = ROOT / 'src/run_b2.py'
SOURCE_SHA256 = 'e85ebf2b4065885756d62f5bd20d88e23d499ffb18c160bd80a586a6bec22ca5'


def generate():
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
    text = raw.decode()
    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, (text.count(old), old[:100])
        text = text.replace(old, new)
    end = text.index('"""', 3) + 3
    text = '"""Generated B2 working fear cell. Kernels and callback preserved from pinned source."""' + text[end:]
    replace('from pain_axis_b.verify_vectors import load_verified', 'from pain_axis_b.fear_inputs import load_inputs')
    replace('PROTOCOL = "2btnN names+saltseed v1"', 'PROTOCOL = "four-missing-fear-b2-working-v1"')
    replace('pain_path, sadness_path, output, resume=None):',
            'pain_path, fear_path, output, resume=None, *, verified_inputs, nominal_batch_rows):')
    replace('    assert len(grid) == 15580\n    if sadness_path is None:\n        grid = [g for g in grid if g[2][1] != "sadness"]',
            '    assert len(grid) == 410\n    assert all(g[2] == ("fear_on_button_works", "fear", True, True) for g in grid)')
    replace('    tok, base, model, layers = rt.load_model(REPO, MODEL_NAME, adapter_dir, output)',
            '    tok, base, model, layers = rt.load_model(REPO, MODEL_NAME, adapter_dir, output)\n'
            '    assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if "lora_" in n)')
    replace('    data, identity_receipt = load_verified(pain_path, MODEL_NAME,\n'
            '        json.loads((pain_path.parent/"original_tensor_hashes_v2.json").read_text()),\n'
            '        pain_path.with_suffix(".json"))',
            '    data, fv, identity_receipt = verified_inputs')
    replace('    if sadness_path:\n'
            '        sadness = torch.load(sadness_path, map_location="cpu", weights_only=True)\n'
            '        sv = sadness["sadness_vector_matched_fp32"].float()\n'
            '        assert sv.shape == v.shape and torch.isfinite(sv).all()\n'
            '        assert torch.isclose(sv.norm(), v.norm(), rtol=1e-6, atol=1e-5)\n'
            '        DIR["sadness"] = sv.to("cuda", dtype=torch.bfloat16)',
            '    assert fv.dtype == torch.float32 and fv.shape == v.shape\n'
            '    assert torch.isclose(fv.norm(), v.norm(), rtol=1e-6, atol=1e-5)\n'
            '    DIR["fear"] = fv.to("cuda", dtype=torch.bfloat16)')
    replace('"sadness_sha256": rt.sha256(sadness_path) if sadness_path else None,',
            '"fear_sha256": rt.sha256(fear_path), "input_identity": identity_receipt,')
    replace('"nominal_batch_rows": next(m[-1] for m in MODELS if m[1] == MODEL_NAME),',
            '"nominal_batch_rows": nominal_batch_rows,')
    replace('"requested_trials":15580', '"requested_trials":410')
    # Rebind the CLI only; every nested function in run_model remains byte-identical.
    start = text.index('\ndef main():\n')
    text = text[:start] + '''
def main():
    from pain_axis_b.entry import runner_main
    runner_main(globals())


if __name__ == "__main__":
    main()
'''
    compile(text, str(OUTPUT), 'exec')
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    text = generate()
    if args.check:
        assert OUTPUT.read_text() == text, 'Generated runner is stale'
        print('Generated runner matches anchored source')
    else:
        OUTPUT.write_text(text)


if __name__ == '__main__':
    main()
