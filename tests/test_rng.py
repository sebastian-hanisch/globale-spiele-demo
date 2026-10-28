"""SplitMix64-Kopie: wörtlich wie im Original, bekannte Werte, Block gleich Strom, keine anderen Importe."""

import ast
from pathlib import Path

import numpy as np
import pytest

from glo_rng import SplitMix64, uniform_block

ROOT = Path(__file__).resolve().parent.parent
RNG_SOURCE = ("ab-test-demo", "abt_rng.py")


def test_rng_copy_is_verbatim_apart_from_the_header():
    folder, original = RNG_SOURCE
    source = ROOT.parent / folder / original
    mine = (ROOT / "glo_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """SplitMix64 mit Seed 0: die bekannten ersten Werte der Konvention (Vigna)."""
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_block_equals_the_scalar_stream():
    for seed in (0, 1, 35, 2 ** 63 + 5):
        rng = SplitMix64(seed)
        assert np.array_equal(uniform_block(seed, 100), np.array([rng.uniform() for _ in range(100)]))


def test_rng_module_has_no_other_imports_than_numpy_and_math():
    tree = ast.parse((ROOT / "glo_rng.py").read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert names == {"math", "numpy"}


def test_shifting_the_seed_by_blocks_continues_the_stream():
    """Grundlage der Blockbildung in glo_email.simulate_protocol: Seed + c·γ liefert den Strom ab Position c."""
    import glo_email as EM
    full = uniform_block(123, 30)
    assert np.array_equal(full[10:], uniform_block(EM._block_seed(123, 10), 20))
