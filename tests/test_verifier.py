"""Tests of the standalone FLINT verifier, including tampered certificates that must be rejected."""

import copy
import importlib.util
import json
from pathlib import Path

import flint
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_certificate", ROOT / "verify" / "verify_certificate.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

CERTS = sorted((ROOT / "data" / "certificates").glob("*_round.json"))


def load(path):
    return json.loads(Path(path).read_text())


def run(cert):
    return V.verify(cert, V.Log(quiet=True))


def test_vca_root_counts():
    P = flint.fmpz_poly
    assert V.vca_roots_01(P([2, -9, 9]) * P([-2, 1])) == 2  # (3t - 1)(3t - 2)(t - 2)
    assert V.vca_roots_01(P([1, 1]) * P([-3, 1])) == 0
    assert V.vca_roots_01(P([-1, 2]) * P([-5, 1])) == 1  # root exactly at t = 1/2
    assert V.arb_roots_01(P([2, -9, 9]) * P([-2, 1])) == 2


def test_interpolate():
    vals = [x**3 - 2 * x + 5 for x in range(6)]
    assert V.interpolate(vals) == flint.fmpz_poly([5, -2, 0, 1])


@pytest.mark.parametrize("path", CERTS, ids=lambda p: p.name)
def test_certificates_verify(path):
    cert = load(path)
    assert run(cert) == cert["proven_ovals"]


@pytest.fixture
def cert():
    return load(ROOT / "data" / "certificates" / "cache_H_deg5_20261004-141957_4_round.json")


def test_rejects_wrong_f(cert):
    cert["f"][5] += 1
    assert run(cert) is None


def test_rejects_polygon_through_oval(cert):
    k = 0
    cert["lower"]["paths"][k]["polygon"][0] = cert["witnesses"][1]["point"]  # a vertex inside an oval
    assert run(cert) is None


def test_polygons_needed(cert):
    # each pair needs only one of its two polygons: 8 of the 9 polygons suffice, 7 do not
    del cert["lower"]["paths"][3]
    assert run(cert) == 9
    del cert["lower"]["paths"][3]
    assert run(cert) is None


def test_rejects_contractible_loop(cert):
    loop = cert["upper"]["loop"]
    loop[-1] = list(loop[0])  # closed loop: lift returns to O
    assert run(cert) is None


def test_rejects_base_point_mismatch(cert):
    bad = copy.deepcopy(cert)
    w = bad["witnesses"][1]["point"]  # a point inside an oval
    for r in range(3):
        bad["upper"]["M"][r][1] = w[r]
    assert run(bad) is None
