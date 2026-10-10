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


TYPE_CERTS = sorted((ROOT / "data").glob("*_types/certificates/*.json"))


@pytest.mark.parametrize("path", TYPE_CERTS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_type_certificates_verify(path):
    cert = load(path)
    assert run(cert) == cert["proven_ovals"]
    assert cert["_type"] == cert["type"]


def _interior_cert():
    return next(load(p) for p in TYPE_CERTS if "upper_interior" in load(p))


def test_rejects_wrong_type_claim():
    cert = load(next(p for p in TYPE_CERTS if load(p)["type"] == "1<1>"))
    cert["type"] = "2"
    assert run(cert) is None


def test_interior_bound_needed():
    cert = _interior_cert()
    assert run(copy.deepcopy(cert)) == cert["proven_ovals"]
    del cert["upper_interior"]
    assert run(cert) is None  # the plain pencil bound is loose


def test_rejects_bad_interior_line():
    cert = _interior_cert()
    O = [int(V.Fraction(v)) for v in (row[1] for row in cert["upper_interior"]["M"])]
    cert["upper_interior"]["line_point"] = O  # the line O O is not a line
    assert run(cert) is None


def test_depth_three_nest_is_not_one_and_nest():
    """Three nested circles (scheme 1<1<1>>) have the region signs of 1 u 1<1>; the line test must reject it."""
    X, Y, Z = V.CTX3.gens()
    H = (X**2 + Y**2 - Z**2) * (X**2 + Y**2 - 4 * Z**2) * (X**2 + Y**2 - 9 * Z**2)
    terms = V.int_terms(H)
    wit = [[0, 0, 1], [3, 0, 2], [5, 0, 2], [4, 0, 1]]  # radii 0, 1.5, 2.5, 4: one point per region
    wsign = [V.sign(int(H(*w))) for w in wit]
    assert V.verify_type("1 u 1<1>", 3, 6, terms, wit, wsign, [0, 1, 2, 3], wsign[3], V.Log(quiet=True), 0) is None
