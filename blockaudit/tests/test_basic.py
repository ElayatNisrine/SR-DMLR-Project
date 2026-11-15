import numpy as np
from blockaudit import audit_blocks

def test_masking_detected():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(150, 8)); NM = rng.normal(size=(150, 8)); R = rng.normal(size=(150, 8))
    S = M + rng.normal(scale=0.05, size=M.shape)          # bloc sensible copié des membres : fuite forte
    F = lambda n: rng.normal(size=(n, 32))                # blocs étrangers indépendants
    a = audit_blocks(np.c_[S, F(150)], np.c_[M, F(150)], np.c_[NM, F(150)], np.c_[R, F(150)],
                     {"sensitive": slice(0, 8), "foreign": slice(8, 40)}, sensitive=["sensitive"])
    assert a.blocks["sensitive"]["DCR"] > 0.9
    assert a.masking("sensitive", "DCR") > 0.3

def test_no_leak_near_chance():
    rng = np.random.default_rng(1)
    X = lambda n: rng.normal(size=(n, 8))
    a = audit_blocks(X(200), X(150), X(150), X(150), {"b": slice(0, 8)})
    assert abs(a.blocks["b"]["DCR"] - 0.5) < 0.12


def test_piag_reversed_full_auroc():
    from blockaudit.audit import BlockAudit
    # bloc à 0.55, profil complet à 0.40 : l'écart signé (+0.15) suggère un masquage,
    # mais le profil complet s'écarte davantage du hasard -> PIAG négatif
    a = BlockAudit({"c": {"DCR": 0.55}}, {"DCR": 0.40}, 0.0, ["c"])
    assert abs(a.masking("c", "DCR") - 0.15) < 1e-12
    assert abs(a.piag("c", "DCR") - (-0.05)) < 1e-12
    assert a.masked_fraction("c", "DCR") is None

def test_piag_masking():
    from blockaudit.audit import BlockAudit
    a = BlockAudit({"c": {"DOMIAS": 0.66}}, {"DOMIAS": 0.52}, 0.0, ["c"])
    assert abs(a.piag("c", "DOMIAS") - 0.14) < 1e-12
    assert abs(a.masked_fraction("c", "DOMIAS") - 87.5) < 1e-9


def test_report_no_false_flag_reversed_full():
    from blockaudit.audit import BlockAudit
    a = BlockAudit({"c": {"DCR": 0.56}}, {"DCR": 0.38}, 0.0, ["c"])   # PIAG = 0.06 - 0.12 < 0
    assert "masquée par" not in a.report()

def test_report_flags_true_masking():
    from blockaudit.audit import BlockAudit
    a = BlockAudit({"c": {"DOMIAS": 0.66}}, {"DOMIAS": 0.52}, 0.0, ["c"])
    assert "masquée par" in a.report()
