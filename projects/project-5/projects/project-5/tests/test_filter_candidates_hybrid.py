import json
from pathlib import Path

from rdflib import Graph, RDFS
from rdflib.term import URIRef


def _write_candidates(path: Path, pairs):
    g = Graph()
    for s, o in pairs:
        g.add((URIRef(s), RDFS.subClassOf, URIRef(o)))
    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=path.as_posix(), format="turtle")


def _write_metrics(path: Path, selected_tau=0.7):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "selected_tau": selected_tau,
        # minimal extras (not used by the test due to mocks)
        "train_path": "/dev/null",
        "valid_path": "/dev/null",
        "dim": 2,
        "lr": 1e-3,
        "margin": 1.0,
        "batch": 32,
        "reg_norm": 1.0,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_hybrid_filter_accepts_expected_axioms(tmp_path, monkeypatch):
    # Import the script as a module
    import sys
    # tests/ -> project-5/ -> scripts/
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, scripts_dir.as_posix())
    import filter_candidates_hybrid as mod  # type: ignore

    # Create temporary input files
    candidates_path = tmp_path / "generated" / "candidate_el.ttl"
    out_path = tmp_path / "generated" / "accepted_el.ttl"
    metrics_path = tmp_path / "reports" / "mowl_metrics.json"

    pairs = [
        ("http://e.org/A1", "http://e.org/B1"),  # will be accepted
        ("http://e.org/A2", "http://e.org/B2"),  # will be rejected
    ]
    _write_candidates(candidates_path, pairs)
    _write_metrics(metrics_path, selected_tau=0.7)

    # Mock out heavy parts: model init, cosine scoring, LLM scoring
    def fake_init_model_for_embeddings(_metrics):
        return object(), None  # model placeholder, dataset not used

    def fake_cosine_scores(_model, p_iter):
        mapping = {
            pairs[0]: 0.9,  # high cosine
            pairs[1]: 0.4,  # low cosine
        }
        return {p: mapping.get(p, 0.0) for p in p_iter}

    def fake_plausibility_scores(p_iter, _cfg, _defs, dry_run=False):
        mapping = {
            pairs[0]: 0.5,  # neutral plausibility
            pairs[1]: 0.2,  # low plausibility
        }
        return {p: mapping.get(p, 0.5) for p in p_iter}

    monkeypatch.setattr(mod, "_init_model_for_embeddings", fake_init_model_for_embeddings)
    monkeypatch.setattr(mod, "_cosine_scores", fake_cosine_scores)
    monkeypatch.setattr(mod, "_score_plausibility", fake_plausibility_scores)

    # Run the script's main with our file paths
    argv = [
        "filter_candidates_hybrid.py",
        "--candidates",
        candidates_path.as_posix(),
        "--out",
        out_path.as_posix(),
        "--metrics",
        metrics_path.as_posix(),
        "--w-cos",
        "0.7",
        "--dry-run",  # still mocked but keep flag to test code path
    ]
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    monkeypatch.setattr(sys, "argv", argv)

    mod.main()

    # Verify outputs
    assert out_path.exists(), "accepted_el.ttl should be created"
    # accepted should include only the first pair (0.7*0.9 + 0.3*0.5 = 0.78 >= 0.7)
    g = Graph()
    g.parse(out_path.as_posix())
    triples = set((str(s), str(o)) for s, _, o in g.triples((None, RDFS.subClassOf, None)))
    assert (pairs[0][0], pairs[0][1]) in triples
    assert (pairs[1][0], pairs[1][1]) not in triples

    # Sidecar JSON report should exist and be parseable
    sidecar = Path(out_path.as_posix() + ".json")
    assert sidecar.exists(), "sidecar JSON report should be written"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data.get("num_candidates") == 2
    assert data.get("num_accepted") == 1
