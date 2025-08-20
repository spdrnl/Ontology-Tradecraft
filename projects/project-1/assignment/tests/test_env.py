import json, subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

def run():
    out = subprocess.check_output([sys.executable, "src/check_env.py"], cwd=ROOT)
    return json.loads(out.decode())

def test_rdflib_and_parse():
    data = run()
    assert int(data["triple_count"]) == 2  # sample.ttl has 2 triples

def test_python_version():
    data = run()
    major = int(data["python"].split(".")[0])
    assert major >= 3
