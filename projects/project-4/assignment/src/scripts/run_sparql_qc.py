#!/usr/bin/env python3
"""
Run every *.rq in a folder against a TTL and fail if any returns rows.
"""
import sys, pathlib
from rdflib import Graph

def run_query(g: Graph, qpath: pathlib.Path) -> int:
    q = qpath.read_text(encoding="utf-8")
    res = list(g.query(q))
    if len(res) > 0:
        print(f"❌ QC failed: {qpath.name} -> {len(res)} row(s)")
        # print up to 10 rows for visibility
        for r in res[:10]:
            print("  ", [str(x) for x in r])
        return 1
    print(f"✅ QC passed: {qpath.name}")
    return 0

def main(ttl_path: str, qc_dir: str) -> int:
    ttl = pathlib.Path(ttl_path)
    qc  = pathlib.Path(qc_dir)
    g = Graph(); g.parse(ttl, format="turtle")
    failures = 0
    for qpath in sorted(qc.glob("*.rq")):
        failures += run_query(g, qpath)
    if failures:
        print(f"\nQC summary: {failures} failing query(ies).")
        return 1
    print("\nQC summary: all checks passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
