#!/usr/bin/env python3
import sys
from pyshacl import validate

def main(data_ttl: str, shapes_ttl: str):
    conforms, report_graph, report_text = validate(
        data_graph=data_ttl,
        shacl_graph=shapes_ttl,
        inference='rdfs',
        abort_on_first=False,
        allow_infos=False,
        allow_warnings=True
    )
    print(report_text)
    if not conforms:
        print("❌ SHACL validation failed.")
        sys.exit(1)
    print("✅ SHACL validation passed.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: shacl_validate.py <data.ttl> <shapes.ttl>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
