#!/usr/bin/env python3
"""
Create train.ttl and valid.ttl splits from a source ontology.

Goal (per MOWL PathDataset and ELEmbeddings examples):
- Use train ontology for learning mainly from simple named-class subclass axioms.
- Use valid ontology to evaluate on held-out named-class rdfs:subClassOf axioms.
- Both splits must preserve entity vocabulary: include all owl:Class and owl:ObjectProperty declarations
  (and optionally DataProperty declarations) in BOTH outputs so that entities exist in both graphs.
- Only partition simple named-class subclass axioms; exclude complex expressions and blank nodes.

This script implements a deterministic split with configurable ratio and seed.

Usage:
  python3 scripts/split_train_valid.py \
      --input src/InformationEntityOntology.ttl \
      --train src/train.ttl \
      --valid src/valid.ttl \
      --ratio 0.8 \
      --seed 42

Notes about MOWL docs:
- In MOWL, PathDataset(train_path, valid_path) expects two ontologies. Typical practice is to keep
  all class declarations in both and to put held-out rdfs:subClassOf axioms in the validation file.
- This script follows that pattern so ELEmbeddings can embed the same entity set across splits and
  evaluate cosine similarity for validation subClassOf pairs.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import random

from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef


def get_logger() -> logging.Logger:
    try:
        from util.logger_config import config as _cfg
        logger = logging.getLogger(__name__)
        _cfg(logger)
        return logger
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)


logger = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split an ontology into train.ttl and valid.ttl for MOWL training")
    p.add_argument("--input", default="src/InformationEntityOntology.ttl", help="Source ontology TTL/OWL path")
    p.add_argument("--train", default="src/train.ttl", help="Output training TTL path")
    p.add_argument("--valid", default="src/valid.ttl", help="Output validation TTL path")
    p.add_argument("--ratio", type=float, default=0.8, help="Proportion of simple subclass axioms to put in train")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling before split")
    return p.parse_args()


def is_iri(node) -> bool:
    # rdflib URIRef behaves like a string with .startswith('http') in many ontologies
    try:
        s = str(node)
        return s.startswith("http://") or s.startswith("https://")
    except Exception:
        return False


def collect_simple_subclass_axioms(g: Graph) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        # Only named class -> named class axioms
        if is_iri(s) and is_iri(o):
            pairs.append((str(s), str(o)))
    return pairs


def copy_prefixes(src: Graph, dst: Graph) -> None:
    # Copy all namespace bindings for nicer TTL output
    for prefix, ns in src.namespace_manager.namespaces():
        try:
            dst.namespace_manager.bind(prefix, Namespace(str(ns)), override=False)
        except Exception:
            # Ignore binding errors, rdflib will still serialize with URIs
            pass


def add_declarations_to(graph: Graph, source: Graph) -> int:
    count = 0
    # Classes
    for c in set(source.subjects(RDF.type, OWL.Class)):
        graph.add((c, RDF.type, OWL.Class))
        count += 1
    # Object properties
    for p in set(source.subjects(RDF.type, OWL.ObjectProperty)):
        graph.add((p, RDF.type, OWL.ObjectProperty))
        count += 1
    # Data properties (optional, but keep for completeness)
    for p in set(source.subjects(RDF.type, OWL.DatatypeProperty)):
        graph.add((p, RDF.type, OWL.DatatypeProperty))
        count += 1
    return count


def main() -> None:
    args = parse_args()
    src_path = Path(args.input)
    train_path = Path(args.train)
    valid_path = Path(args.valid)

    if not src_path.exists():
        raise FileNotFoundError(f"Input ontology not found: {src_path}")

    logger.info("Reading source ontology: %s", src_path)
    g = Graph()
    g.parse(src_path.as_posix())

    # Collect simple subclass axioms
    pairs = collect_simple_subclass_axioms(g)
    logger.info("Found %d simple named-class subclass axioms", len(pairs))

    # Deterministic shuffle and split
    random.seed(args.seed)
    random.shuffle(pairs)
    n_train = int(len(pairs) * args.ratio)
    train_pairs = pairs[:n_train]
    valid_pairs = pairs[n_train:]

    # Build output graphs with copied prefixes and declarations
    g_train = Graph()
    g_valid = Graph()
    copy_prefixes(g, g_train)
    copy_prefixes(g, g_valid)

    decls_added_train = add_declarations_to(g_train, g)
    decls_added_valid = add_declarations_to(g_valid, g)

    for s_iri, o_iri in train_pairs:
        g_train.add((URIRef(s_iri), RDFS.subClassOf, URIRef(o_iri)))
    for s_iri, o_iri in valid_pairs:
        g_valid.add((URIRef(s_iri), RDFS.subClassOf, URIRef(o_iri)))

    # Ensure output directories exist
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)

    g_train.serialize(destination=train_path.as_posix(), format="turtle")
    g_valid.serialize(destination=valid_path.as_posix(), format="turtle")

    logger.info(
        "Wrote train split: %s (decls=%d, axioms=%d)", train_path, decls_added_train, len(train_pairs)
    )
    logger.info(
        "Wrote valid split: %s (decls=%d, axioms=%d)", valid_path, decls_added_valid, len(valid_pairs)
    )


if __name__ == "__main__":
    main()
