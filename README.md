# Ontology Tradecraft

* **Week 1**
  - [Course Introduction & Tooling Setup]()
    - **Topics:** Course structure, Semantic Web stack (RDF, OWL, SPARQL, SHACL), VS Code, GitHub, Docker, Jupyter
    - **Readings:** Keet Ch. 1–2; Noy & McGuinness, *Ontology Development 101* (skim)
    - **Lab:** GitHub Classroom onboarding, Protégé refresher, local dev setup

* **Week 2**
  - [Ontology Engineering Methodology]()
    - **Topics:** Competency questions, hub-and-spoke, reuse, design patterns
    - **Readings:** Keet Ch. 3–4; Gangemi et al. (2008)
    - **Lab:** Draft competency questions, model ontology in Protégé

* **Week 3**
  - [Semantic Enrichment of Raw Data]()
    - **Topics:** RDF serialization, RDFS hierarchy, Linked Data best practices
    - **Readings:** W3C RDF Primer (2023)
    - **Lab:** Convert CSVs to RDF, publish to GitHub

* **Week 4**
  - [SPARQL for Querying]()
    - **Topics:** SELECT, CONSTRUCT, ASK, FILTER, OPTIONAL, aggregates; using SPARQL in Jupyter
    - **Readings:** W3C SPARQL 1.1 sections 1–13
    - **Lab:** Query DBpedia and Wikidata using SPARQLWrapper in notebooks

* **Week 5**
  - [OWL2 Reasoning]()
    - **Topics:** OWL2 profiles, reasoning (classification, consistency), reasoners (HermiT, Pellet, ELK)
    - **Readings:** Keet Ch. 5 & 7
    - **Lab:** Run reasoners in Protégé and Python; text bank challenge

* **Week 6**
  - [SHACL Validation]()
    - **Topics:** SHACL basics, node and property shapes, constraint definitions
    - **Readings:** SHACL by Example (Labra Gayo)
    - **Lab:** Use pySHACL to validate RDF graphs

* **Week 7**
  - [Data Integration with Ontologies]()
    - **Topics:** ETL from legacy data, R2RML & OpenRefine, Ontop, SQL vs SPARQL
    - **Readings:** FAIR Cookbook on RDF conversion; Rodriguez-Muro Ontop tutorial
    - **Lab:** Map relational DB to OWL2 ontology, run virtual SPARQL queries

* **Week 8**
  - [Midterm Project: ETL + Validation]()
    - **Topics:** RDF generation from CSVs, SHACL shape creation, OWL reasoning
    - **Lab:** Full integration task with pipeline, validation, and inference checks

* **Week 9**
  - [Knowledge Graph Design]()
    - **Topics:** Designing KGs for real-world use, linking ontologies
    - **Readings:** Hogan et al., *Knowledge Graphs* (CACM 2021)
    - **Lab:** Integrate two datasets into a unified RDF knowledge graph, federated querying and validation

* **Week 10**
  - [Ontology Embeddings with MOWL]()
    - **Topics:** Embeddings and semantic similarity
    - **Readings:** MOWL tutorial; Kulmanov et al. (2020)
    - **Lab:** Generate embeddings and cluster classes using Jupyter notebooks

* **Week 11**
  - [ML Applications in Ontology Engineering]()
    - **Topics:** Link prediction, classification, evaluation of ontology-based ML
    - **Readings:** Arnaout et al., *Ontology Embeddings: A Survey* (2023)
    - **Lab:** Train a basic ML model on embeddings

* **Week 12**
  - [LLMs in Ontology Engineering]()
    - **Topics:** Prompting for ontology building, SPARQL generation, LLM risk analysis
    - **Readings:** Tamma (2023); NVIDIA Blog on LLM + KGs
    - **Lab:** Use LLMs to generate competency questions, queries, and content

* **Week 13**
  - [Semantic Pipelines]()
    - **Topics:** Integrating ontologies in enterprise systems; case studies from healthcare, defense, logistics
    - **Readings:** Ontotext and Cambridge Semantics blogs
    - **Lab:** Build an RDF pipeline with SHACL validation and reasoning

* **Week 14**
  - [Student Presentations]()

* **Week 15**
  - [Student Presentations]()

## Extra Content

  - [MOWL Tutorial (Google Colab)](https://github.com/bio-ontology-research-group/MOWL)
  - [FAIR Cookbook – RDF Conversion](https://faircookbook.elixir-europe.org/content/recipes/interoperability/knowledge_representation/rdf-conversion.html)
  - [SHACL by Example (Labra Gayo)](https://labra.github.io/SHACL/)

## Repository Content
This repository contains the following directories: 

* **lectures** - Slides for lectures given by the instructor of the course.
* **presentations** - Slides for presentations given by participants in the course. 
* **readings** - Required readings for the course.

This course trains students to build semantically interoperable systems using the full Semantic Web stack, automated reasoning, and ontology-based machine learning workflows. Each week blends theoretical grounding with hands-on practice. Final projects will demonstrate students’ ability to model, query, validate, and augment knowledge using formal ontologies and AI.
