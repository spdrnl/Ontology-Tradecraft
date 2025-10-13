# Week 4 Project: ETL + Mapping + Validation Midterm

## Learning Objectives

By the end of this project, you will be able to:

1. Extract, normalize, and transform provided measurement data into a structured format prior to building RDF.
2. Using RDFlib, construct RDF graphs that conform to a provided Common Core Ontologies (CCO) design pattern.
3. Construct and appply SPARQL quality-control (QC) queries to verify ontology integrity.
4. Construct and apply SHACL shapes to check semantic conformance and data integrity.
5. Automate semantic validation in a continuous integration (CI) pipeline using GitHub Actions.

This project serves as your midterm milestone, combining data engineering, ontology modeling, and semantic quality assurance within a reproducible workflow.

## Project Overview

You are provided with two raw data sources (`src/data/sensor_A.csv` and `src/data/sensor_B.json`) containing measurement data with inconsistent schemas, units, and encoding. Your task is to:

1. Combine both raw files into a clean intermediate dataset (`src/data/readings_normalized.csv`) using the Python library Pandas. In doing so, you must ensure consistent column names, datatypes, timestamp formats, and unit codes.
2. Using RDFlib, construct RDF triples that instantiate a provided CCO Measurement Design Pattern (`src/cco_measurement_dp.png`), outputting the graph to `src/measure_cco.ttl`.
3. Using the descriptions in each `*.rq` file located in `src/sparql`, construct 8 SPARQL QC queries, each of which should return **zero results** when the queried structure is valid. For example, one such query will evaluate whether each ontology element includes a unique `rdfs:label` value. If so, this query will return no results; otherwise, the offending ontology element will be returned. 
4. Run the `*.rq` queries automatically using `src/scripts/run_sparql_qc.py`.
5. Using the descriptions found in the `src/cco_shapes.ttl` file, construct SHACL shapes to validate your RDF graph’s structure and datatype conformance. 
6. Run `cco_shapes.ttl` file automatically using `src/scripts/shacl_validate.py`.
7. Create CI workflow using a `yml` file that will run your ETL script, generate the CCO-based RDF graph, execute all SPARQL QC checks and SHACL validation returning a report when concluded, and which fails if any QC or SHACL test fails.

When you have successfully concluded this project, you will have created a CI workflow in which you clean data, transform it into a CCO-based RDF graph, which is then validated for quality and accuracy. This pipeline can be used to automate the integration of similarly structured data sources, resulting in high-quality CCO-based knowledge graphs. 

## Automated Grading

**Rubric:**

## Files in the Repository

### ✅ Expected Deliverables


