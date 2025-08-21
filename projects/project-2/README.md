# Week 2 Project: Enriching Raw Data

## Learning Objectives
By completing this project, you will:
- Leverage **competency questions** to design a small RDF graph, **reusing** ontology elements where sensible.
- Use **rdflib** to construct RDF from **raw data**.
- Follow **Linked Data best practices** (stable HTTP IRIs, labels, datatypes).
- Validate your output via **automated tests**.

## Project

### A Competency Questions (provided)
Model a small slice of the domain so that your RDF can answer these four questions:

1. **Equipment assignment** — *Which equipment items are assigned to which personnel?*  
2. **Training completion** — *Which training events has a given individual completed, and on what date?*  
3. **Mission roles** — *Which roles are associated with each type of mission?*  
4. **Data provenance & attributes** — *Which data sources record operational activities, and which attributes do they capture?*

> Your job is to: (i) model a tiny vocabulary, (ii) generate RDF instances, such that **SPARQL queries run by the autograder return non-empty results** for each question.

### B Minimal Vocabulary (local names to use)
You may choose any base IRI/namespace, but **use these local names** so the tests can find your terms (case-sensitive):

**Classes:**  
`Person`, `Equipment`, `TrainingEvent`, `MissionType`, `Role`, `OperationalActivity`, `DataSource`

**Properties:**  
`assignedTo` (Equipment → Person)  
`completedTraining` (Person → TrainingEvent)  
`completionDate` (TrainingEvent → xsd:date or xsd:dateTime)  
`hasRole` (MissionType → Role)  
`recordedIn` (OperationalActivity → DataSource)  
`capturesAttribute` (DataSource → literal attribute name; xsd:string)

> You can define extra classes/properties as needed. You may also reuse external terms (FOAF, schema.org, etc.), in addition to these.

### C Hub-and-Spoke 
- Put your **“hub” vocabulary** (core class/property declarations) in `assignment/src/data/core.ttl`.  
- Split sub-areas into **spokes** (e.g., `module-training.ttl`, `module-ops.ttl`).  
- If you split modules, include `owl:imports` in `core.ttl`.

### D Generate RDF instances with `rdflib`
- Create a tiny CSV or define instances directly in code—notebook—your choice.
- Build at least **20 triples** total.
- Use **HTTP(S) IRIs** for most resources and include at least **one `rdfs:label`**.

### E You do **not** write SPARQL for grading
- The autograder will run SPARQL queries to verify that your data can answer the CQs.
- Your task is to ensure your RDF uses the **local names above**, so the tests can match patterns regardless of your namespace.

## Automated Grading
Grading is performed by `pytest` + SPARQL checks against your generated graph. You **don’t** need to write SPARQL; just follow the local-name contract and produce coherent RDF.

What the grader does:
1. Loads all `.ttl` files in `assignment/src/data/`.
2. Verifies: parses, **≥20 triples**, majority subjects are **URIRefs**, at least one `rdfs:label`.
3. Runs four internal SPARQL queries (one per CQ) that check for **non-empty results** using your classes/properties **by local name** (so any base IRI works).
4. (Bonus) If your `core.ttl` contains **≥1 `owl:imports`**, you get credit for modularization.

---

**Rubric:**
- **CQ coverage via data (SPARQL returns non-empty)** — 40%  
- **RDF quality & Linked Data** (parse OK, ≥20 triples, labels, URI subjects) — 25%  
- **Vocabulary correctness** (required class/property local names present) — 20%  
- **Modularity/reuse** (hub file present; bonus for `owl:imports` and any external reuse noted in README) — 10%  
- **Repo hygiene** (runs locally and in CI without errors) — 5%

*A passing grade requires all four CQs to return non-empty results.*

---

## Files in the Repository

