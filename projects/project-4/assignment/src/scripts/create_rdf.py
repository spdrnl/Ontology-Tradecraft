import argparse
import logging
import pathlib

import pandas as pd
from pandas import DataFrame
from rdflib import Graph, URIRef, Namespace, OWL, RDF, RDFS, Literal, SKOS


from rdf_helper import unique_qname, label_from_class, write_ttl_kg

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "source"
OUT_DIR = ROOT / "data" / "interim"

INPUT_PATH = OUT_DIR / "readings_normalized.csv"
OUTPUT_PATH = OUT_DIR / "readings_normalized.ttl"
DEFAULT_NS = Namespace("http://www.newfoundland.nl/otc/project-4")
NS_CCO = Namespace("https://www.commoncoreontologies.org/")


def get_args() -> argparse.Namespace:
    """
    Fetches command-line arguments required for processing SPARQL query files against Turtle files
    and outputs the results into a CSV file. Parses the provided arguments and validates their
    presence and expected types.

    :return: Namespace object containing parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    # Initialize
    parser = argparse.ArgumentParser(
        description="The program import a normalized readings CSV input file and outputs an ontology in Turtle (TTL) file.",
        epilog="Happy ontology hacking!",
        prog="create-rdf",
    )

    # Adding optional parameters
    parser.add_argument(
        "-i", "--input-file", help="CSV input file.", default=INPUT_PATH, type=str
    )

    parser.add_argument(
        "-o", "--output-file", help="Turtle output file.", default=OUTPUT_PATH, type=str
    )

    parser.add_argument(
        "-ns", "--namespace", help="Namespace.", default=DEFAULT_NS, type=str
    )

    return parser.parse_args()


def main():
    # Resolve arguments
    args = get_args()
    input_file = args.input_file
    output_file = args.output_file
    namespace = args.namespace
    ns = Namespace(namespace)

    # Read CSV
    df = pd.read_csv(input_file, header=0)

    # Create ontology
    g = Graph()
    create_ontology(g, namespace)

    # Translate data to RDF
    translate_to_rdf(df, g, ns)

    # Write graph to file
    write_ttl_kg(g, output_file, Namespace(ns))


def create_ontology(g: Graph, namespace):
    """
    Creates an ontology in the RDF graph for sensor data readings and measurements.

    This function adds ontology information including its URI, type, label, and
    description to the provided RDF graph.

    :param g: The RDF graph where ontology definition will be added.
    :type g: Graph
    :param namespace: The namespace URI for the ontology.
    :type namespace: str
    :return: None
    """
    ontology_uri = URIRef(namespace)
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    g.add((ontology_uri, RDFS.label, Literal("Sensor Data Ontology", lang='en')))
    g.add((ontology_uri, RDFS.comment, Literal("An ontology for sensor readings and measurements", lang='en')))


def translate_to_rdf(df: DataFrame, g: Graph, ns: Namespace):
    create_sensors(df, g, ns)
    create_observed_entities(df, g, ns)
    create_sensor_readings(df, g, ns)


def create_sensor_readings(df: DataFrame, g: Graph, ns: Namespace):
    """
    Create sensor readings by iterating through the provided dataset and generating instances
    of the 'SensorObservation' class in the given graph. Each instance corresponds to a
    measurement taken at a specific time by a sensor on an observed entity, associating the
    necessary properties derived from the data.

    :param df: A DataFrame containing the input data for generating sensor readings. Each row
        is expected to include specific observed entity details, source, and timestamp.
    :type df: DataFrame
    :param g: RDF graph where the sensor observation instances will be added.
    :type g: Graph
    :param ns: Namespace defining IRIs for the entities being created.
    :type ns: Namespace
    :return: None
    """
    # Create sensor readings
    class_name = "SensorObservation"
    g.add((ns[class_name], RDF.type, OWL.Class))
    g.add((ns[class_name], RDFS.label, label_from_class(class_name)))
    g.add((ns[class_name], SKOS.definition, Literal("A sensor measurement of an entity at a certain time.", lang='en')))

    for index, row in df.iterrows():
        q_name = unique_qname(class_name, [row.observed_entity_id, row.source, row.timestamp])
        g.add((ns[q_name], RDF.type, ns[class_name]))
        label = f"Sensor {row.source} on {row.observed_entity_id} at {row.timestamp}"
        g.add((ns[q_name], RDFS.label, Literal(label, lang='en')))


def create_observed_entities(df: DataFrame, g: Graph, ns: Namespace):
    """
    Creates observed entities in a given RDF graph based on a data frame and namespace.

    This function processes data in the input DataFrame, identifies unique
    observed entity IDs, and creates corresponding classes in the RDF graph.
    Each observed entity class is assigned a label derived from its ID for identification,
    and its type is specified as an OWL Class. The function adds descriptions
    and annotations to the RDF graph using RDF, SKOS, OWL, and RDFS ontologies.

    :param df: DataFrame containing observed entity data. Must include the
        'observed_entity_id' column.
    :type df: pandas.DataFrame
    :param g: RDF Graph where the observed entities will be added.
    :type g: rdflib.Graph
    :param ns: Namespace to define and reference RDF entities being created.
    :type ns: rdflib.Namespace
    :return: None
    """
    class_name = "ObservedEntity"
    g.add((ns[class_name], RDF.type, OWL.Class))
    g.add((ns[class_name], RDFS.label, label_from_class(class_name)))
    g.add((ns[class_name], SKOS.definition, Literal("An entity that is measured by a sensor.", lang='en')))

    observed_entity_ids = df['observed_entity_id'].unique()
    for observed_entity_id in observed_entity_ids:
        q_name = unique_qname(class_name, [observed_entity_id])
        g.add((ns[q_name], RDF.type, ns[class_name]))
        g.add((ns[q_name], RDFS.label, Literal(observed_entity_id, lang='en')))


def create_sensors(df: DataFrame, g: Graph, ns: Namespace):
    """
    Creates sensor classes and instances in the given graph based on the provided DataFrame.

    The function defines a new OWL class for sensors and adds sensor instances
    for each unique combination of `observed_entity_id` and `source` in the input
    DataFrame. Each sensor instance will have a unique identifier and a label
    to describe its relation with the observed entity and the source.

    :param df: A pandas DataFrame containing the data for observed entities and
        sources. It must have the columns `observed_entity_id` and `source`.
    :param g: The RDF graph where sensor classes and instances will be added.
    :param ns: The RDF namespace for creating classes and instances.
    :return: None
    """
    ## Create sensor class
    class_name = "Sensor"
    g.add((ns[class_name], RDF.type, OWL.Class))
    g.add((ns[class_name], RDFS.label, label_from_class(class_name)))
    g.add((ns[class_name], SKOS.definition, Literal("A mechanical device that can generate a measurement.", lang='en')))

    ## Create sensor instances
    source_ids = df[['observed_entity_id', 'source']].drop_duplicates()
    for _, row in source_ids.iterrows():
        q_name = unique_qname(class_name, [row.observed_entity_id, row.source])
        label = f"Sensor {row.source} on {row.observed_entity_id}"
        g.add((ns[q_name], RDF.type, ns[class_name]))
        g.add((ns[q_name], RDFS.label, Literal(label, lang='en')))


if __name__ == "__main__":
    main()
