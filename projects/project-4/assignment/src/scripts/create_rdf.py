import argparse
import pandas as pd
from pandas import DataFrame
from rdflib import OWL, RDF, RDFS, SKOS
from rdflib import Namespace, URIRef, Literal, Graph
import base64
import logging
import pathlib
import re
from functools import lru_cache
from hashlib import sha1
from typing import Iterable

logger = logging.getLogger(__name__)

# Path settings
SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_SOURCE = SRC_ROOT / "data" / "source"
DATA_INTERIM = SRC_ROOT / "data" / "interim"
INPUT_PATH = DATA_INTERIM / "readings_normalized.csv"
OUTPUT_PATH = DATA_INTERIM / "readings_normalized.ttl"

# Namespace settings
DEFAULT_NS = Namespace("http://www.newfoundland.nl/otc/project-4/")
NS_CCO = Namespace("https://www.commoncoreontologies.org/")


def write_ttl_kg(
        graph: Graph, filename: pathlib.Path, default_ns: Namespace = None, base: str = None
) -> None:
    """
    Writes an RDF graph to a Turtle (.ttl) file with optional default namespace
    and base URI configuration.

    This function allows serializing an RDF graph object to the Turtle format
    and writing it to a file. A default namespace can optionally be bound to
    the graph, and a base URI can also be specified for serialization.

    :param graph: The RDF graph to be serialized and written to the file.
    :type graph: Graph
    :param filename: The path to the file where the Turtle data will be written.
    :type filename: str
    :param default_ns: Optional default namespace to bind to the graph.
    :type default_ns: Namespace, optional
    :param base: Optional base URI to use during serialization.
    :type base: str, optional
    :return: This method does not return any value.
    :rtype: None
    """
    logger.info(f"Writing graph to {filename}.")
    logger.info(f"Using default namespace {default_ns}.")
    logger.info(f"And base URI {base}.")

    if default_ns:
        graph.bind("", default_ns)

    try:
        with open(filename, "wb") as f:
            graph.serialize(f, format="turtle", base=base)
    except Exception as ex:
        logger.info(ex)
        raise Exception(f"Could not write graph to {filename}: {ex}.") from ex


def unique_qname(class_name: str, elements: Iterable[str]) -> URIRef:
    """
    Generate a qualified name (QName) for an instance based on its namespace, class name,
    and a collection of elements. The QName is constructed by converting the class name
    to lowercase with dashes and appending a unique identifier derived from the elements.
    This function is helpful in namespaces and linked data contexts for creating unique
    identifiers.

    :param class_name: The name of the class for which the instance QName is being generated.
    :type class_name: str
    :param elements: An iterable collection of elements used to derive a unique identifier.
        This ensures the uniqueness of the generated QName.
    :type elements: Iterable[str]
    :return: A generated QName that uniquely identifies an instance within the namespace.
    :rtype: URIRef
    """
    qname_prefix = qname_from_class(class_name)
    qname_suffix = qname_id_suffix(elements)
    qname = f"{qname_prefix}-{qname_suffix}"
    return qname


def qname_from_class(class_name: str) -> str:
    """
    Generates a "qualified" name for a class by converting its class name to a string that is
    in lowercase, hyphen-separated format. The class name is first converted from camel case
    to a human-readable string with spaces, and then further processed to match the desired format.

    :param class_name: The name of the class to convert.
    :type class_name: str
    :return: A string representation of the class name in lowercase, hyphen-separated format.
    :rtype: str
    """
    return camel_case_to_words(class_name).lower().replace(' ', '-')


def qname_id_suffix(elements: Iterable[str]) -> str:
    """Generate a SHA-
    1 hash-based identifier from a collection of string elements.

    Args:
        elements: An iterable of strings to be hashed together

    Returns:
        Base64-encoded string representation of the SHA-1 hash, safe for use as a QName
    """
    joined_elements = ':'.join(elements)
    encoded_elements = joined_elements.encode('utf-8')
    hash_object = sha1(encoded_elements)
    # Use urlsafe_b64encode for QName compatibility (- and _ instead of + and /)
    # Remove padding (=) and decode to string
    qname_safe_id = base64.urlsafe_b64encode(hash_object.digest()).rstrip(b'=').decode('ascii')
    return qname_safe_id


def label_from_class_id(class_name: str, id: str) -> str:
    """
    Generates a descriptive label by combining a class name with its identifier.

    This function facilitates the creation of a formatted label string that combines
    a given class name and its identifier. It is intended to enhance readability
    and identification of objects or entities based on their class and unique ID.

    :param class_name: The name of the class to generate the label for.
    :type class_name: str
    :param id: The unique identifier associated with the class.
    :type id: str
    :return: The formatted label combining the class name and identifier.
    :rtype: str
    """
    label = f"{label_from_class(class_name)} '{id}'"
    return label


def label_from_class(class_name: str, lang: str = 'en') -> Literal:
    """
    Converts a camel case class name into a readable label and wraps it
    as an RDF `Literal` with a specified language code. The function
    modifies the class name to make the first letter uppercase and
    the rest lowercase for better readability.

    :param class_name: The camel case class name to be converted
        into a readable label.
    :type class_name: str
    :param lang: The language code for the RDF `Literal`. Defaults to 'en'.
    :type lang: str, optional

    :return: An RDF `Literal` object containing the converted readable
        label and language code.
    :rtype: Literal
    """
    label = camel_case_to_words(class_name)
    label = label[0].upper() + label[1:].lower()
    literal = Literal(label, lang=lang)
    return literal


@lru_cache(maxsize=1024 * 1024)
def camel_case_to_words(text) -> str:
    """
    Convert camel case words to separate words with spaces.
    Results are cached for improved performance on repeated calls.

    Args:
        text: String containing camel case words

    Returns:
        String with camel case words separated by spaces

    Examples:
        >>> camel_case_to_words("thisIsATest")
        'this Is A Test'
        >>> camel_case_to_words("readingValue")
        'reading Value'
        >>> camel_case_to_words("deviceName")
        'device Name'
    """
    if not text:
        return text

    # Insert space before uppercase letters that follow lowercase letters
    # or before uppercase letters that are followed by lowercase letters (for acronyms)
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', result)

    return result


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
