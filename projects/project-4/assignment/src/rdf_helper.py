"""
Provide utility functions to generate qualified names (QNames), create labels from class names and IDs,
and convert camel case strings to readable words. These functions are primarily useful in creating
unique identifiers and labels for classes and instances within linked data or namespace contexts.
"""
import base64
import logging
import pathlib
import re
from functools import lru_cache
from hashlib import sha1
from typing import Iterable

from rdflib import Namespace, URIRef, Literal, Graph

logger = logging.getLogger(__name__)

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
