import logging
from pathlib import Path
from typing import Any

from common.ontology_utils import get_parent_by_label
from common.string_normalization import contains, remove_snake_case, apply_label_casing, apply_single_quotes, \
    replace_x_and_y
from common.vectorization import vector_top_k_old, vector_top_k
from util.logger_config import config

logger = logging.getLogger(__name__)
config(logger)


def add_definition_prefix(elem_definition: str, elem_label: str, elem_type: str) -> str:
    if not contains(elem_definition, elem_label):
        if elem_type == "class":
            elem_definition = f"individual i is a '{elem_label}' iff individual i is a {elem_definition}"
        if elem_type == "property":
            elem_definition = f"individual i '{elem_label}' individual j iff {elem_label} is a {elem_definition}"
    return elem_definition

def create_automatic_class_definition(elem_label: str, phrase_diffs_dict, target_info: dict[str, Any]) -> str:
    parent_label = target_info.get("parent_label")

    if not (parent_label):
        axiomatic_definition = "There is not enough information for an axiomatic definition of this property."
    else:
        axiomatic_definition = f"individual i is a '{elem_label}' iff"
        if target_info.get("parent_label"):
            axiomatic_definition += f" individual i is a '{target_info.get('parent_label')}' "
        if phrase_diffs_dict.get(elem_label) and str(phrase_diffs_dict[elem_label]) != "nan":
            axiomatic_definition += f" where {phrase_diffs_dict[elem_label]}"
        axiomatic_definition += "."
    return axiomatic_definition

def create_automatic_property_definition(elem_label: str, phrase_diffs_dict, target_info: dict[str, Any]) -> str:
    parent_label = target_info.get("parent_label")
    child_domain = target_info.get("child_domain")
    child_range = target_info.get("child_range")

    if not (parent_label or child_domain or child_range):
        axiomatic_definition = "There is not enough information for an axiomatic definition of this property."
    else:
        axiomatic_definition = f"individual i '{elem_label}' individual j iff"
        if target_info.get("parent_label"):
            axiomatic_definition += f" individual i '{target_info.get('parent_label')}' individual j "
        if target_info.get("child_domain"):
            axiomatic_definition += f" {'and' if parent_label else ''} individual i is an instance of '{target_info.get('child_domain')}'"
        if target_info.get("child_range"):
            axiomatic_definition += f" {'and' if parent_label or child_domain else ''} individual j is an instance of '{target_info.get('child_range')}'"
        if phrase_diffs_dict.get(elem_label) and str(phrase_diffs_dict[elem_label]) != "nan":
            axiomatic_definition += f" {'and' if parent_label or child_domain or child_range else ''} {phrase_diffs_dict[elem_label]}"
        axiomatic_definition += "."
    return axiomatic_definition


def normalize_definition_prefix(improved_definition: str,
                                type_name: str,
                                label: str) -> str:
    improved_definition = improved_definition.split("iff")[1].strip()
    if type_name == "property":
        improved_definition = f"individual i '{remove_snake_case(label)}' individual j iff {improved_definition}"
    elif type_name == "class":
        improved_definition = f"individual i is a '{remove_snake_case(label)}' iff {improved_definition}"
    else:
        logger.warning(f"Unknown element type: {type_name}")
    return improved_definition


def clean_up_definition(elem_definition: str, elem_label: str, elem_type: str, ref_labels) -> str:
    elem_definition = remove_snake_case(elem_definition)
    elem_definition = apply_label_casing(elem_definition, ref_labels)
    elem_definition = apply_single_quotes(elem_definition, ref_labels)
    elem_definition = replace_x_and_y(elem_definition)
    elem_definition = add_definition_prefix(elem_definition, elem_label, elem_type)
    return elem_definition


def create_class_definition_prompt(elem_iri: str, elem_label: str, elem_type: str, elem_definition: str,
                                   prompts, phrase_diffs_dict, ref_entries, ref_labels,
                                   settings: dict) -> str:

    prompt = prompts.get("property").get("system") + "\n" + prompts.get("property").get("user")

    ###################################################################################
    # Clean up the definition
    elem_definition = clean_up_definition(elem_definition, elem_label, elem_type, ref_labels)

    prompt = prompt.replace("{label}", elem_label)
    prompt = prompt.replace("{definition}", elem_definition)

    ##################################################################################
    # Create automatic definition
    target_info = get_parent_by_label(elem_label, "class", Path("src/ConsolidatedCCO.ttl"))
    automatic_definition = create_automatic_class_definition(elem_label, phrase_diffs_dict, target_info)

    prompt = prompt.replace("{automatic_definition}", automatic_definition)

    ##################################################################################
    # Create reference contexts
    query = f"{elem_label or ''} {elem_definition or ''}"
    ref_top_k = settings.get("ref_top_k")

    # Perform vector search for property and class references
    property_references = vector_top_k(query, "property", ref_top_k, settings)
    class_references = vector_top_k(query, "class", ref_top_k, settings)

    # Add property and class references to prompt
    property_reference_lines = []
    for ref in property_references:
        property_reference_lines.append(f"- {ref['label']} {ref['definition']}")
    property_reference_context = "\n".join(property_reference_lines)
    prompt = prompt.replace("{property_reference_context}", property_reference_context)

    class_reference_lines = []
    for ref in class_references:
        class_reference_lines.append(f"- {ref['label']} {ref['definition']}")
    class_reference_context = "\n".join(class_reference_lines)
    prompt = prompt.replace("{class_reference_context}", class_reference_context)

    return prompt


def create_property_definition_prompt(elem_iri: str, elem_label: str, elem_type: str, elem_definition: str,
                                      prompts, phrase_diffs_dict, ref_entries, ref_labels,
                                      settings: dict) -> str:
    prompt = prompts.get("property").get("system") + "\n" + prompts.get("property").get("user")

    ###################################################################################
    # Clean up the definition
    elem_definition = clean_up_definition(elem_definition, elem_label, elem_type, ref_labels)

    prompt = prompt.replace("{label}", elem_label)
    prompt = prompt.replace("{definition}", elem_definition)

    ##################################################################################
    # Create automatic definition
    target_info = get_parent_by_label(elem_label, "property", Path("src/ConsolidatedCCO.ttl"))
    automatic_definition = create_automatic_property_definition(elem_label, phrase_diffs_dict, target_info)
    prompt = prompt.replace("{automatic_definition}", automatic_definition)

    lines = []
    if target_info.get("parent_domain"):
        lines.append(
            f"The domain 'C' of the parent property 'Y' is: '{target_info.get('parent_domain')}'")
    else:
        lines.append("This the parent property 'Y' has no domain 'C'.")

    if target_info.get("parent_range"):
        lines.append(f"The range 'D' of the parent property 'Y' is: '{target_info.get('parent_range')}'")
    else:
        lines.append("The parent property 'Y' has no range 'D'.")
    parent_context = "\n".join(lines)
    prompt = prompt.replace("{parent_context}", parent_context)


    ##################################################################################
    # Create reference contexts
    query = f"{elem_label or ''} {elem_definition or ''}"
    ref_top_k = settings.get("ref_top_k")

    # Perform vector search for property and class references
    property_references = vector_top_k(query, "property", ref_top_k, settings)
    class_references = vector_top_k(query, "class", ref_top_k, settings)

    # Add property and class references to prompt
    property_reference_lines = []
    for ref in property_references:
        property_reference_lines.append(f"- {ref['label']} {ref['definition']}")
    property_reference_context = "\n".join(property_reference_lines)
    prompt = prompt.replace("{property_reference_context}", property_reference_context)

    class_reference_lines = []
    for ref in class_references:
        class_reference_lines.append(f"- {ref['label']} {ref['definition']}")
    class_reference_context = "\n".join(class_reference_lines)
    prompt = prompt.replace("{class_reference_context}", class_reference_context)

    return prompt
