"""Post-filter vector matches and build SPARQL-style binding rows.

Validates each Milvus hit against the triple pattern and bound values, then
maps variables to URI/literal terms. Returns result rows plus the set of
entity ids that passed validation (used as the adaptive k stability key).

``parse_rdf_triple`` is injected to avoid a circular import with ``vectorEndpoint``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


def filter_matches_to_rows(
    matches: list[dict],
    *,
    pattern_subject: Any,
    pattern_predicate: Any,
    pattern_object: Any,
    validation_info: dict,
    variables: list[str],
    pattern_variable_roles: dict[str, str],
    value_row: dict,
    parse_rdf_triple: Callable[[str], Optional[dict]],
    log: bool = False,
) -> tuple[list[dict], set[int]]:
    """Validate each match and build result rows.

    Returns:
        (results_rows, filtered_ids)
        - results_rows: list of variable-binding rows ready for the response
        - filtered_ids: set of `match["id"]` for matches that passed validation
    """
    subject = pattern_subject
    predicate = pattern_predicate
    obj = pattern_object

    subject_value = validation_info.get("subject_value")
    predicate_value = validation_info.get("predicate_value")
    object_value = validation_info.get("object_value")
    subj_is_var = validation_info.get("subj_is_var", True)
    pred_is_var = validation_info.get("pred_is_var", True)
    obj_is_var = validation_info.get("obj_is_var", True)

    results_rows: list[dict] = []
    filtered_ids: set[int] = set()

    for match_idx, match in enumerate(matches):
        triple_text = match.get("text", "")
        triple_data = parse_rdf_triple(triple_text)
        if not triple_data:
            continue

        subject_matched = True
        if isinstance(subject, dict) and subject.get('type') == 'iri':
            if triple_data['subject'] != subject['value']:
                subject_matched = False
        elif subject_value and not subj_is_var:
            expected_subj = subject_value.lstrip('<').rstrip('>')
            if triple_data['subject'] != expected_subj:
                subject_matched = False

        if not subject_matched:
            continue

        predicate_matched = True
        if isinstance(predicate, dict) and predicate.get('type') == 'iri':
            if triple_data['predicate'] != predicate['value']:
                predicate_matched = False
        elif predicate_value and not pred_is_var:
            expected_pred = predicate_value.lstrip('<').rstrip('>')
            if triple_data['predicate'] != expected_pred:
                predicate_matched = False

        if not predicate_matched:
            continue

        object_matched = True
        if isinstance(obj, dict):
            if obj.get('type') == 'literal':
                if triple_data['object'] != obj['value'] or triple_data['object_type'] != 'literal':
                    object_matched = False
            elif obj.get('type') == 'iri':
                if triple_data['object'] != obj['value'] or triple_data['object_type'] != 'uri':
                    object_matched = False
        elif object_value and not obj_is_var:
            expected_obj = object_value.lstrip('<').rstrip('>')
            if triple_data['object'] != expected_obj:
                object_matched = False

        if not object_matched:
            continue

        match_id = match.get("id")
        if match_id is not None:
            filtered_ids.add(match_id)

        row: dict = {}
        for var in variables:
            var_name = var.lstrip('?')
            role = pattern_variable_roles.get(var_name)

            if role == 'subject':
                row[var] = {"type": "uri", "value": triple_data['subject']}
            elif role == 'predicate':
                row[var] = {"type": "uri", "value": triple_data['predicate']}
            elif role == 'object':
                obj_type = 'literal' if triple_data['object_type'] == 'literal' else 'uri'
                row[var] = {"type": obj_type, "value": triple_data['object']}
            elif var_name in value_row:
                term_json = value_row[var_name]
                if term_json is not None:
                    if isinstance(term_json, dict):
                        row[var] = {
                            "type": term_json.get('type', 'uri'),
                            "value": term_json.get('value'),
                        }
                    else:
                        row[var] = {"type": "uri", "value": term_json}
            elif var in value_row:
                term_json = value_row[var]
                if term_json is not None:
                    if isinstance(term_json, dict):
                        row[var] = {
                            "type": term_json.get('type', 'uri'),
                            "value": term_json.get('value'),
                        }
                    else:
                        row[var] = {"type": "uri", "value": term_json}
        if row:
            results_rows.append(row)

    return results_rows, filtered_ids
