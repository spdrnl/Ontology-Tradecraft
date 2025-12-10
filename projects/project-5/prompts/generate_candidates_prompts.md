## Class

### System

# How to create a candidate OWL restriction for an existing OWL class

This prompt uses a Markdown format for human readability.

# Goal

Create a candidate owl:Restriction axiom for an existing OWL class, using its definition in English.
An owl:Restriction allows you to better define a class by showing what differentia distinguishes it from its superclass.

### User

# Instructions

- Identify the phrases placed in single quotes in the section 'Definition in English'.
- Your are looking for combinations of a verb phrase and a noun phrase.
- Look out for the mentioning of 'some' or 'only' in the context of this combination of phrases.
- You can also identify phrases that are near those mentioned in the section 'Mapping of phrases to URIREF'.
- Ignore phrases that are not mentioned in the section 'Mapping of phrases to URIREF'.
- Rewrite the definition in English so that its adheres to the mapping in the section 'Mapping of phrases to URIREF'.
- The combination of a verb phrase and a related noun phrase in English is a candidate owl:Restriction axiom.
- Verb phrases in the English definition refer to OWL properties.
- Noun phrases in the English definition refer to OWL classes.
- Use the section 'Mapping of verb and noun phrases to URIREF' below to lookup the OWL URIREF for the two phrases.
- You MUST only use the URIREFs from the section 'Mapping of verb and noun phrases to URIREF'.
- Create a candidate owl:Restriction axiom for {target_uri_ref} using the section 'New OWL restriction templates'.
- Check if the candidate owl:Restriction is already in place by using the section 'OWL definition'.
- If the owl:Restriction is already in place, discard new owl:Restriction.
- Output ONLY json with the following fields:
  - candidate_axioms: a list of candidate owl:Restriction axioms in Turtle format.
  - reasoning: text explaining why the owl:Restriction axioms were created.

# Definition in English

{target_definition}

# Mapping of phrases to URIREF

DO NOT USE ANY OTHER URIREFS THAN THESE
Here is a list for mapping verb phrases to their corresponding URIREFs:

```csv
Verb Phrase, URIREF
{properties_section}
```

Here is a list of verb phrases and their corresponding URIREFs:

```csv
Noun Phrase, URIREF
{classes_section}
```

# OWL definition

Here is the current OWL definition for the definition in Turtle format.

This section can be used to check if the new owl:Restriction is already in place.

```turtle
{owl_definition}
```

# New OWL restriction templates

Here are two templates for adding new owl:Restriction axioms to the current OWL definition.

Instructions for creating a new owl:Restriction:

- You need a combination of a verb phrase and a noun phrase.
- Both verb phrases and noun phrases MUST be occur in 'Definition in English'.
- Check if the verb and noun phrase URIREFs are different.
- Both URIREFs MUST be mentioned in the section 'Mapping of phrases to URIREF'.
- You need to replace {verb phrase URIREF} with the URIREF of the verb phrase.
- You need to replace {noun phrase URIREF} with the URIREF of the noun phrase.

The first template is for adding a restriction where the qualifier 'some' is used.
If there is no qualifier 'only', use this template.

```turtle

<{target_uri_ref}> rdfs:subClassOf [ a owl:Restriction ;
                        owl:onProperty <{verb phrase URIREF}> ;
                        owl:someValuesFrom <{noun phrase URIREF}> ] .
```

The second template is for adding a restriction where the qualifier 'only' is used.
If there is no qualifier 'only', use this template.

```turtle

<{target_uri_ref}> rdfs:subClassOf [ a owl:Restriction ;
                        owl:onProperty <{verb phrase URIREF}> ;
                        owl:allValuesFrom <{noun phrase URIREF}> ] .
```

## Property

### System

# How to create a candidate OWL restriction for an existing OWL class

This prompt uses a Markdown format for human readability.

# Goal

Create a candidate owl:Restriction axiom for an existing OWL class, using its definition in English.
An owl:Restriction allows you to better define a class by showing what differentia distinguishes it from its superclass.

### User

# Instructions

- Identify the phrases placed in single quotes in the section 'Definition in English'.
- Your are looking for combinations of a verb phrase and a noun phrase.
- Look out for the mentioning of 'some' or 'only' in the context of this combination of phrases.
- You can also identify phrases that are near those mentioned in the section 'Mapping of phrases to URIREF'.
- Ignore phrases that are not mentioned in the section 'Mapping of phrases to URIREF'.
- Rewrite the definition in English so that its adheres to the mapping in the section 'Mapping of phrases to URIREF'.
- The combination of a verb phrase and a related noun phrase in English is a candidate owl:Restriction axiom.
- Verb phrases in the English definition refer to OWL properties.
- Noun phrases in the English definition refer to OWL classes.
- Use the section 'Mapping of verb and noun phrases to URIREF' below to lookup the OWL URIREF for the two phrases.
- You MUST only use the URIREFs from the section 'Mapping of verb and noun phrases to URIREF'.
- Create a candidate owl:Restriction axiom for {target_uri_ref} using the section 'New OWL restriction templates'.
- Check if the candidate owl:Restriction is already in place by using the section 'OWL definition'.
- If the owl:Restriction is already in place, discard new owl:Restriction.
- Output ONLY json with the following fields:
  - candidate_axioms: a list of candidate owl:Restriction axioms in Turtle format.
  - reasoning: text explaining why the owl:Restriction axioms were created.

# Definition in English

{target_definition}

# Mapping of phrases to URIREF

DO NOT USE ANY OTHER URIREFS THAN THESE
Here is a list for mapping verb phrases to their corresponding URIREFs:

```csv
Verb Phrase, URIREF
{properties_section}
```

Here is a list of verb phrases and their corresponding URIREFs:

```csv
Noun Phrase, URIREF
{classes_section}
```

# OWL definition

Here is the current OWL definition for the definition in Turtle format.

This section can be used to check if the new owl:Restriction is already in place.

```turtle
{owl_definition}
```

# New OWL restriction templates

Here are two templates for adding new owl:Restriction axioms to the current OWL definition.

Instructions for creating a new owl:Restriction:

- You need a combination of a verb phrase and a noun phrase.
- Both verb phrases and noun phrases MUST be occur in 'Definition in English'.
- Check if the verb and noun phrase URIREFs are different.
- Both URIREFs MUST be mentioned in the section 'Mapping of phrases to URIREF'.
- You need to replace {verb phrase URIREF} with the URIREF of the verb phrase.
- You need to replace {noun phrase URIREF} with the URIREF of the noun phrase.

The first template is for adding a restriction where the qualifier 'some' is used.
If there is no qualifier 'only', use this template.

```turtle

<{target_uri_ref}> rdfs:subClassOf [ a owl:Restriction ;
                        owl:onProperty <{verb phrase URIREF}> ;
                        owl:someValuesFrom <{noun phrase URIREF}> ] .
```

The second template is for adding a restriction where the qualifier 'only' is used.
If there is no qualifier 'only', use this template.

```turtle

<{target_uri_ref}> rdfs:subClassOf [ a owl:Restriction ;
                        owl:onProperty <{verb phrase URIREF}> ;
                        owl:allValuesFrom <{noun phrase URIREF}> ] .
```