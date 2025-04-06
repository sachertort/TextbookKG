import re
import string
from collections import defaultdict
import json
from typing import Optional

import spacy
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import numpy as np
import networkx as nx
from tqdm.auto import tqdm

from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.vector_stores.base import VectorStoreDocument
from graphrag.model.types import TextEmbedder


nlp = spacy.load("en_core_web_sm")
text_embedder = OpenAIEmbeddings(model="text-embedding-3-small")

def singularize_nouns(text: str) -> str:
    """
    Singularizes all noun tokens in the input text using spaCy's lemmatizer.

    Args:
        text: The input text which may contain plural nouns.

    Returns:
        A new string where all noun tokens (including proper nouns) are converted to their singular form.
    """
    doc = nlp(text.lower())
    singularized_text = "".join(
        (token.lemma_ if token.pos_ in {"NOUN", "PROPN"} else token.text) + token.whitespace_
        for token in doc
    )
    return singularized_text.upper()

def tokenize_text(text: str) -> tuple[str, ...]:
    """
    Tokenizes the input text based on spaces and punctuation marks (including hyphen).

    Args:
        text: The string to tokenize.

    Returns:
        A tuple containing the tokens extracted from the text.
    """
    pattern = f"[\\s{re.escape(string.punctuation)}]+"
    tokens = re.split(pattern, text)
    return tuple(token for token in tokens if token)

def is_potential_acronym(text: str) -> bool:
    """
    Checks if a text might be an acronym based on its length and absence of spaces.
    
    Args:
        text: The string to check if it's a potential acronym.
        
    Returns:
        True if the text appears to be an acronym (3-4 letters with no spaces), 
        False otherwise.
    """
    if " " not in text and 3 <= len(text) <= 4:
        return True
    else:
        return False

def extract_acronyms(text: str) -> tuple[str, str]:
    """
    Extracts acronyms and their full forms from text containing patterns like "TERM (ACRONYM)" or "ACRONYM (TERM)".
    
    Args:
        text: The string that potentially contains an acronym in parentheses.
        
    Returns:
        A tuple containing (full_form, acronym) where full_form is the expanded text and acronym is the abbreviated form.
    """
    if "(" in text and ")" in text:
        parts = text.split("(")
        potential_acronym = parts[1].split(")")[0].strip()
        full_form = parts[0].strip()
        
        # Check if what's in parentheses is the acronym (shorter) or the full form (longer)
        if len(potential_acronym) < len(full_form):
            return full_form, potential_acronym
        else:
            # If what's in parentheses is longer, it's likely the full form and outside is the acronym
            return potential_acronym, full_form
    else:
        return None, None

def check_acronym(full_form, potential_acronym):
    """
    Checks if the full form and potential acronym are a valid acronym expansion.
    
    Args:
        full_form: The full form of the acronym.
        potential_acronym: The potential acronym.
        
    Returns:
        True if the potential acronym is a valid expansion of the full form, False otherwise.
    """
    full_form_tokens = tokenize_text(full_form)
    acronym_tokens = list(potential_acronym)
    
    # Check if the acronym is a valid expansion of the full form
    if len(full_form_tokens) == len(acronym_tokens):
        for full_token, acronym_token in zip(full_form_tokens, acronym_tokens):
            if full_token[0] != acronym_token[0]:
                return False
        return True
    else:
        return False
    
def build_candidates_index(candidates: list[str]) -> dict[tuple[int, tuple[str, ...]], list[str]]:
    """
    Builds an index from candidate full forms based on their token count and the first letters of each token.
    
    Args:
        candidate_full_forms: A list of strings that may contain full forms for acronyms.
        
    Returns:
        A dictionary where the key is a tuple (number of tokens, tuple of first letters) and the value is a list of full forms.
    """
    index: dict[tuple[int, tuple[str, ...]], list[str]] = defaultdict(list)
    for full_form in candidates:
        tokens = tokenize_text(full_form)
        if tokens:
            key = (len(tokens), tuple(token[0] for token in tokens))
            index[key].append(full_form)
    return index

def find_full_form(acronym: str, index: dict[tuple[int, tuple[str, ...]], list[str]]) -> Optional[str]:
    """
    Finds the full form of an acronym using the candidate index.
    
    Args:
        acronym: The acronym to find the full form for.
        index: A dictionary where the key is a tuple (number of tokens, tuple of first letters) 
               and the value is a list of candidate full forms.
    
    Returns:
        The full form of the acronym if found, None otherwise.
    """
    key = (len(acronym), tuple(acronym))
    candidates = index.get(key, [])
    for full_form in candidates:
        if check_acronym(full_form, acronym):
            return full_form
    return None

def merge_keys(d: defaultdict[str, set[str]]) -> None:
    """
    Improved merge_keys function using union-find (disjoint-set) to merge keys efficiently.
    Merges keys in the defaultdict if a key is present in the value set of another key.
    For each key in d, if the key is found in the value set of another key, this function adds
    the associated duplicate values to the canonical key's set and then removes the duplicate key from d.
    
    Args:
        d: A defaultdict mapping strings to sets of strings.
    
    This function modifies d in place.
    """
    parent = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootY] = rootX

    # Initialize union-find: each key is its own parent.
    for key in d.keys():
        parent[key] = key

    # Union keys if a key appears in the value set of another.
    for key, values in d.items():
        for val in values:
            if val in d:  # Only merge if val is also a key.
                union(key, val)

    # Group keys by their root representative.
    groups = defaultdict(set)
    for key in d.keys():
        groups[find(key)].add(key)

    # Merge the groups: canonical key gets the union of all duplicate value sets, 
    # but we do NOT add the keys themselves into the duplicate sets.
    for rep, keys in groups.items():
        merged_set = set()
        for k in keys:
            merged_set.update(d[k])
        d[rep] = merged_set
        for k in keys:
            if k != rep:
                del d[k]

def concept_deduplication(concepts: list[str]) -> defaultdict[str, set[str]]:
    """
    Deduplicates and normalizes a list of concept strings by consolidating variations based on singularization,
    tokenization, formatting, and acronym extraction.

    The process is performed in several steps:
      1. Singularization: Each concept is processed with `singularize_nouns`. If the singularized version exists
         in the original list, it is used as the canonical form and any differing original concept is recorded as a duplicate.
      2. Tokenization and Formatting: The singularized concepts are tokenized using `tokenize_text` and grouped by their
         tokenized representation. For groups with multiple variants, a preferred formatting option is chosen (favoring
         concepts containing a hyphen if available), and the other variants are marked as duplicates.
      3. Acronym Extraction: The function identifies potential acronyms using `is_potential_acronym` and extracts
         acronyms along with their full forms via `extract_acronyms`. Valid acronym expansions are confirmed with
         `check_acronym`, and duplicates are recorded accordingly.
      4. Full Form Matching: For potential acronyms without an existing mapping, the function searches for corresponding
         full forms using a candidate index built from the singularized concepts and consolidates the results.
      5. Merging: Finally, keys in the duplicates mapping are merged if one key is found within the value set of another,
         ensuring a unified representation of equivalent concepts.

    Args:
        concepts: A list of concept strings to be deduplicated and normalized.

    Returns:
        A defaultdict where each key is a preferred concept string (possibly including formatted full forms and acronyms)
        and the corresponding value is a set of duplicate strings representing alternative forms of that concept.
    """
    duplicates = defaultdict(set)
    
    # --- Changed: Optimize membership checks by converting to a set ---
    concepts_set = set(concepts)
    
    singularized_concepts = set()
    singularize_cache = {}  # --- Changed: Cache for singularize_nouns ---
    for concept in tqdm(concepts, desc="Singularizing concepts"):
        if concept in singularize_cache:
            singularized = singularize_cache[concept]
        else:
            singularized = singularize_nouns(concept)
            singularize_cache[concept] = singularized
        if singularized in concepts_set:  # Using the set for O(1) lookup
            singularized_concepts.add(singularized)
            if singularized != concept:
                duplicates[singularized].add(concept)
        else:
            singularized_concepts.add(concept)

    tokenized_concepts = set()
    tokenized_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    tokenize_cache = {}  # --- Changed: Cache for tokenize_text ---
    for concept in tqdm(list(singularized_concepts), desc="Tokenizing concepts"):
        if concept in tokenize_cache:
            tokens = tokenize_cache[concept]
        else:
            tokens = tokenize_text(concept)
            tokenize_cache[concept] = tokens
        tokenized_groups[tokens].append(concept)
    for tokenized_group in tqdm(tokenized_groups, desc="Selecting preferred formatting options"):
        if len(tokenized_groups[tokenized_group]) == 1:
            tokenized_concepts.add(tokenized_groups[tokenized_group][0])
        else:
            preferred_concept = [concept for concept in tokenized_groups[tokenized_group] if "-" in concept]
            if preferred_concept:
                preferred_concept = preferred_concept[0]
            else:
                preferred_concept = tokenized_groups[tokenized_group][0]
            tokenized_concepts.add(preferred_concept)
            for concept in tokenized_groups[tokenized_group]:
                if concept != preferred_concept:
                    duplicates[preferred_concept].add(concept)

    potential_acronyms = set()
    acronyms = set()
    for concept in tqdm(list(tokenized_concepts), desc="Extracting acronyms"):
        if is_potential_acronym(concept):
            potential_acronyms.add(concept)

        full_form, potential_acronym = extract_acronyms(concept)
        if full_form and check_acronym(full_form, potential_acronym):
            acronyms.add(potential_acronym)
            duplicates[f"{full_form} ({potential_acronym})"].add(potential_acronym)
            duplicates[f"{full_form} ({potential_acronym})"].add(full_form)

    candidates_index = build_candidates_index(list(singularized_concepts))
    for potential_acronym in tqdm(list(potential_acronyms), desc="Finding full forms for acronyms"):
        if potential_acronym in acronyms:
            continue
        full_form = find_full_form(potential_acronym, candidates_index)
        if full_form:
            duplicates[f"{full_form} ({potential_acronym})"].add(potential_acronym)
            duplicates[f"{full_form} ({potential_acronym})"].add(full_form)

    merge_keys(duplicates)
    return duplicates

class GraphDeduplicator:
    def __init__(
        self,
        duplicates,
        entities_path,
        relationships_path,
        text_units_path,
        graphml_in_path,
        out_entities_path,
        out_relationships_path,
        out_text_units_path,
        graphml_out_path,
        vectorstore_config: dict = None,
        new_vectorstore_config: dict = None,
    ):
        """
        Initializes the deduplicator with input file paths, a duplicates mapping, and vector store configurations.
        
        Args:
            duplicates (dict): Mapping from canonical entity names to a set of duplicate names.
            entities_path (str): Input parquet file path for entities.
            relationships_path (str): Input parquet file path for relationships.
            text_units_path (str): Input parquet file path for text units.
            graphml_in_path (str): Input GraphML file path.
            out_entities_path (str): Output parquet file path for deduplicated entities.
            out_relationships_path (str): Output parquet file path for deduplicated relationships.
            out_text_units_path (str): Output parquet file path for updated text units.
            graphml_out_path (str): Output GraphML file path.
            vectorstore_config (dict, optional): Configuration for the existing vector store (e.g. {"db_uri": ..., "collection_name": ...}).
            new_vectorstore_config (dict, optional): Configuration for the new vector store connection.
        """
        self.duplicates = duplicates
        self.entities_path = entities_path
        self.relationships_path = relationships_path
        self.text_units_path = text_units_path
        self.graphml_in_path = graphml_in_path
        self.out_entities_path = out_entities_path
        self.out_relationships_path = out_relationships_path
        self.out_text_units_path = out_text_units_path
        self.graphml_out_path = graphml_out_path

        # Use provided vector store configs or default values.
        self.vectorstore_config = vectorstore_config or {
            "db_uri": "graphrag/output/lancedb",
            "collection_name": "default-entity-description"
        }
        self.new_vectorstore_config = new_vectorstore_config or {
            "db_uri": "graphrag/output/lancedb"
        }

        self.entity_canonical_map = self.build_entity_canonical_map(self.duplicates)
        self.old_to_new_entity_id_map = {}
        self.old_to_new_relationship_id_map = {}
    
    def build_entity_canonical_map(self, duplicates):
        mapping = {}
        for canonical, dup_set in duplicates.items():
            mapping[canonical] = canonical
            for dup in dup_set:
                mapping[dup] = canonical
        return mapping

    def deduplicate_entities(self):
        entities_df = pd.read_parquet(self.entities_path)
        entities_df["canonical_title"] = entities_df["title"].apply(
            lambda x: self.entity_canonical_map.get(x, x)
        )

        def merge_entities(group):
            canonical = group["canonical_title"].iloc[0]
            rep_rows = group[group["title"] == canonical]
            rep = rep_rows.iloc[0] if not rep_rows.empty else group.iloc[0]
            for _, row in group.iterrows():
                self.old_to_new_entity_id_map[row["id"]] = rep["id"]
            merged_description = " ".join(group["description"].dropna().astype(str).unique())
            merged_text_units = set()
            for ids in group["text_unit_ids"]:
                if isinstance(ids, np.ndarray):
                    merged_text_units.update(ids.tolist())
                elif isinstance(ids, list):
                    merged_text_units.update(ids)
            aliases = list(set(group["title"].unique()) - {canonical})
            return pd.Series({
                "id": rep["id"],
                "human_readable_id": rep["human_readable_id"],
                "title": canonical,
                "type": rep["type"],
                "description": merged_description,
                "text_unit_ids": list(merged_text_units),
                "aliases": aliases
            })

        dedup_entities_df = entities_df.groupby("canonical_title", group_keys=False).apply(merge_entities).reset_index(drop=True)
        self.dedup_entities_df = dedup_entities_df

    def deduplicate_relationships(self):
        relationships_df = pd.read_parquet(self.relationships_path)
        relationships_df["new_source"] = relationships_df["source"].apply(
            lambda x: self.entity_canonical_map.get(x, x)
        )
        relationships_df["new_target"] = relationships_df["target"].apply(
            lambda x: self.entity_canonical_map.get(x, x)
        )

        def merge_relationships(group):
            rep = group.iloc[0]
            for _, row in group.iterrows():
                self.old_to_new_relationship_id_map[row["id"]] = rep["id"]
            merged_description = " ".join(group["description"].dropna().astype(str).unique())
            merged_text_units = set()
            for ids in group["text_unit_ids"]:
                if isinstance(ids, np.ndarray):
                    merged_text_units.update(ids.tolist())
                elif isinstance(ids, list):
                    merged_text_units.update(ids)
            return pd.Series({
                "id": rep["id"],
                "human_readable_id": rep["human_readable_id"],
                "source": group["new_source"].iloc[0],
                "target": group["new_target"].iloc[0],
                "description": merged_description,
                "weight": group["weight"].sum(),
                "combined_degree": group["combined_degree"].sum(),
                "text_unit_ids": list(merged_text_units)
            })

        dedup_relationships_df = relationships_df.groupby(["new_source", "new_target"], group_keys=False).apply(merge_relationships).reset_index(drop=True)
        self.dedup_relationships_df = dedup_relationships_df

    def update_text_units(self):
        text_units_df = pd.read_parquet(self.text_units_path)

        def update_ids(id_list, mapping):
            if isinstance(id_list, (list, np.ndarray)):
                if isinstance(id_list, np.ndarray):
                    id_list = id_list.tolist()
                new_ids = {mapping.get(eid, eid) for eid in id_list}
                return list(new_ids)
            return id_list

        if "entity_ids" in text_units_df.columns:
            text_units_df["entity_ids"] = text_units_df["entity_ids"].apply(
                lambda x: update_ids(x, self.old_to_new_entity_id_map)
            )
        if "relationships_ids" in text_units_df.columns:
            text_units_df["relationships_ids"] = text_units_df["relationships_ids"].apply(
                lambda x: update_ids(x, self.old_to_new_relationship_id_map)
            )
        self.text_units_df = text_units_df

    def deduplicate_graphml(self):
        G = nx.read_graphml(self.graphml_in_path)
        edge_weight_lookup = {}
        for _, row in self.dedup_relationships_df.iterrows():
            key = tuple(sorted([row["source"], row["target"]]))
            edge_weight_lookup[key] = row["weight"]

        new_G = nx.Graph()
        for node, data in G.nodes(data=True):
            canonical = self.entity_canonical_map.get(node, node)
            if not new_G.has_node(canonical):
                new_G.add_node(canonical)
        for u, v, data in G.edges(data=True):
            canonical_u = self.entity_canonical_map.get(u, u)
            canonical_v = self.entity_canonical_map.get(v, v)
            if canonical_u == canonical_v:
                continue
            key = tuple(sorted([canonical_u, canonical_v]))
            weight = edge_weight_lookup.get(key, 1)
            if new_G.has_edge(canonical_u, canonical_v):
                new_G[canonical_u][canonical_v]["weight"] += weight
            else:
                new_G.add_edge(canonical_u, canonical_v, weight=weight)
        nx.write_graphml(new_G, self.graphml_out_path)

    def update_vectorstore(self, text_embedder: TextEmbedder) -> None:
        """
        Updates the existing vector store for entity descriptions by computing new embeddings for
        merged (canonical) entity descriptions and overwriting the collection.
        
        Args:
            text_embedder (TextEmbedder): A callable that takes a string and returns its embedding (list[float]).
        """
        db_uri = self.vectorstore_config.get("db_uri")
        collection_name = self.vectorstore_config.get("collection_name")
        vectorstore = LanceDBVectorStore(collection_name=collection_name)
        vectorstore.connect(db_uri=db_uri)
        
        documents = []
        for _, row in self.dedup_entities_df.iterrows():
            # If the entity was not merged (aliases empty), attempt to copy its existing embedding.
            if not row["aliases"]:
                try:
                    existing_doc = vectorstore.search_by_id(row["id"])
                    vector = existing_doc.vector if existing_doc.vector is not None else text_embedder.embed_query(row["description"])
                except Exception:
                    vector = text_embedder.embed_query(row["description"])
            else:
                vector = text_embedder.embed_query(row["description"])
            doc = VectorStoreDocument(
                id=row["id"],
                text=row["description"],
                vector=vector,
                attributes={
                    "title": row["title"],
                    "aliases": row["aliases"],
                    "type": row["type"]
                }
            )
            documents.append(doc)
        vectorstore.load_documents(documents, overwrite=True)

    def update_new_vectorstore(self, text_embedder: TextEmbedder, new_collection_name: str) -> None:
        """
        Creates a new vector store (using a new collection name) to store the embeddings of merged
        entity descriptions. For entities that were not merged (i.e. have empty aliases), their embeddings
        are copied from the existing vector store.
        
        Args:
            text_embedder (TextEmbedder): A callable that takes a string and returns its embedding (list[float]).
            new_collection_name (str): The name of the new collection to store the updated embeddings.
        """
        db_uri_new = self.new_vectorstore_config.get("db_uri")
        new_vectorstore = LanceDBVectorStore(collection_name=new_collection_name)
        new_vectorstore.connect(db_uri=db_uri_new)
        
        # Connect to the existing vector store for lookups.
        db_uri_old = self.vectorstore_config.get("db_uri")
        existing_collection = self.vectorstore_config.get("collection_name")
        old_vectorstore = LanceDBVectorStore(collection_name=existing_collection)
        old_vectorstore.connect(db_uri=db_uri_old)
        
        documents = []
        for _, row in self.dedup_entities_df.iterrows():
            if not row["aliases"]:
                try:
                    existing_doc = old_vectorstore.search_by_id(row["id"])
                    vector = existing_doc.vector if existing_doc.vector is not None else text_embedder.embed_query(row["description"])
                except Exception:
                    vector = text_embedder.embed_query(row["description"])
            else:
                vector = text_embedder.embed_query(row["description"])
            doc = VectorStoreDocument(
                id=row["id"],
                text=row["description"],
                vector=vector,
                attributes={
                    "title": row["title"],
                    "aliases": row["aliases"],
                    "type": row["type"]
                }
            )
            documents.append(doc)
        new_vectorstore.load_documents(documents, overwrite=True)

    def write_outputs(self):
        self.dedup_entities_df.to_parquet(self.out_entities_path, index=False)
        self.dedup_relationships_df.to_parquet(self.out_relationships_path, index=False)
        self.text_units_df.to_parquet(self.out_text_units_path, index=False)

    def run(self):
        self.deduplicate_entities()
        self.deduplicate_relationships()
        self.update_text_units()
        self.write_outputs()

    def run_graphml(self):
        self.deduplicate_graphml()


if __name__ == "__main__":
    concepts_duplicated = pd.read_parquet("graphrag/output/create_final_entities.parquet")["title"].tolist()
    duplicates = concept_deduplication(concepts_duplicated)
    duplicates_reversed = {}
    for key, values in duplicates.items():
        for value in values:
            if value not in duplicates_reversed:
                duplicates_reversed[value] = key
    with open("postprocessing/duplicates_reversed.json", "w") as json_file:
        json.dump(duplicates_reversed, json_file, indent=4)
    
    deduplicator = GraphDeduplicator(
        duplicates=duplicates,
        entities_path="graphrag/output/create_final_entities.parquet",
        relationships_path="graphrag/output/create_final_relationships.parquet",
        text_units_path="graphrag/output/create_final_text_units.parquet",
        graphml_in_path="graphrag/output/graph.graphml",
        out_entities_path="postprocessing/dedup_create_final_entities.parquet",
        out_relationships_path="postprocessing/dedup_create_final_relationships.parquet",
        out_text_units_path="postprocessing/dedup_create_final_text_units.parquet",
        graphml_out_path="postprocessing/dedup_graph.graphml",
        vectorstore_config={
            "db_uri": "graphrag/output/lancedb",
            "collection_name": "default-entity-description"
        },
        new_vectorstore_config={
            "db_uri": "graphrag/output/lancedb"
        }
    )
    deduplicator.run()
    deduplicator.run_graphml()

    deduplicator.update_new_vectorstore(text_embedder, new_collection_name="dedup-entity-description")
