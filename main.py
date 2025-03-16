#!/usr/bin/env python3
"""
Main demo script for MSTFT (MetaScaffold TaxoFrame Template) method.
This script demonstrates taxonomy construction using LLM-driven meta-scaffolding.
It loads the meta-structure from a JSON file, builds indexes for taxonomy data and meta-prompts,
executes a graph-based workflow to construct and refine a taxonomy, and then prints the results.
"""

import os
import json
import pickle
import datetime
import re
import numpy as np
import openai

# --- Set up API key ---
api_key = "your-api-key-here"  # Replace with your actual OpenAI API key.
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = os.environ["OPENAI_API_KEY"]

# --- Import MSTFT Core Modules ---
from mstftt_core.mstft_config import get_client_config
from mstftt_core.mstft_metadata import MSTFTMetadata
from mstftt_core.mstft_embeddings import get_embedding, calculate_similarity

# --- Import LangChain and Graph Framework Modules ---
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from typing import TypedDict, List

###############################################################################
# Helper Functions and Classes (based on langgraph_logic.py)
###############################################################################

def ensure_directory_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Directory created: {path}")
        except OSError as error:
            print(f"Error creating directory: {error}")

# --- Taxonomy and Concept Classes ---

class Concept:
    def __init__(self, concept_name: str, parent=None, taxonomical_rank='root', taxonomical_level=0, taxonomical_ranks_list_number='root') -> None:
        self.name = concept_name
        self.descriptions = []
        self.definitions = []
        self.definition = ""
        self.children = []
        self.parent = parent
        self.taxonomical_rank = taxonomical_rank
        self.taxonomical_level = taxonomical_level
        self.taxonomical_ranks_list_number = taxonomical_ranks_list_number

    def info(self, i=0) -> None:
        print('\n' + '-+' * i + f"Concept\nname: {self.name}")
        if self.definition:
            print(f"definition: {self.definition}")
        print(f"taxonomical rank: {self.taxonomical_rank} | level: {self.taxonomical_level} | rank number: {self.taxonomical_ranks_list_number}")
        if self.definitions:
            print(f"has {len(self.definitions)} definitions: {self.definitions}")
        if self.descriptions:
            print(f"has {len(self.descriptions)} descriptions: {self.descriptions}")
        i += 1
        if self.parent:
            print('---' * i + f"is subconcept of: {self.parent.name}")
        if self.children:
            print('---' * i + f"has {len(self.children)} subconcepts:")
            for child in self.children:
                child.info(i=i)

    def get_semantic_cotopy(self):
        hypernyms = []
        hyponyms = []
        def get_hypernyms(self, hypernyms):
            if self.parent:
                hypernyms.append(self.parent)
                get_hypernyms(self.parent, hypernyms)
        def get_hyponyms(self, hyponyms):
            if self.children:
                hyponyms += self.children
                for child in self.children:
                    get_hyponyms(child, hyponyms)
        get_hypernyms(self, hypernyms)
        get_hyponyms(self, hyponyms)
        return hypernyms, hyponyms

class Taxonomy:
    def __init__(self, root_concept_name: str) -> None:
        self.created_at = datetime.datetime.now()
        self.last_edit_time = datetime.datetime.now()
        self.name = 'Taxonomy_' + str(self.created_at).replace(' ', '_T').replace(':', '-')[:22]
        self.save_path = os.path.join(os.getcwd(), "data", "taxonomies")
        self.token_usage = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
        self.root = Concept(root_concept_name)
        self.property_groups = ""
        self.key_aspects = ""
        self.rare_info = ""
        self.hierarchies = []
        self.present_features = ""
        self.missing_features = ""
        self.save()

    def update_last_edit_time(self) -> None:
        self.last_edit_time = datetime.datetime.now()

    def update_token_usage(self, token_usage) -> None:
        for key in self.token_usage:
            self.token_usage[key] += token_usage.get(key, 0)

    def save(self, suffix="") -> None:
        ensure_directory_exists(self.save_path)
        full_path = os.path.join(self.save_path, self.name + suffix + '.pkl')
        with open(full_path, 'wb') as file:
            pickle.dump(self, file)
        self.saved_to = full_path

    def update_hierarchies(self, new_hierarchies) -> None:
        for hierarchy in new_hierarchies:
            new_hierarchy_dict = {
                "name": None,  # will be updated later
                "description": hierarchy,
                "properties": [],
                "taxonomical ranks": [],
                "concepts": []
            }
            self.hierarchies.append(new_hierarchy_dict)
        self.update_last_edit_time()
        self.save()

    def get_hierarchy_descriptions(self):
        return [h.get("description", "No description available") for h in self.hierarchies]

    def update_hierarchies_feature(self, values, key):
        try:
            for i, value in enumerate(values):
                if i < len(self.hierarchies):
                    self.hierarchies[i][key] = value
                else:
                    break
            if len(values) < len(self.hierarchies):
                print(f"Warning: \"{key}\" list is shorter than number of hierarchies.")
        except Exception as e:
            print(f"Error updating hierarchies: {e}")
        self.update_last_edit_time()
        self.save()

    def info(self) -> None:
        print("--------TAXONOMY INFO:-------")
        print(f"Root concept: \"{self.root.name}\"")
        print(f"Created at: {self.created_at}")
        print(f"Last edited at: {self.last_edit_time}")
        print(f"Name: {self.name}")
        print(f"Save path: {self.save_path}")
        print(f"Token usage: {self.token_usage}")
        print(f"Saved to: {self.saved_to}")
        print("\nProperty Groups:", self.property_groups)
        print("Key Aspects:", self.key_aspects)
        print("Rare Info:", self.rare_info)
        print("Hierarchies:")
        for i, hierarchy in enumerate(self.hierarchies):
            print(f"{i+1}. Name: {hierarchy.get('name')}, Description: {hierarchy.get('description')}, Properties: {hierarchy.get('properties')}")
        print("Present Features:", self.present_features)
        print("Missing Features:", self.missing_features)
        print("--------END OF INFO--------")

###############################################################################
# Model and Prompt Templates
###############################################################################

class Model:
    def __init__(self, name: str, model_checkpoint: str, temperature=1, top_p=1, presence_penalty=1, frequency_penalty=0) -> None:
        self.model = ChatOpenAI(
            model=model_checkpoint,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )
        self.name = name
        self.temperature = temperature
        self.model_checkpoint = model_checkpoint
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.info = f"model name: {name}\nmodel checkpoint: {model_checkpoint}\ntemperature: {temperature}\ntop p: {top_p}\npresence penalty: {presence_penalty}\nfrequency penalty: {frequency_penalty}"

# --- Define a simplified set of chat prompt templates ---
chat_templates = {}
chat_templates['get_property_groups'] = ChatPromptTemplate.from_messages(
    [("system", '''Role: You are an expert.
Task: For the concept "{root_concept}", list property groups separated by commas.
Return only the group names.
'''), ("human", "")])
chat_templates['get_key_aspects'] = ChatPromptTemplate.from_messages(
    [("system", '''Role: Provide a semicolon-separated list of key aspects for "{root_concept}".
'''), ("human", "")])
chat_templates['get_rare_info'] = ChatPromptTemplate.from_messages(
    [("system", '''Role: Provide a semicolon-separated list of rare taxonomical features for "{root_concept}".
'''), ("human", "")])
chat_templates['get_initial_hierarchies'] = ChatPromptTemplate.from_messages(
    [("system", '''Role: Generate semicolon-separated hierarchy descriptions for "{root_concept}" using property groups: "{properties}".
Format: Hierarchy 1: [description]; Hierarchy 2: [description];
'''), ("human", "")])
chat_templates['find_missing_hierarchies'] = ChatPromptTemplate.from_messages(
    [("system", '''Role: Identify additional hierarchy descriptions for "{root_concept}" based on current hierarchies:
{current_hierarchies}
Return semicolon-separated descriptions.
'''), ("human", "")])
# (Additional prompt templates can be added similarly if needed.)

###############################################################################
# Indexing and Retrieval Functions
###############################################################################

def build_index_from_section(section, parent_text="", parent_path=""):
    records = []
    name = section.get("name", "")
    description = section.get("description", "")
    value = section.get("value", "")
    current_text = f"{name}: {description} {value}".strip()
    combined_text = (parent_text + " " if parent_text else "") + current_text
    full_path = (parent_path + " > " if parent_path else "") + name
    embedding = get_embedding(combined_text)
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()
    record = {
        "path": full_path,
        "name": name,
        "description": description,
        "value": value,
        "full_text": combined_text,
        "embedding": embedding
    }
    records.append(record)
    if "subsections" in section:
        for sub in section["subsections"]:
            records.extend(build_index_from_section(sub, parent_text=combined_text, parent_path=full_path))
    return records

def build_index_for_top_level(meta_scaffold, key, index_file_path):
    top_level_section = meta_scaffold.get(key, {})
    records = []
    if "subsections" in top_level_section:
        for section in top_level_section["subsections"]:
            records.extend(build_index_from_section(section))
    else:
        records.extend(build_index_from_section(top_level_section))
    with open(index_file_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return records

class BaseRetriever:
    def __init__(self, index_file_path):
        self.index_file_path = index_file_path
        with open(index_file_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)
    def search(self, query, top_k=3):
        query_emb = get_embedding(query)
        scored = []
        for record in self.index:
            rec_emb = np.array(record["embedding"])
            sim = cosine_similarity(query_emb, rec_emb)
            scored.append((sim, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for sim, record in scored[:top_k]]

class DataRetriever(BaseRetriever):
    pass

class RuleRetriever(BaseRetriever):
    def search_best(self, query):
        results = self.search(query, top_k=1)
        return results[0] if results else None

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

###############################################################################
# Graph-Based Workflow Functions
###############################################################################

# Define a TypedDict for graph state
class GraphState(TypedDict):
    taxonomy: 'Taxonomy'
    messages: List[str]

# The following functions implement steps in the MSTFT workflow.
def get_property_groups(state: GraphState):
    taxonomy = pickle.loads(state['taxonomy'])
    print("Executing: Get Property Groups")
    prompt = chat_templates['get_property_groups'].format_messages(root_concept=taxonomy.root.name)
    response = model.invoke(prompt)
    taxonomy.property_groups = response.content.strip().replace('\n', ' ')
    taxonomy.update_last_edit_time()
    taxonomy.update_token_usage(response.response_metadata.get('token_usage', {}))
    taxonomy.save()
    state['messages'].append(f"Property groups fetched: {response.content}")
    print(f"Property groups: {taxonomy.property_groups}")
    state['taxonomy'] = pickle.dumps(taxonomy)
    return state

def get_key_aspects(state: GraphState):
    taxonomy = pickle.loads(state['taxonomy'])
    prompt = chat_templates['get_key_aspects'].format_messages(root_concept=taxonomy.root.name)
    response = model.invoke(prompt)
    taxonomy.key_aspects = response.content.strip()
    taxonomy.update_last_edit_time()
    taxonomy.update_token_usage(response.response_metadata.get('token_usage', {}))
    taxonomy.save()
    state['messages'].append(f"Key aspects fetched: {response.content}")
    print(f"Key aspects: {taxonomy.key_aspects}")
    state['taxonomy'] = pickle.dumps(taxonomy)
    return state

def get_rare_info(state: GraphState):
    taxonomy = pickle.loads(state['taxonomy'])
    prompt = chat_templates['get_rare_info'].format_messages(root_concept=taxonomy.root.name)
    response = model.invoke(prompt)
    taxonomy.rare_info = response.content.strip()
    taxonomy.update_last_edit_time()
    taxonomy.update_token_usage(response.response_metadata.get('token_usage', {}))
    taxonomy.save()
    state['messages'].append(f"Rare info fetched: {response.content}")
    print(f"Rare info: {taxonomy.rare_info}")
    state['taxonomy'] = pickle.dumps(taxonomy)
    return state

def construct_initial_hierarchies(state: GraphState):
    taxonomy = pickle.loads(state['taxonomy'])
    prompt = chat_templates['get_initial_hierarchies'].format_messages(root_concept=taxonomy.root.name, properties=taxonomy.property_groups)
    response = model.invoke(prompt)
    # For demo, assume the response content is a semicolon-separated list of hierarchy descriptions.
    new_hierarchies = [desc.strip() for desc in response.content.split(';') if desc.strip()]
    taxonomy.update_hierarchies(new_hierarchies)
    # For simplicity, set the hierarchy name equal to its description prefix.
    names = [desc.split(':')[0].strip() for desc in new_hierarchies]
    taxonomy.update_hierarchies_feature(names, "name")
    taxonomy.update_last_edit_time()
    taxonomy.update_token_usage(response.response_metadata.get('token_usage', {}))
    taxonomy.save()
    state['messages'].append(f"Initial hierarchies constructed: {response.content}")
    print(f"Initial hierarchies: {taxonomy.hierarchies}")
    state['taxonomy'] = pickle.dumps(taxonomy)
    return state

# Additional functions (find_missing_hierarchies, find_present_features, etc.) can be defined similarly.
# For brevity, we assume the initial workflow is sufficient for this demo.

###############################################################################
# Build and Execute Graph Workflow
###############################################################################

# Create the state graph and add nodes (using functions defined above)
graph = StateGraph(GraphState)
graph.add_node("Get Property Groups", get_property_groups)
graph.add_node("Get Key Aspects", get_key_aspects)
graph.add_node("Get Rare Info", get_rare_info)
graph.add_node("Construct Initial Hierarchies", construct_initial_hierarchies)
graph.add_edge(START, "Get Property Groups")
graph.add_edge("Get Property Groups", "Get Key Aspects")
graph.add_edge("Get Key Aspects", "Get Rare Info")
graph.add_edge("Get Rare Info", "Construct Initial Hierarchies")
graph.add_edge("Construct Initial Hierarchies", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

def build_taxonomy(taxonomy: Taxonomy):
    initial_state: GraphState = {
        "taxonomy": pickle.dumps(taxonomy),
        "messages": [HumanMessage(content="Start").content]
    }
    final_state = app.invoke(
        {"taxonomy": initial_state["taxonomy"], "messages": initial_state["messages"]},
        config={"configurable": {"thread_id": 42}}
    )
    deserialized_taxonomy = pickle.loads(final_state["taxonomy"])
    print(final_state["messages"][-1])
    return deserialized_taxonomy

def process_concept(concept_name: str):
    taxonomy = Taxonomy(concept_name)
    result = build_taxonomy(taxonomy)
    result.info()
    return result

###############################################################################
# Main Demo Execution
###############################################################################

if __name__ == "__main__":
    # Load meta-structure JSON from data folder.
    meta_structure_path = os.path.join(os.path.dirname(__file__), "data", "meta-structure.json")
    if not os.path.exists(meta_structure_path):
        print(f"Meta structure file not found at {meta_structure_path}")
        exit(1)
    with open(meta_structure_path, "r", encoding="utf-8") as f:
        meta_structure = json.load(f)
    
    # Build indexes for taxonomy data and meta-prompts (optional demonstration)
    data_index_file = os.path.join(os.path.dirname(__file__), "data_index.json")
    rules_index_file = os.path.join(os.path.dirname(__file__), "rules_index.json")
    print("Building data index...")
    build_index_for_top_level(meta_structure, "data", data_index_file)
    print("Building rules index...")
    build_index_for_top_level(meta_structure, "metaPrompts", rules_index_file)
    
    # Initialize the LLM model for prompting.
    llm = Model('taxonomy construction', 'gpt-4o', temperature=0.8, top_p=0.90, presence_penalty=0.80, frequency_penalty=0.30)
    model = llm.model

    # Process a target concept using the MSTFT workflow.
    target_concept = "Lumber Wood"
    taxonomy = process_concept(target_concept)
    
    # Demonstrate retrieval from built indexes.
    data_retriever = DataRetriever(data_index_file)
    rule_retriever = RuleRetriever(rules_index_file)
    query = "How to ensure there are no cyclic dependencies in the taxonomy?"
    print("\n=== Data Retrieval Results ===")
    data_results = data_retriever.search(query, top_k=3)
    print(json.dumps(data_results, indent=2))
    print("\n=== Best Rule Retrieval Result ===")
    rule_result = rule_retriever.search_best(query)
    print(json.dumps(rule_result, indent=2))
