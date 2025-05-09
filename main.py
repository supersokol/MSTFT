#!/usr/bin/env python3
"""
Main demo script for MSTFT (MetaScaffold TaxoFrame Template) method.
This script demonstrates taxonomy construction using LLM-driven meta-scaffolding.
It loads the meta-structure from a JSON file, builds indexes for taxonomy data and meta-prompts,
executes a graph-based workflow to construct and refine a taxonomy, and then prints the results.
"""
import os
import json
from dotenv import load_dotenv
import openai
from mstftt_core.mstft_workflow import (
    build_index_for_top_level,
    DataRetriever,
    RuleRetriever,
    process_concept_improved
)
from mstftt_core.mstft_model import model

# --- Set up API key ---
#api_key = "your-api-key-here"  # Replace with your actual OpenAI API key.
#os.environ["OPENAI_API_KEY"] = api_key
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

###############################################################################
# Main Demo Execution
###############################################################################

if __name__ == "__main__":
    # Load meta-structure JSON
    meta_structure_path = os.path.join(os.path.dirname(__file__), "data", "meta-structure.json")
    if not os.path.exists(meta_structure_path):
        print(f"Meta structure file not found at {meta_structure_path}")
        exit(1)

    with open(meta_structure_path, "r", encoding="utf-8") as f:
        meta_structure = json.load(f)

    # Build indexes for taxonomy data and meta-prompts
    data_index_file = os.path.join(os.path.dirname(__file__), "data_index.json")
    rules_index_file = os.path.join(os.path.dirname(__file__), "rules_index.json")
    print("Building data index...")
    build_index_for_top_level(meta_structure, "data", data_index_file)
    print("Building rules index...")
    build_index_for_top_level(meta_structure, "metaPrompts", rules_index_file)

    # Initialize retrievers
    data_retriever = DataRetriever(data_index_file)
    rule_retriever = RuleRetriever(rules_index_file)

    

    # Process the target concept using the MSTFT workflow
    target_concept = "Software"
    taxonomy = process_concept_improved(target_concept, data_retriever, rule_retriever, model, levels=3)
    

'''    # Demonstrate retrieval capabilities
    query = "How to ensure there are no cyclic dependencies in the taxonomy?"
    print("\n=== Data Retrieval Results ===")
    data_results = data_retriever.search(query, top_k=3)
    print(json.dumps(data_results, indent=2))

    print("\n=== Best Rule Retrieval Result ===")
    rule_result = rule_retriever.search_best(query)
    print(json.dumps(rule_result, indent=2))'''
