from calendar import c
import os
import json
import pickle
import datetime
import numpy as np




# --- Import MSTFT Core Modules ---
from mstftt_core.mstft_model import model
from mstftt_core.mstft_config import get_client_config
from mstftt_core.mstft_metadata import MSTFTMetadata
from mstftt_core.mstft_embeddings import get_embedding, calculate_similarity

# --- Import LangChain and Graph Framework Modules ---
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List

###############################################################################
# Helper Functions and Classes
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
        self.hierarchies = ""
        self.present_features = ""
        self.missing_features = ""
        self.subconcepts_tree = {}
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

    # Adds subconcepts tree for a specific hierarchy
    def add_subconcepts(self, subconcepts_tree):#(self, hierarchy_index, subconcepts_tree):
        self.subconcepts_tree = subconcepts_tree
        self.update_last_edit_time()
        self.save()
        #if hierarchy_index < len(self.hierarchies):
        #    self.hierarchies[hierarchy_index]['subconcepts_tree'] = subconcepts_tree
        #    self.update_last_edit_time()
        #    self.save()

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
        print(self.hierarchies)
        print("Present Features:", self.present_features)
        print("Missing Features:", self.missing_features)
        print("--------END OF INFO--------")

    # Prints subconcepts tree nicely
    def print_subconcepts_tree(self):
        print("\n------ Final Subconcepts Trees ------")
        print(self.subconcepts_tree)
    
    
###############################################################################
# Prompt Templates
###############################################################################


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

Rules:
Each hierarchy description must be a single, concise block of text.
Each description must begin with the hierarchy’s name, followed by a colon (e.g., "Hierarchy 1: Evolutionary Classification: This hierarchy classifies biological species based on...").
The description should cover what the hierarchy deals with, its purpose, key property groups, and criteria used for classification (without specific examples). End with a brief summary of key criteria.
Keep each description under 50 tokens.
Never use semicolons (;)a nd newlines (\\n) in the descriptions.
Output only new hierarchies, omitting any existing ones.
Stop generating if irrelevant or nonsensical content is produced.
Always separate the hierarchies by semicolons (;) in the output.
Provide no additional text besides the list of hierarchies.

Format:
Hierarchy X: [description]; Hierarchy X+1: [next description];
             
Instructions:
Continue in this format for each hierarchy. Generate as many semicolon-separated hierarchies as necessary and finish with the new line.
'''), ("human", "")])
chat_templates['find_missing_hierarchies'] = ChatPromptTemplate.from_messages(
    [("system", '''Role: Identify additional hierarchy descriptions for "{root_concept}" based on current hierarchies:
{current_hierarchies}
Return semicolon-separated descriptions.
'''), ("human", "")])
chat_templates["find_additional_hierarchies"] = ChatPromptTemplate.from_messages(
        [
            ("system", '''Role: You are an expert in conceptual modeling and taxonomy.

Task:
You are provided with the current hierarchical classification descriptions for the root concept "{root_concept}".
You are also given new information that may affect the classification of "{root_concept}".
The goal is to improve the existing structure and move towards a well-structured taxonomy that fully classifies the root concept and its sub-concepts across multiple levels, either by confirming the sufficiency of the existing hierarchies or introducing new ones.
Examine the current hierarchies in light of the new information, and evaluate whether additional hierarchies are required based on this analysis.
If new hierarchies are required, identify the minimum number of new hierarchies needed to account for the newly provided information and maintain a coherent classification, then formulate and describe them.
If no new hierarchies are necessary, return an empty response.

Rules:
Each hierarchy description must be a single, concise block of text.
Each description must begin with the hierarchy’s name, followed by a colon (e.g., "Hierarchy 1: Evolutionary Classification: This hierarchy classifies biological species based on...").
The description should cover what the hierarchy deals with, its purpose, key property groups, and criteria used for classification (without specific examples). End with a brief summary of key criteria.
Keep each description under 50 tokens.
Never use semicolons (;)a nd newlines (\\n) in the descriptions.
Output only new hierarchies, omitting any existing ones.
Stop generating if irrelevant or nonsensical content is produced.
Always separate the hierarchies by semicolons (;) in the output.
Provide no additional text besides the list of hierarchies.

Current Hierarchies:
{current_hierarchies}

New information:
{context}

Format:
Hierarchy X: [description]; Hierarchy X+1: [next description];
             
Instructions:
Continue in this format for each hierarchy. Generate as many semicolon-separated hierarchies as necessary and finish with the new line.
'''
),
            ("human", '''''')
        ]
    )
chat_templates["find_present_features"] = ChatPromptTemplate.from_messages(
        [
            ("system", '''Role: You are a leading authority in scientific taxonomy, with unmatched expertise in identifying and organizing key taxonomic criteria.

Instruction:

You will be provided with a root concept and a series of hierarchical classifications. Your task is to extract and list all the unique taxonomic criteria—the defining properties or qualities used to classify the concept into different levels of abstraction.
Identify only the key taxonomic criteria—the essential properties used for classification.
Do not list examples, instances, categories, or models.
Focus purely on the defining characteristics or criteria that guide the classification of the concept.
Skip any explanations—simply list the criteria.
List all unique taxonomic criteria using a semicolon as a delimiter, in the format below:
criterion 1; criterion 2; criterion 3;

             
Context:

Root concept: {root_concept}
Hierarchies: 
{current_hierarchies}
'''
),
            ("human", '''Start''')
        ]
    )
chat_templates["find_missing_features"] = ChatPromptTemplate.from_messages(
        [
            ("system", '''Role: You are a highly skilled scientific expert. 

Instructions:
I will provide you with a list of properties covered by existing hierarchies. Based on this list, identify properties that differ the most from the covered ones and that could introduce entirely new aspects of the concept.
Skip any explanations in the respond. Provide only highly distinct key properties in the following semicolon separated format like this:
property 1; property 2; property 3
Continue in this format for each found property.

Context:

Root concept: "{root_concept}"
Hierarchies: 
{current_hierarchies}

Covered properties: {properties}
'''
),
            ("human", '''''')
        ]
    )
# from new features
chat_templates["find_additional_hierarchies_for_features"] = ChatPromptTemplate.from_messages(
        [
            ("system", '''Role: You are an expert in conceptual modeling and taxonomy.

Task:
You are provided with the current hierarchical classification descriptions for the root concept "{root_concept}." Additionally, you are given a list of new properties that require integration into the taxonomy. Your objective is to evaluate the current hierarchies in light of these new properties and determine whether they are already covered or whether new hierarchies need to be introduced to account for them.
If the existing hierarchies are sufficient to cover these properties, provide an empty response. Otherwise, identify the minimum number of new hierarchies required to account for the newly provided properties to ensure the full classification of the root concept and its sub-concepts and maintain a coherent classification, then formulate and describe them.

Rules:
Each hierarchy description must be a single, concise block of text.
Each description must begin with the hierarchy’s name, followed by a colon (e.g., "Hierarchy 1: Evolutionary Classification: This hierarchy classifies biological species based on...").
The description should cover what the hierarchy deals with, its purpose, key property groups, and criteria used for classification (without specific examples). End with a brief summary of key criteria.
Keep each description under 50 tokens.
Never use semicolons (;)a nd newlines (\\n) in the descriptions.
Output only new hierarchies, omitting any existing ones.
Stop generating if irrelevant or nonsensical content is produced.
Always separate the hierarchies by semicolons (;) in the output.
Provide no additional text besides the list of hierarchies.

Current Hierarchies:
{current_hierarchies}

New Properties: 
{new_properties}

Format:
Hierarchy X: [description]; Hierarchy X+1: [next description];
             
Instructions:
Continue in this format for each hierarchy. Generate as many semicolon-separated hierarchies as necessary and finish with the new line.
'''
),
            ("human", '''''')
        ]
    )
chat_templates['get_feature_lists'] = ChatPromptTemplate.from_messages(
        [
            ("system",'''You are a top-tier specialist in organizing and analyzing taxonomic data. Your expertise in identifying and organizing key taxonomic features is unparalleled. You are provided with a list of taxonomic features for the "{root_concept}" concept and a generated hierarchy list. Your task is to create independent, non-overlapping, and ordered lists of features, where each list corresponds to a specific hierarchy from the hierarchy list.

CONTEXT:

Taxonomic Features: "{properties}"
Hierarchy List: 
{hierarchies}

INSTRUCTIONS:

For each hierarchy, select the key taxonomic features whose variation defines the placement of concepts within that hierarchy. 
Ensure that:
- Each feature is assigned to only one hierarchy.
- The features are ordered logically, based on their relevance to the hierarchy’s structure.
- Each list must include only key features that act as taxonomic criteria, not categories, groups, labels, or other classifications.
It is essential that the result meets these requirements precisely, as failure to do so may lead to serious consequences.. Double-check your work, as accuracy is critical to the success of the task.
Return the result as ordered lists of features, with each list corresponding to the hierarchy number. Features should be separated by commas, with a semicolon at the end of each list. No brackets or other delimiters should be used. Format the output as follows:
Hierarchy 1: feature1, feature2, feature3;
Hierarchy 2: feature1, feature2, feature3;
...

Finish the output with a newline.
'''
),
            ("human", '''''')
        ]
    )


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
    print(f"Response content: {response.content}\n\n")
    # Extract hierarchies from the response
    taxonomy.hierarchies += f"{response.content}\n"
    taxonomy.update_last_edit_time()
    taxonomy.update_token_usage(response.response_metadata.get('token_usage', {}))
    taxonomy.save()
    state['messages'].append(f"Initial hierarchies constructed: {response.content}")
    print(f"Initial hierarchies: {taxonomy.hierarchies}\n\n")
    state['taxonomy'] = pickle.dumps(taxonomy)
    return state

def find_missing_hierarchies_basic(state: GraphState):
    # Deserialize taxonomy
    taxonomy = pickle.loads(state['taxonomy']) if isinstance(state['taxonomy'], bytes) else state['taxonomy']

    prompt = chat_templates['find_missing_hierarchies'].format_messages(
        root_concept=taxonomy.root.name,
        current_hierarchies='\n'.join(taxonomy.hierarchies)
    )
    response = model.invoke(prompt)
    print(f"Response content: {response.content}\n\n")
    # Extract hierarchies from the response
    taxonomy.hierarchies += f"{response.content}\n"
    token_usage = response.response_metadata['token_usage']
    taxonomy.update_last_edit_time()
    taxonomy.update_token_usage(token_usage)
    taxonomy.save()
    state['messages'].append(f"Missing hierarchies (basic) found: {response.content}")

    # Serialize taxonomy
    state['taxonomy'] = pickle.dumps(taxonomy)
    return  state #{"messages": state['messages'], "taxonomy": state['taxonomy']}

def find_present_features(state: GraphState):
    # Deserialize taxonomy
    taxonomy = pickle.loads(state['taxonomy']) if isinstance(state['taxonomy'], bytes) else state['taxonomy']

    prompt = chat_templates['find_present_features'].format_messages(
        root_concept=taxonomy.root.name,
        current_hierarchies='\n'.join(taxonomy.hierarchies)
    )
    response = model.invoke(prompt)
    taxonomy.present_features = response.content
    token_usage = response.response_metadata['token_usage']
    taxonomy.update_last_edit_time()
    taxonomy.update_token_usage(token_usage)
    taxonomy.save()
    state['messages'].append(f"Present features found: {response.content}")
    print(f"Present features: {taxonomy.present_features}")

    # Serialize taxonomy
    state['taxonomy'] = pickle.dumps(taxonomy)
    return state #{"messages": state['messages'], "taxonomy": state['taxonomy']}

def find_missing_features(state: GraphState):
    # Deserialize taxonomy
    taxonomy = pickle.loads(state['taxonomy']) if isinstance(state['taxonomy'], bytes) else state['taxonomy']

    prompt = chat_templates['find_missing_features'].format_messages(
        root_concept=taxonomy.root.name,
        current_hierarchies='\n'.join(taxonomy.get_hierarchy_descriptions()),
        properties=taxonomy.present_features
    )
    response = model.invoke(prompt)
    taxonomy.missing_features = response.content
    token_usage = response.response_metadata['token_usage']
    taxonomy.update_last_edit_time()
    taxonomy.update_token_usage(token_usage)
    taxonomy.save()
    state['messages'].append(f"Distinct features found: {response.content}")
    print(f"Missing distinct features: {taxonomy.missing_features}")

    # Serialize taxonomy
    state['taxonomy'] = pickle.dumps(taxonomy)
    return state #{"messages": state['messages'], "taxonomy": state['taxonomy']}


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
graph.add_node("Find Missing Hierarchies Basic", find_missing_hierarchies_basic)
graph.add_node("Find Present Features", find_present_features)
graph.add_node("Find Missing Features", find_missing_features)
graph.add_node("Find Present Features 2", find_present_features)

graph.add_edge(START, "Get Property Groups")
graph.add_edge("Get Property Groups", "Get Key Aspects")
graph.add_edge("Get Key Aspects", "Get Rare Info")
graph.add_edge("Get Rare Info", "Construct Initial Hierarchies")
graph.add_edge("Construct Initial Hierarchies", "Find Missing Hierarchies Basic")
graph.add_edge("Find Missing Hierarchies Basic", "Find Present Features") 
graph.add_edge("Find Present Features", END)

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

# Recursively generates subconcepts using data_retriever, rule_retriever, and LLM
def generate_subconcepts(concept, root_concept, context_info, taxonomical_context, data_retriever, rule_retriever, model, depth=3, current_level=1, subconcepts_amount=5):
    if current_level > depth:
        return {}

    # Retrieve relevant context data
    if concept == root_concept:
        data_query = f"Classification criteria and taxonomy structure related to {concept}"
    else:
        data_query = f"Classification criteria and taxonomy structure related to {concept} in the context of {root_concept}"
    data_context = data_retriever.search(data_query, top_k=3)
    data_context_string = "\n".join([f"{item['name']}: {item['description']} - {item['value']}" for item in data_context])
    print(f"Data context for {concept}: {data_context_string}")
    
    # Retrieve relevant validation rules
    if concept == root_concept:
        validation_query = f"Rules for structural and semantic validation of subconcepts for {concept}"
    else:
        validation_query = f"Rules for structural and semantic validation of subconcepts for {concept} in the context of {root_concept}"
    validation_rules = rule_retriever.search(validation_query, top_k=3)
    validation_rules_string = "\n".join([f"{item['name']}: {item['description']} - {item['value']}" for item in validation_rules])
    print(f"Validation rules for {concept}: {validation_rules_string}")
    
    # Complex prompt construction using retrieved data and rules
    prompt_text = f"""
    Role: You are a world-class expert in taxonomical classification specializing in "{root_concept}".

    Task:
    1. Analyze provided taxonomical, hierarchial and classification context and validation rules.
    2. Generate exactly {subconcepts_amount} most relevant, distinctive subconcepts for the given concept "{concept}".
    
    Taxonomical Context:
    {context_info}
    
    Hierarchical Context:
    {taxonomical_context}
    
    Classification Context:
    {data_context_string}

    Validation Rules:
    {validation_rules_string}

    Constraints:
    - Generated subconcepts must strictly belong to the "{root_concept}" taxonomy.
    - Subconcepts must directly follow "{concept}" in the hierarchy (exactly one level lower).
    - Skip any detailed explanations. Return results only as a comma-separated list.

    Generate the list now.
    """

    # Invoke the model
    response = model.invoke([HumanMessage(content=prompt_text)])

    # Parse response
    subconcept_names = [sc.strip() for sc in response.content.split(',') if sc.strip()]
    
    # Initialize validated subconcepts dictionary
    validated_subconcepts = {name: {} for name in subconcept_names}
    print(f"Generated subconcepts for {concept}: {subconcept_names}")
    # Recursive call to generate deeper subconcepts
    for subconcept in validated_subconcepts:
        next_taxonomical_context = f"{taxonomical_context} > {subconcept}"
        validated_subconcepts[subconcept] = generate_subconcepts(
            subconcept,
            root_concept,
            context_info,
            next_taxonomical_context,
            data_retriever,
            rule_retriever,
            model,
            depth,
            current_level + 1,
            subconcepts_amount
        )

    return validated_subconcepts


def process_concept(concept_name: str, data_retriever, rule_retriever, model, subconcepts_amount=10):
    taxonomy = Taxonomy(concept_name)
    result = build_taxonomy(taxonomy)
    result.info()

    root_concept = concept_name
    context_info = f"{result.property_groups}; {result.key_aspects}; {result.rare_info}"

    for idx, hierarchy in enumerate(result.hierarchies):
        hierarchy_name = hierarchy['name']
        hierarchy_description = hierarchy['description']
        taxonomical_context = f"{hierarchy_name} ({hierarchy_description})"

        subconcepts_tree = generate_subconcepts(
            concept=root_concept,
            root_concept=root_concept,
            context_info = context_info,
            taxonomical_context=taxonomical_context,
            data_retriever=data_retriever,
            rule_retriever=rule_retriever,
            model=model,
            depth=3,
            subconcepts_amount=subconcepts_amount
        )

        result.add_hierarchy_subconcepts(idx, subconcepts_tree)

    result.print_subconcepts_tree()
    return result

###############################################################################
# Meta-Prompt Construction and Execution
###############################################################################

def construct_meta_prompt(root_concept, properties, key_aspects, rare_info, hierarchies, present_features, levels, context_data, constraints=""):
    """
    Constructs a meta-prompt for generating hierarchical subconcepts.
    """
    meta_prompt = f"""
    Role: You are a top-tier taxonomy expert focusing on "{root_concept}".

    Task:
    - Analyze provided properties, key aspects, rare info, and hierarchies.
    - Filter and select the most relevant features from provided context data.
    - Prepare structured instructions to generate nested subconcept hierarchies up to {levels} levels with at least 30 subconcepts at each level.

    Root Concept: "{root_concept}"

    Property Groups:
    {properties}

    Key Aspects:
    {key_aspects}

    Rare Info:
    {rare_info}

    Hierarchies:
    {json.dumps(hierarchies, indent=2)}

    Present Features:
    {present_features}

    Context Data:
    {context_data}
    """
    return meta_prompt

def generate_taxonomy_tree(model, meta_prompt):
    """
    Generates taxonomy tree using an LLM based on provided meta-prompt.
    """
    response = model.invoke([SystemMessage(content=meta_prompt)])
    try:
        subconcepts_tree = json.loads(response.content)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        subconcepts_tree = response.content
    return subconcepts_tree

def validate_taxonomy_tree(model, subconcepts_tree, validation_context, constraints=""):
    """
    Validates and improves the taxonomy tree using additional context and validation rules.
    """
    validation_prompt = f"""
    Role: Expert taxonomist in validation and refinement.

    Task:
    - Validate and refine the given taxonomy tree structure based on provided validation context.
    - Correct inaccuracies, structural inconsistencies, or semantic errors.

    Taxonomy Tree:
    {json.dumps(subconcepts_tree, indent=2)}

    Validation Context:
    {validation_context}

    {constraints}

    Return only the corrected nested dictionary structure.
    """
    # Invoke the model for validation
    print(f"Validation prompt: {validation_prompt}")
    response = model.invoke([HumanMessage(content=validation_prompt)])
    try:
        validated_tree = json.loads(response.content)
    except json.JSONDecodeError as e:
        print(f"Validation JSON decoding error: {e}")
        validated_tree = response.content
    return validated_tree

def process_concept_improved(concept_name, data_retriever, rule_retriever, model, levels=3):
    """
    Improved unified function to handle concept taxonomy generation, validation, and refinement.
    """
    taxonomy = Taxonomy(concept_name)
    result = build_taxonomy(taxonomy)
    result.info()

    # Retrieve context for meta-prompt
    data_query = f"How can I generate a taxonomy tree of subconcepts for the concept {concept_name}?"
    rule_query = f"How can I validate the taxonomy tree for the concept {concept_name}?"

    data_context = data_retriever.search(data_query, top_k=3)
    rules_context = rule_retriever.search(rule_query, top_k=3)

    # Exclude 'embedding' from contexts
    def remove_embedding(context):
        return [{k: v for k, v in item.items() if k != 'embedding'} for item in context]

    clean_data_context = remove_embedding(data_context)
    clean_rules_context = remove_embedding(rules_context)

    # Create JSON string without embeddings
    #context_data_string = json.dumps({"data": clean_data_context, "rules": clean_rules_context}, indent=2)

    constraints = """
    Constraints:
    Generate hierarchical nested dictionaries in the format:
    {{
        "{root_concept}": {{
            "subconcept_1": {{}},
            "subconcept_2": {{
                "sub_subconcept_1": {{}}
            }}
        }}
    }}
    Skip explanations; provide only the nested dictionary.
    - Ensure JSON format correctness.
    - Generate nested subconcept hierarchies with at least 30 subconcepts at each level.
    - Return strictly nested dictionaries only.
    """
    
    meta_prompt = construct_meta_prompt(
        root_concept=concept_name,
        properties=result.property_groups,
        key_aspects=result.key_aspects,
        rare_info=result.rare_info,
        hierarchies=result.hierarchies,
        present_features=result.present_features,
        levels=levels,
        context_data=json.dumps(clean_data_context, indent=2),
        constraints=constraints
    )
    print(f"Meta-prompt: {meta_prompt}")
    response = model.invoke([SystemMessage(content=meta_prompt)])
    meta_prompt_next = response.content.strip() + "\n" + constraints
    print(f"Meta-prompt response: {meta_prompt_next}")
    
    # Generate taxonomy tree
    subconcepts_tree = generate_taxonomy_tree(model, meta_prompt_next)  
    print(f"Generated subconcepts tree: {subconcepts_tree}")

    # Validate and refine taxonomy tree
    validated_tree = validate_taxonomy_tree(
        model,
        subconcepts_tree,
        validation_context=json.dumps(clean_rules_context, indent=2),
        constraints=constraints
    )
    print(f"Validated tree: {validated_tree}")
    result.add_subconcepts(validated_tree)

    result.print_subconcepts_tree()
    return result
