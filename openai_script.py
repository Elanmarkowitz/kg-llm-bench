import requests
"""
This script retrieves a subgraph from Wikidata based on a list of entity IDs, pseudonymizes the entities, builds a graph, generates questions about the graph, and evaluates the questions using the OpenAI API.
Functions:
- get_wikidata_subgraph(entity_ids: List[str]) -> List[Dict]: Queries Wikidata and returns a subgraph based on the provided entity IDs.
- pseudonymize_entities(bindings: List[Dict]) -> Tuple[Dict[str, str], Dict[str, str], List[Dict]]: Pseudonymizes the entities in the provided bindings and returns the entity mapping, property mapping, and pseudonymized bindings.
- build_graph(bindings: List[Dict]) -> nx.Graph: Builds a graph based on the provided bindings.
- generate_questions(G: nx.Graph, entity_mapping: Dict[str, str], property_mapping: Dict[str, str]) -> List[Dict]: Generates questions about the graph using the provided entity mapping and property mapping.
- evaluate_questions(questions: List[Dict], context: str) -> List[Dict]: Evaluates the questions using the OpenAI API and returns the results.
- main(): The main function that orchestrates the execution of the script.
Usage:
1. Set your OpenAI API key as an environment variable or uncomment the line to set it directly in the script.
2. Specify the entity IDs for the subgraph in the `seed_entities` list.
3. Run the script.
Note: This script requires the `requests`, `random`, `networkx`, and `openai` libraries.
"""
import random
import networkx as nx
import openai
import os
from typing import List, Dict, Tuple

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure you set your key as an environment variable
# If you prefer, uncomment the following line and put your API key directly
# openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to query Wikidata and get a subgraph
def get_wikidata_subgraph(entity_ids: List[str]) -> List[Dict]:
    query = """
    SELECT ?item ?itemLabel ?prop ?propLabel ?value ?valueLabel
    WHERE {
      VALUES ?item { %s }
      ?item ?prop ?value .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    """ % (' '.join('wd:' + eid for eid in entity_ids))

    url = 'https://query.wikidata.org/sparql'
    headers = {
        'Accept': 'application/json'
    }
    r = requests.get(url, params={'query': query}, headers=headers)
    data = r.json()
    return data['results']['bindings']

# Function to pseudonymize entities
def pseudonymize_entities(bindings: List[Dict]) -> Tuple[Dict[str, str], Dict[str, str], List[Dict]]:
    entity_mapping = {}
    property_mapping = {}
    entity_counter = 1
    property_counter = 1

    for b in bindings:
        for key in ['item', 'value']:
            uri = b[key]['value']
            label = b[key + 'Label']['value']
            if uri not in entity_mapping:
                pseudonym = f"E{entity_counter}"
                entity_counter += 1
                entity_mapping[uri] = pseudonym
        prop_uri = b['prop']['value']
        prop_label = b['propLabel']['value']
        if prop_uri not in property_mapping:
            pseudonym = f"P{property_counter}"
            property_counter += 1
            property_mapping[prop_uri] = pseudonym

    # Pseudonymize the bindings
    pseudonymized_bindings = []
    for b in bindings:
        pseudonymized_bindings.append({
            'item': entity_mapping[b['item']['value']],
            'itemLabel': b['itemLabel']['value'],
            'prop': property_mapping[b['prop']['value']],
            'propLabel': b['propLabel']['value'],
            'value': entity_mapping.get(b['value']['value'], b['value']['value']),
            'valueLabel': b.get('valueLabel', {}).get('value', b['value']['value'])
        })

    return entity_mapping, property_mapping, pseudonymized_bindings

# Function to build a graph
def build_graph(bindings: List[Dict]) -> nx.Graph:
    G = nx.Graph()
    for b in bindings:
        G.add_edge(b['item'], b['value'], prop=b['prop'])
    return G

# Function to generate questions
def generate_questions(G: nx.Graph, entity_mapping: Dict[str, str], property_mapping: Dict[str, str]) -> List[Dict]:
    questions = []

    # Simple recall questions
    nodes = list(G.nodes())
    random.shuffle(nodes)
    for node in nodes[:5]:
        q = f"What are the direct connections of {node}?"
        a = ', '.join(G[node])
        questions.append({
            'type': 'simple_recall',
            'question': q,
            'answer': a
        })

    # Path existence questions
    node_pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
    random.shuffle(node_pairs)
    for n1, n2 in node_pairs[:5]:
        exists = nx.has_path(G, n1, n2)
        q = f"Is there a path between {n1} and {n2}?"
        a = 'Yes' if exists else 'No'
        questions.append({
            'type': 'path_existence',
            'question': q,
            'answer': a
        })

    # Shortest path questions
    for n1, n2 in node_pairs[5:10]:
        if nx.has_path(G, n1, n2):
            path = nx.shortest_path(G, n1, n2)
            q = f"What is the shortest path between {n1} and {n2}?"
            a = ' -> '.join(path)
            questions.append({
                'type': 'shortest_path',
                'question': q,
                'answer': a
            })
    return questions

# Function to evaluate questions using OpenAI API
def evaluate_questions(questions: List[Dict], context: str) -> List[Dict]:
    results = []
    for q in questions:
        prompt = f"{context}\n\nQuestion: {q['question']}\nAnswer:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0,
            n=1,
            stop=None
        )
        model_answer = response['choices'][0]['text'].strip()
        results.append({
            'question': q['question'],
            'expected_answer': q['answer'],
            'model_answer': model_answer,
            'correct': model_answer.lower() == q['answer'].lower()
        })
    return results

# Main function
def main():
    # Example entity IDs (Wikidata IDs). You can choose different ones as needed.
    seed_entities = ['Q76', 'Q95', 'Q159', 'Q42', 'Q64']  # Barack Obama, Earth, Apple Inc., Douglas Adams, Berlin

    # Get subgraph data
    bindings = get_wikidata_subgraph(seed_entities)

    # Pseudonymize entities
    entity_mapping, property_mapping, p_bindings = pseudonymize_entities(bindings)

    # Build graph
    G = build_graph(p_bindings)

    # Generate context (knowledge graph description)
    context_lines = []
    for b in p_bindings:
        s = f"{b['item']} --[{b['prop']}]--> {b['value']}"
        context_lines.append(s)
    context = '\n'.join(context_lines)

    # Generate questions
    questions = generate_questions(G, entity_mapping, property_mapping)

    # Evaluate questions
    results = evaluate_questions(questions, context)

    # Print results
    for res in results:
        print("Question:", res['question'])
        print("Expected Answer:", res['expected_answer'])
        print("Model Answer:", res['model_answer'])
        print("Correct:", res['correct'])
        print("-" * 50)
    

if name == "main":
    main()