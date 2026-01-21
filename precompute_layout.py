#!/usr/bin/env python3
"""
Pre-compute force-directed layout positions for the EDGI network.
"""

import json
import math
import random
from collections import defaultdict

def load_data(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        return json.load(f)

def force_directed_layout(nodes, edges, iterations=100, min_docs=20):
    """Simple force-directed layout algorithm."""
    print(f"Filtering to nodes with >= {min_docs} documents...")

    filtered_nodes = {n['id']: n for n in nodes if n.get('count', 0) >= min_docs}
    node_ids = set(filtered_nodes.keys())

    print(f"Computing layout for {len(node_ids)} nodes...")

    filtered_edges = [e for e in edges if e['source'] in node_ids and e['target'] in node_ids]
    print(f"With {len(filtered_edges)} edges...")

    adjacency = defaultdict(list)
    for e in filtered_edges:
        adjacency[e['source']].append((e['target'], e.get('weight', 1)))
        adjacency[e['target']].append((e['source'], e.get('weight', 1)))

    positions = {}
    for node_id in node_ids:
        positions[node_id] = {
            'x': random.uniform(-1000, 1000),
            'y': random.uniform(-1000, 1000)
        }

    k = math.sqrt((2000 * 2000) / len(node_ids))

    for iteration in range(iterations):
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}/{iterations}...")

        displacement = {nid: {'x': 0, 'y': 0} for nid in node_ids}

        node_list = list(node_ids)
        for i, n1 in enumerate(node_list):
            for n2 in node_list[i+1:]:
                dx = positions[n1]['x'] - positions[n2]['x']
                dy = positions[n1]['y'] - positions[n2]['y']
                dist = math.sqrt(dx*dx + dy*dy) + 0.01

                force = (k * k) / dist
                fx = (dx / dist) * force
                fy = (dy / dist) * force

                displacement[n1]['x'] += fx
                displacement[n1]['y'] += fy
                displacement[n2]['x'] -= fx
                displacement[n2]['y'] -= fy

        for edge in filtered_edges:
            n1, n2 = edge['source'], edge['target']
            dx = positions[n1]['x'] - positions[n2]['x']
            dy = positions[n1]['y'] - positions[n2]['y']
            dist = math.sqrt(dx*dx + dy*dy) + 0.01

            weight = math.log(edge.get('weight', 1) + 1) + 1
            force = (dist * dist) / k * weight * 0.1
            fx = (dx / dist) * force
            fy = (dy / dist) * force

            displacement[n1]['x'] -= fx
            displacement[n1]['y'] -= fy
            displacement[n2]['x'] += fx
            displacement[n2]['y'] += fy

        temp = 100 * (1 - iteration / iterations)
        for node_id in node_ids:
            dx = displacement[node_id]['x']
            dy = displacement[node_id]['y']
            dist = math.sqrt(dx*dx + dy*dy) + 0.01

            capped = min(dist, temp)
            positions[node_id]['x'] += (dx / dist) * capped
            positions[node_id]['y'] += (dy / dist) * capped

        for node_id in node_ids:
            positions[node_id]['x'] *= 0.99
            positions[node_id]['y'] *= 0.99

    print("Layout complete!")
    return positions, filtered_nodes, filtered_edges

def save_with_positions(original_data, positions, output_path):
    """Save the network data with pre-computed positions."""

    nodes_with_pos = []
    for node in original_data['nodes']:
        node_copy = node.copy()
        if node['id'] in positions:
            node_copy['x'] = positions[node['id']]['x']
            node_copy['y'] = positions[node['id']]['y']
        nodes_with_pos.append(node_copy)

    output_data = {
        'nodes': nodes_with_pos,
        'edges': original_data['edges'],
    }

    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f)

    print("Done!")

if __name__ == '__main__':
    data = load_data('edgi_network.json')

    positions, _, _ = force_directed_layout(
        data['nodes'],
        data['edges'],
        iterations=150,
        min_docs=10
    )

    save_with_positions(data, positions, 'edgi_network_layout.json')

    print(f"\nPre-computed positions for {len(positions)} nodes")
    print("Output: edgi_network_layout.json")
