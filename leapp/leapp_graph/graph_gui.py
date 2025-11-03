#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as patches


def visualize_graph(nodes, connections, feedback_connections, inputs, outputs, save_path, graph_name):
    """
    Visualize the computational graph showing nodes, connections, and dangling I/O.

    Args:
        nodes: Dict of node_name -> NodeContext objects
        connections: List of connection dicts with 'source' and 'targets' keys
        feedback_connections: List of feedback connection dicts with 'source' and 'targets' keys
        inputs: List of dangling graph inputs (not connected internally)
        outputs: List of dangling graph outputs (not connected internally) 
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add all computation nodes
    for node_name in nodes.keys():
        G.add_node(node_name, node_type='computation',
                   color='lightblue', shape='box')

    # Add dangling input and output nodes
    dangling_inputs = set()
    dangling_outputs = set()

    # Parse dangling inputs and outputs to get node names
    for inp in inputs:
        if '/' in inp:
            node_name, input_name = inp.split('/', 1)
            dangling_inputs.add(f"INPUT_{input_name}")
            G.add_node(f"INPUT_{input_name}", node_type='input',
                       color='lightgreen', shape='ellipse')
            G.add_edge(f"INPUT_{input_name}", node_name,
                       label=input_name, edge_type='input')

    for out in outputs:
        if '/' in out:
            node_name, output_name = out.split('/', 1)
            dangling_outputs.add(f"OUTPUT_{output_name}")
            G.add_node(f"OUTPUT_{output_name}", node_type='output',
                       color='lightcoral', shape='ellipse')
            G.add_edge(
                node_name, f"OUTPUT_{output_name}", label=output_name, edge_type='output')

        # Add internal connections between computation nodes
    # First collect all connections between node pairs to combine labels
    node_pair_connections = {}
    node_pair_feedback_connections = {}

    # Process forward connections
    for connection in connections:
        source = connection['source']
        source_node = source['node'].name
        source_output = source['node'].outputs[source['idx']].name_str

        for target in connection['targets']:
            target_node = target['node'].name

            pair_key = (source_node, target_node)
            if pair_key not in node_pair_connections:
                node_pair_connections[pair_key] = []
            node_pair_connections[pair_key].append(source_output)

    # Process feedback connections
    for connection in feedback_connections:
        source = connection['source']
        source_node = source['node'].name
        source_output = source['node'].outputs[source['idx']].name_str

        for target in connection['targets']:
            target_node = target['node'].name

            pair_key = (source_node, target_node)
            if pair_key not in node_pair_feedback_connections:
                node_pair_feedback_connections[pair_key] = []
            node_pair_feedback_connections[pair_key].append(source_output)

    # Add edges with combined labels for forward connections
    for (source_node, target_node), outputs in node_pair_connections.items():
        if len(outputs) == 1:
            label = outputs[0]
        else:
            # Combine multiple outputs into one label
            label = '\n'.join(outputs)

        G.add_edge(source_node, target_node,
                   label=label,
                   edge_type='internal')

    # Add edges with combined labels for feedback connections
    for (source_node, target_node), outputs in node_pair_feedback_connections.items():
        if len(outputs) == 1:
            label = outputs[0]
        else:
            # Combine multiple outputs into one label
            label = '\n'.join(outputs)

        G.add_edge(source_node, target_node,
                   label=label,
                   edge_type='feedback')

    def draw_graph_elements(G, pos, graph_name, show_hint=True):
        """Helper function to draw all graph elements with consistent styling."""
        # Draw different types of nodes with different styles
        computation_nodes = [n for n in G.nodes() if G.nodes[n]
                             ['node_type'] == 'computation']
        input_nodes = [n for n in G.nodes() if G.nodes[n]
                       ['node_type'] == 'input']
        output_nodes = [n for n in G.nodes() if G.nodes[n]
                        ['node_type'] == 'output']

        # Draw computation nodes (main nodes) - larger and more prominent
        nx.draw_networkx_nodes(G, pos, nodelist=computation_nodes,
                               node_color='lightblue', node_shape='s',
                               node_size=4000, alpha=0.9, linewidths=2, edgecolors='darkblue')

        # Draw input nodes - slightly smaller with border
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes,
                               node_color='lightgreen', node_shape='o',
                               node_size=2500, alpha=0.8, linewidths=2, edgecolors='darkgreen')

        # Draw output nodes - slightly smaller with border
        nx.draw_networkx_nodes(G, pos, nodelist=output_nodes,
                               node_color='lightcoral', node_shape='o',
                               node_size=2500, alpha=0.8, linewidths=2, edgecolors='darkred')

        # Draw edges with different styles
        internal_edges = [(u, v) for u, v, d in G.edges(
            data=True) if d['edge_type'] == 'internal']
        input_edges = [(u, v) for u, v, d in G.edges(
            data=True) if d['edge_type'] == 'input']
        output_edges = [(u, v) for u, v, d in G.edges(
            data=True) if d['edge_type'] == 'output']
        feedback_edges = [(u, v) for u, v, d in G.edges(
            data=True) if d['edge_type'] == 'feedback']

        # Separate feedback edges into self-loops and regular edges
        feedback_self_loops = [(u, v) for u, v in feedback_edges if u == v]
        feedback_regular = [(u, v) for u, v in feedback_edges if u != v]

        # Draw internal connections (solid lines) - thicker and more prominent
        nx.draw_networkx_edges(G, pos, edgelist=internal_edges,
                               edge_color='black', arrows=True,
                               arrowsize=25, width=2.5, arrowstyle='->',
                               min_source_margin=40, min_target_margin=40)

        # Draw regular feedback connections (curved lines) - red theme with curvature
        if feedback_regular:
            nx.draw_networkx_edges(G, pos, edgelist=feedback_regular,
                                   edge_color='red', arrows=True,
                                   arrowsize=25, width=2.5, arrowstyle='->',
                                   connectionstyle='arc3,rad=0.15',
                                   min_source_margin=40, min_target_margin=40)

        # Draw self-loop feedback connections with larger radius for better visibility
        if feedback_self_loops:
            nx.draw_networkx_edges(G, pos, edgelist=feedback_self_loops,
                                   edge_color='red', arrows=True,
                                   arrowsize=25, width=2.5, arrowstyle='->',
                                   connectionstyle='angle3,angleA=45,angleB=135',
                                   min_source_margin=40, min_target_margin=40)

        # Draw input connections (dashed lines) - green theme
        nx.draw_networkx_edges(G, pos, edgelist=input_edges,
                               edge_color='darkgreen', arrows=True,
                               arrowsize=20, width=2, style='dashed', arrowstyle='->',
                               min_source_margin=30, min_target_margin=40)

        # Draw output connections (dashed lines) - red theme
        nx.draw_networkx_edges(G, pos, edgelist=output_edges,
                               edge_color='darkred', arrows=True,
                               arrowsize=20, width=2, style='dashed', arrowstyle='->',
                               min_source_margin=40, min_target_margin=30)

        # Create custom labels without INPUT_/OUTPUT_ prefixes
        labels = {}
        for node in G.nodes():
            if node.startswith('INPUT_'):
                labels[node] = node[6:]  # Remove 'INPUT_' prefix
            elif node.startswith('OUTPUT_'):
                labels[node] = node[7:]  # Remove 'OUTPUT_' prefix
            else:
                labels[node] = node  # Keep original name for computation nodes

        # Add node labels with better font
        nx.draw_networkx_labels(G, pos, labels, font_size=11,
                                font_weight='bold')

        # Add edge labels for internal and feedback connections with better positioning
        edge_labels = {}
        self_loop_labels = {}
        for u, v, d in G.edges(data=True):
            if d['edge_type'] == 'internal' or d['edge_type'] == 'feedback':
                if u == v:  # Self-loop
                    self_loop_labels[(u, v)] = d['label']
                else:
                    edge_labels[(u, v)] = d['label']

        # Draw regular edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9,
                                     font_weight='bold', alpha=0.8,
                                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Draw self-loop labels manually positioned above the loop
        for (u, v), label in self_loop_labels.items():
            node_pos = pos[u]
            # Position label significantly above the node to clear the loop
            label_x = node_pos[0]
            label_y = node_pos[1] + 0.2  # Higher offset

            plt.text(label_x, label_y, label,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=9,
                     fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='gray', alpha=0.9))

        # Create legend
        legend_elements = [
            patches.Patch(color='lightblue', label='Computation Nodes'),
            patches.Patch(color='lightgreen', label='Graph Inputs'),
            patches.Patch(color='lightcoral', label='Graph Outputs'),
            plt.Line2D([0], [0], color='black', linewidth=2,
                       label='Internal Connections'),
            plt.Line2D([0], [0], color='red', linewidth=2,
                       label='Feedback Connections'),
            plt.Line2D([0], [0], color='green', linewidth=1.5,
                       linestyle='--', label='Input Connections'),
            plt.Line2D([0], [0], color='darkred', linewidth=1.5,
                       linestyle='--', label='Output Connections')
        ]

        plt.legend(handles=legend_elements, loc='upper right',
                   bbox_to_anchor=(1.15, 1))

        # Conditionally show hint in title
        if show_hint:
            plt.title(f"{graph_name} Graph Visualization\n(Drag nodes to reposition, then close window to save)",
                      fontsize=16, fontweight='bold', pad=20)
        else:
            plt.title(f"{graph_name} Graph Visualization",
                      fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()

        return computation_nodes, input_nodes, output_nodes, internal_edges, feedback_edges, input_edges, output_edges

    # Create the visualization with better proportions
    plt.figure(figsize=(16, 8))

    # Use default NetworkX layout
    pos = nx.spring_layout(G)

    # Draw the graph using the helper function
    computation_nodes, input_nodes, output_nodes, internal_edges, feedback_edges, input_edges, output_edges = draw_graph_elements(
        # Show hint during initial interactive display
        G, pos, graph_name, show_hint=True)

    # Add interactive node dragging functionality
    dragging = {'active': False, 'node': None, 'offset': (0, 0)}

    def find_closest_node(x, y, pos, threshold=0.3):
        """Find the closest node to the given coordinates."""
        min_dist = float('inf')
        closest_node = None
        for node, (node_x, node_y) in pos.items():
            dist = ((x - node_x)**2 + (y - node_y)**2)**0.5
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_node = node
        return closest_node

    def on_press(event):
        """Handle mouse press events."""
        if event.inaxes is None:
            return

        # Find the closest node to the click
        node = find_closest_node(event.xdata, event.ydata, pos)
        if node:
            dragging['active'] = True
            dragging['node'] = node
            dragging['offset'] = (event.xdata - pos[node]
                                  [0], event.ydata - pos[node][1])

    def on_motion(event):
        """Handle mouse motion events."""
        if not dragging['active'] or event.inaxes is None:
            return

        node = dragging['node']
        # Update node position
        new_x = event.xdata - dragging['offset'][0]
        new_y = event.ydata - dragging['offset'][1]
        pos[node] = (new_x, new_y)

        # Redraw the graph with updated positions
        redraw_graph()

    def on_release(event):
        """Handle mouse release events."""
        if dragging['active']:
            dragging['active'] = False
            dragging['node'] = None

    def redraw_graph():
        """Redraw the entire graph with current positions."""
        plt.clf()  # Clear the current figure
        # Show hint during interaction
        draw_graph_elements(G, pos, graph_name, show_hint=True)
        plt.draw()  # Update the display

    # Connect the event handlers
    plt.gcf().canvas.mpl_connect('button_press_event', on_press)
    plt.gcf().canvas.mpl_connect('motion_notify_event', on_motion)
    plt.gcf().canvas.mpl_connect('button_release_event', on_release)

    print("Interactive graph displayed! Drag nodes to reposition them.")
    print("Close the window when you're satisfied with the layout to save the image.")

    # Show the interactive plot first
    plt.show()

    # After user closes the window, save the final graph with updated positions
    plt.figure(figsize=(16, 8))
    # Hide hint in saved image
    draw_graph_elements(G, pos, graph_name, show_hint=False)
    save_path = f"{save_path}/{graph_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure to free memory

    print(f"Graph visualization saved as: {save_path}")
