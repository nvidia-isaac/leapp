#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as patches
from matplotlib.widgets import Button
from leapp.utils.logging import _get_logger


class InteractiveGraphVisualizer:
    """Interactive graph visualizer with draggable nodes that don't auto-resize."""
    
    def __init__(self, G, pos, graph_name):
        self.G = G
        self.pos = pos
        self.graph_name = graph_name
        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        
        # Artist storage for selective updates
        self.node_artists = {}  # node_name -> PathCollection
        self.label_artists = {}  # node_name -> Text
        self.edge_artists = []  # List of FancyArrowPatch objects
        self.edge_label_artists = {}  # (u, v) -> Text
        self.self_loop_label_artists = {}  # (u, v) -> Text
        
        # Dragging state
        self.dragging = {'active': False, 'node': None, 'offset': (0, 0)}
        
        # Categorize nodes
        self.computation_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'computation']
        self.input_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'input']
        self.output_nodes = [n for n in G.nodes() if G.nodes[n]['node_type'] == 'output']
        
        # Categorize edges
        self.internal_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'internal']
        self.input_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'input']
        self.output_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'output']
        self.feedback_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == 'feedback']
        self.feedback_self_loops = [(u, v) for u, v in self.feedback_edges if u == v]
        self.feedback_regular = [(u, v) for u, v in self.feedback_edges if u != v]
        
        # Store fixed axis limits
        self.xlim = None
        self.ylim = None
        
        # Zoom buttons (will be created in draw_initial)
        self.zoom_in_btn = None
        self.zoom_out_btn = None
        self.reset_btn = None
        self.zoom_factor = 0.2  # 20% zoom per click
        self.scale = 1.0  # Scale factor for node/text sizes
        
        # Base sizes (will be scaled)
        self.base_node_sizes = {'computation': 4000, 'input': 2500, 'output': 2500}
        self.base_font_size = 11
        self.base_edge_font_size = 9
        self.base_arrow_size = 25
        self.base_edge_width = 2.5
        self.base_linewidth = 2
    
    def draw_initial(self, show_hint=True):
        """Draw the graph initially and store artist references."""
        # Draw nodes and store references
        self._draw_nodes()
        self._draw_edges()
        self._draw_labels()
        self._draw_edge_labels()
        self._draw_legend()
        self._draw_title(show_hint)
        
        self.ax.axis('off')
        plt.tight_layout()
        
        # Store the initial axis limits
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()
        self.initial_xlim = self.xlim  # For reset functionality
        self.initial_ylim = self.ylim
        
        # Create zoom buttons
        self._create_zoom_buttons()
    
    def _draw_nodes(self):
        """Draw all nodes and store artist references."""
        # Scaled sizes
        s = self.scale
        comp_size = int(self.base_node_sizes['computation'] * s)
        input_size = int(self.base_node_sizes['input'] * s)
        output_size = int(self.base_node_sizes['output'] * s)
        lw = self.base_linewidth * s
        
        # Node styling config (with scaled sizes)
        node_configs = [
            (self.computation_nodes, 'lightblue', 's', comp_size, 0.9, lw, 'darkblue'),
            (self.input_nodes, 'lightgreen', 'o', input_size, 0.8, lw, 'darkgreen'),
            (self.output_nodes, 'lightcoral', 'o', output_size, 0.8, lw, 'darkred'),
        ]
        
        for nodelist, color, shape, size, alpha, linewidth, edgecolor in node_configs:
            if nodelist:
                collection = nx.draw_networkx_nodes(
                    self.G, self.pos, nodelist=nodelist, ax=self.ax,
                    node_color=color, node_shape=shape, node_size=size,
                    alpha=alpha, linewidths=linewidth, edgecolors=edgecolor
                )
                # Store reference for each node in this collection
                for i, node in enumerate(nodelist):
                    self.node_artists[node] = (collection, i, nodelist)
    
    def _draw_edges(self):
        """Draw all edges."""
        # Scaled sizes
        s = self.scale
        arrow_size = int(self.base_arrow_size * s)
        arrow_size_small = int(20 * s)
        edge_width = self.base_edge_width * s
        edge_width_small = 2 * s
        margin = int(40 * s)
        margin_small = int(30 * s)
        
        # Internal edges
        if self.internal_edges:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.internal_edges, ax=self.ax,
                edge_color='black', arrows=True, arrowsize=arrow_size, width=edge_width,
                arrowstyle='->', min_source_margin=margin, min_target_margin=margin
            )
        
        # Feedback regular edges
        if self.feedback_regular:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.feedback_regular, ax=self.ax,
                edge_color='red', arrows=True, arrowsize=arrow_size, width=edge_width,
                arrowstyle='->', connectionstyle='arc3,rad=0.15',
                min_source_margin=margin, min_target_margin=margin
            )
        
        # Feedback self-loops
        if self.feedback_self_loops:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.feedback_self_loops, ax=self.ax,
                edge_color='red', arrows=True, arrowsize=arrow_size, width=edge_width,
                arrowstyle='->', connectionstyle='angle3,angleA=45,angleB=135',
                min_source_margin=margin, min_target_margin=margin
            )
        
        # Input edges
        if self.input_edges:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.input_edges, ax=self.ax,
                edge_color='darkgreen', arrows=True, arrowsize=arrow_size_small, width=edge_width_small,
                style='dashed', arrowstyle='->', min_source_margin=margin_small, min_target_margin=margin
            )
        
        # Output edges
        if self.output_edges:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.output_edges, ax=self.ax,
                edge_color='darkred', arrows=True, arrowsize=arrow_size_small, width=edge_width_small,
                style='dashed', arrowstyle='->', min_source_margin=margin, min_target_margin=margin_small
            )
    
    def _draw_labels(self):
        """Draw node labels and store artist references."""
        # Scaled font size
        font_size = max(6, int(self.base_font_size * self.scale))
        
        labels = {}
        for node in self.G.nodes():
            if node.startswith('INPUT_'):
                labels[node] = node[6:]
            elif node.startswith('OUTPUT_'):
                labels[node] = node[7:]
            else:
                labels[node] = node
        
        label_dict = nx.draw_networkx_labels(
            self.G, self.pos, labels, ax=self.ax, font_size=font_size, font_weight='bold'
        )
        self.label_artists = label_dict
    
    def _draw_edge_labels(self):
        """Draw edge labels and store artist references."""
        # Scaled font size
        font_size = max(5, int(self.base_edge_font_size * self.scale))
        
        edge_labels = {}
        self_loop_labels = {}
        
        for u, v, d in self.G.edges(data=True):
            if d['edge_type'] == 'internal' or d['edge_type'] == 'feedback':
                if u == v:
                    self_loop_labels[(u, v)] = d['label']
                else:
                    edge_labels[(u, v)] = d['label']
        
        # Draw regular edge labels
        if edge_labels:
            label_dict = nx.draw_networkx_edge_labels(
                self.G, self.pos, edge_labels, ax=self.ax, font_size=font_size,
                font_weight='bold', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )
            self.edge_label_artists = label_dict
        
        # Draw self-loop labels
        for (u, v), label in self_loop_labels.items():
            node_pos = self.pos[u]
            label_x = node_pos[0]
            label_y = node_pos[1] + 0.2
            
            text = self.ax.text(
                label_x, label_y, label,
                horizontalalignment='center', verticalalignment='center',
                fontsize=font_size, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
            )
            self.self_loop_label_artists[(u, v)] = text
    
    def _draw_legend(self):
        """Draw the legend."""
        legend_elements = [
            patches.Patch(color='lightblue', label='Computation Nodes'),
            patches.Patch(color='lightgreen', label='Graph Inputs'),
            patches.Patch(color='lightcoral', label='Graph Outputs'),
            plt.Line2D([0], [0], color='black', linewidth=2, label='Internal Connections'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Feedback Connections'),
            plt.Line2D([0], [0], color='green', linewidth=1.5, linestyle='--', label='Input Connections'),
            plt.Line2D([0], [0], color='darkred', linewidth=1.5, linestyle='--', label='Output Connections')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    def _create_zoom_buttons(self):
        """Create zoom control buttons beneath the legend."""
        # Position buttons in figure coordinates (right side, below legend)
        btn_width = 0.06
        btn_height = 0.04
        btn_left = 0.88
        btn_top = 0.45
        btn_spacing = 0.05
        
        # Zoom In button
        ax_zoom_in = self.fig.add_axes([btn_left, btn_top, btn_width, btn_height])
        self.zoom_in_btn = Button(ax_zoom_in, 'Zoom +', color='lightgray', hovercolor='lightblue')
        self.zoom_in_btn.on_clicked(self._on_zoom_in)
        
        # Zoom Out button
        ax_zoom_out = self.fig.add_axes([btn_left, btn_top - btn_spacing, btn_width, btn_height])
        self.zoom_out_btn = Button(ax_zoom_out, 'Zoom -', color='lightgray', hovercolor='lightcoral')
        self.zoom_out_btn.on_clicked(self._on_zoom_out)
        
        # Reset button
        ax_reset = self.fig.add_axes([btn_left, btn_top - 2 * btn_spacing, btn_width, btn_height])
        self.reset_btn = Button(ax_reset, 'Reset', color='lightgray', hovercolor='lightgreen')
        self.reset_btn.on_clicked(self._on_reset_zoom)
    
    def _on_zoom_in(self, event):
        """Zoom in - make things appear larger."""
        # Adjust view limits
        x_center = (self.xlim[0] + self.xlim[1]) / 2
        y_center = (self.ylim[0] + self.ylim[1]) / 2
        x_range = (self.xlim[1] - self.xlim[0]) * (1 - self.zoom_factor) / 2
        y_range = (self.ylim[1] - self.ylim[0]) * (1 - self.zoom_factor) / 2
        
        self.xlim = (x_center - x_range, x_center + x_range)
        self.ylim = (y_center - y_range, y_center + y_range)
        
        # Increase scale for shapes and text
        self.scale *= (1 + self.zoom_factor)
        
        # Full redraw with new sizes
        self.redraw_preserving_limits()
    
    def _on_zoom_out(self, event):
        """Zoom out - make things appear smaller."""
        # Adjust view limits
        x_center = (self.xlim[0] + self.xlim[1]) / 2
        y_center = (self.ylim[0] + self.ylim[1]) / 2
        x_range = (self.xlim[1] - self.xlim[0]) * (1 + self.zoom_factor) / 2
        y_range = (self.ylim[1] - self.ylim[0]) * (1 + self.zoom_factor) / 2
        
        self.xlim = (x_center - x_range, x_center + x_range)
        self.ylim = (y_center - y_range, y_center + y_range)
        
        # Decrease scale for shapes and text (with minimum)
        self.scale = max(0.2, self.scale * (1 - self.zoom_factor))
        
        # Full redraw with new sizes
        self.redraw_preserving_limits()
    
    def _on_reset_zoom(self, event):
        """Reset to initial zoom level."""
        self.xlim = self.initial_xlim
        self.ylim = self.initial_ylim
        self.scale = 1.0  # Reset scale
        
        # Full redraw with original sizes
        self.redraw_preserving_limits()
    
    def _draw_title(self, show_hint=True):
        """Draw the title."""
        if show_hint:
            self.ax.set_title(
                f"{self.graph_name} Graph Visualization\n(Drag nodes to reposition, then close window to save)",
                fontsize=16, fontweight='bold', pad=20
            )
        else:
            self.ax.set_title(
                f"{self.graph_name} Graph Visualization",
                fontsize=16, fontweight='bold', pad=20
            )
    
    def redraw_preserving_limits(self):
        """Redraw the graph but preserve axis limits to prevent auto-resizing."""
        # Store current limits
        xlim = self.xlim
        ylim = self.ylim
        
        # Clear and redraw
        self.ax.clear()
        self._draw_nodes()
        self._draw_edges()
        self._draw_labels()
        self._draw_edge_labels()
        self._draw_legend()
        self._draw_title(show_hint=True)
        self.ax.axis('off')
        
        # Restore limits
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        
        self.fig.canvas.draw_idle()
    
    def find_closest_node(self, x, y, threshold=0.3):
        """Find the closest node to the given coordinates."""
        min_dist = float('inf')
        closest_node = None
        for node, (node_x, node_y) in self.pos.items():
            dist = ((x - node_x)**2 + (y - node_y)**2)**0.5
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_node = node
        return closest_node
    
    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax:
            return
        
        node = self.find_closest_node(event.xdata, event.ydata)
        if node:
            self.dragging['active'] = True
            self.dragging['node'] = node
            self.dragging['offset'] = (
                event.xdata - self.pos[node][0],
                event.ydata - self.pos[node][1]
            )
    
    def on_motion(self, event):
        """Handle mouse motion events."""
        if not self.dragging['active'] or event.inaxes != self.ax:
            return
        
        node = self.dragging['node']
        new_x = event.xdata - self.dragging['offset'][0]
        new_y = event.ydata - self.dragging['offset'][1]
        self.pos[node] = (new_x, new_y)
        
        # Redraw with preserved limits
        self.redraw_preserving_limits()
    
    def on_release(self, event):
        """Handle mouse release events."""
        if self.dragging['active']:
            self.dragging['active'] = False
            self.dragging['node'] = None
    
    def connect_events(self):
        """Connect mouse event handlers."""
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
    
    def show(self):
        """Show the interactive plot."""
        # keep the print here. 
        print("Interactive graph displayed! Drag nodes to reposition them.")
        print("Close the window when you're satisfied with the layout to save the image.")
        plt.show()
    
    def save(self, save_path):
        """Save the final graph with current positions and scale."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Use current scale for saved image
        s = self.scale
        comp_size = int(self.base_node_sizes['computation'] * s)
        input_size = int(self.base_node_sizes['input'] * s)
        output_size = int(self.base_node_sizes['output'] * s)
        lw = self.base_linewidth * s
        arrow_size = int(self.base_arrow_size * s)
        arrow_size_small = int(20 * s)
        edge_width = self.base_edge_width * s
        edge_width_small = 2 * s
        margin = int(40 * s)
        margin_small = int(30 * s)
        font_size = max(6, int(self.base_font_size * s))
        edge_font_size = max(5, int(self.base_edge_font_size * s))
        
        # Draw nodes with scaled sizes
        node_configs = [
            (self.computation_nodes, 'lightblue', 's', comp_size, 0.9, lw, 'darkblue'),
            (self.input_nodes, 'lightgreen', 'o', input_size, 0.8, lw, 'darkgreen'),
            (self.output_nodes, 'lightcoral', 'o', output_size, 0.8, lw, 'darkred'),
        ]
        for nodelist, color, shape, size, alpha, linewidth, edgecolor in node_configs:
            if nodelist:
                nx.draw_networkx_nodes(
                    self.G, self.pos, nodelist=nodelist, ax=ax,
                    node_color=color, node_shape=shape, node_size=size,
                    alpha=alpha, linewidths=linewidth, edgecolors=edgecolor
                )
        
        # Draw edges with scaled sizes
        if self.internal_edges:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.internal_edges, ax=ax,
                edge_color='black', arrows=True, arrowsize=arrow_size, width=edge_width,
                arrowstyle='->', min_source_margin=margin, min_target_margin=margin
            )
        if self.feedback_regular:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.feedback_regular, ax=ax,
                edge_color='red', arrows=True, arrowsize=arrow_size, width=edge_width,
                arrowstyle='->', connectionstyle='arc3,rad=0.15',
                min_source_margin=margin, min_target_margin=margin
            )
        if self.feedback_self_loops:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.feedback_self_loops, ax=ax,
                edge_color='red', arrows=True, arrowsize=arrow_size, width=edge_width,
                arrowstyle='->', connectionstyle='angle3,angleA=45,angleB=135',
                min_source_margin=margin, min_target_margin=margin
            )
        if self.input_edges:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.input_edges, ax=ax,
                edge_color='darkgreen', arrows=True, arrowsize=arrow_size_small, width=edge_width_small,
                style='dashed', arrowstyle='->', min_source_margin=margin_small, min_target_margin=margin
            )
        if self.output_edges:
            nx.draw_networkx_edges(
                self.G, self.pos, edgelist=self.output_edges, ax=ax,
                edge_color='darkred', arrows=True, arrowsize=arrow_size_small, width=edge_width_small,
                style='dashed', arrowstyle='->', min_source_margin=margin, min_target_margin=margin_small
            )
        
        # Draw labels with scaled font
        labels = {}
        for node in self.G.nodes():
            if node.startswith('INPUT_'):
                labels[node] = node[6:]
            elif node.startswith('OUTPUT_'):
                labels[node] = node[7:]
            else:
                labels[node] = node
        nx.draw_networkx_labels(self.G, self.pos, labels, ax=ax, font_size=font_size, font_weight='bold')
        
        # Draw edge labels with scaled font
        edge_labels = {}
        self_loop_labels = {}
        for u, v, d in self.G.edges(data=True):
            if d['edge_type'] == 'internal' or d['edge_type'] == 'feedback':
                if u == v:
                    self_loop_labels[(u, v)] = d['label']
                else:
                    edge_labels[(u, v)] = d['label']
        
        if edge_labels:
            nx.draw_networkx_edge_labels(
                self.G, self.pos, edge_labels, ax=ax, font_size=edge_font_size,
                font_weight='bold', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )
        
        for (u, v), label in self_loop_labels.items():
            node_pos = self.pos[u]
            ax.text(
                node_pos[0], node_pos[1] + 0.2, label,
                horizontalalignment='center', verticalalignment='center',
                fontsize=edge_font_size, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9)
            )
        
        # Legend
        legend_elements = [
            patches.Patch(color='lightblue', label='Computation Nodes'),
            patches.Patch(color='lightgreen', label='Graph Inputs'),
            patches.Patch(color='lightcoral', label='Graph Outputs'),
            plt.Line2D([0], [0], color='black', linewidth=2, label='Internal Connections'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Feedback Connections'),
            plt.Line2D([0], [0], color='green', linewidth=1.5, linestyle='--', label='Input Connections'),
            plt.Line2D([0], [0], color='darkred', linewidth=1.5, linestyle='--', label='Output Connections')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Title without hint
        ax.set_title(f"{self.graph_name} Graph Visualization", fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        _get_logger().info(f"Graph visualization saved as: {save_path}")


def visualize_graph(nodes, connections, feedback_connections, inputs, outputs, save_path, graph_name):
    """
    Visualize the computational graph showing nodes, connections, and dangling I/O.

    Args:
        nodes: Dict of node_name -> FunctionDecoratorNode objects
        connections: List of connection dicts with 'source' and 'targets' keys
        feedback_connections: List of feedback connection dicts with 'source' and 'targets' keys
        inputs: List of dangling graph inputs (not connected internally)
        outputs: List of dangling graph outputs (not connected internally) 
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add all computation nodes
    for node_name in nodes.keys():
        G.add_node(node_name, node_type='computation', color='lightblue', shape='box')

    # Add dangling input and output nodes
    for inp in inputs:
        if '/' in inp:
            node_name, input_name = inp.split('/', 1)
            G.add_node(f"INPUT_{input_name}", node_type='input', color='lightgreen', shape='ellipse')
            G.add_edge(f"INPUT_{input_name}", node_name, label=input_name, edge_type='input')

    for out in outputs:
        if '/' in out:
            node_name, output_name = out.split('/', 1)
            G.add_node(f"OUTPUT_{output_name}", node_type='output', color='lightcoral', shape='ellipse')
            G.add_edge(node_name, f"OUTPUT_{output_name}", label=output_name, edge_type='output')

    # Collect connections between node pairs to combine labels
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
    for (source_node, target_node), edge_outputs in node_pair_connections.items():
        label = edge_outputs[0] if len(edge_outputs) == 1 else '\n'.join(edge_outputs)
        G.add_edge(source_node, target_node, label=label, edge_type='internal')

    # Add edges with combined labels for feedback connections
    for (source_node, target_node), edge_outputs in node_pair_feedback_connections.items():
        label = edge_outputs[0] if len(edge_outputs) == 1 else '\n'.join(edge_outputs)
        G.add_edge(source_node, target_node, label=label, edge_type='feedback')

    # Use NetworkX spring layout
    pos = nx.spring_layout(G)

    # Create interactive visualizer
    visualizer = InteractiveGraphVisualizer(G, pos, graph_name)
    visualizer.draw_initial(show_hint=True)
    visualizer.connect_events()
    visualizer.show()

    # Save after user closes the window
    save_file_path = os.path.join(save_path, f"{graph_name}.png")
    visualizer.save(save_file_path)
