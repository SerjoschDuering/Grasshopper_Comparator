import base64
import copy
import json
import os

import networkx as nx
from bokeh.io import curdoc
from bokeh.layouts import column, gridplot
from bokeh.models import (
    Button,
    CheckboxButtonGroup,
    ColumnDataSource,
    CustomJS,
    CustomJSHover,
    Div,
    HoverTool,
    Legend,
    LegendItem,
)
from bokeh.models.widgets import Button, FileInput, PasswordInput, Slider, TextInput
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from specklepy.api import operations
from specklepy.api.client import SpeckleClient
from specklepy.transports.server import ServerTransport


# Global state variable to track data source mode
global data_source_mode
data_source_mode = "speckle"  # default to speckle

def robust_json_decode(encoded_str):
    """
    Decodes a string (either plain JSON or base64-encoded) into a JSON object,
    trying multiple strategies to handle different encoding and format issues.
    
    Args:
        encoded_str (str): String containing JSON data (plain or base64-encoded).
        
    Returns:
        dict: Decoded JSON object.
        
    Raises:
        ValueError: If the decoding fails for all attempted strategies.
    """
    
    # Define potential encodings to try
    encodings = ['utf-8-sig', 'utf-8', 'latin-1']

    # First, attempt to directly decode the input as a plain JSON string
    for encoding in encodings:
        try:
            # Attempt to parse JSON directly
            return json.loads(encoded_str)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            # If decoding or JSON parsing fails, move to the next encoding
            continue
        except Exception as e:
            # Catch any other exceptions and continue to the next encoding
            print(f"Error with encoding {encoding}: {e}")
            continue

    # If direct JSON decoding fails, try base64 decoding
    for encoding in encodings:
        try:
            # Decode from base64
            decoded_bytes = base64.b64decode(encoded_str)
            
            # Then decode from bytes to string using the specified encoding
            decoded_str = decoded_bytes.decode(encoding)
            
            # Attempt to parse JSON
            return json.loads(decoded_str)

        except (UnicodeDecodeError, json.JSONDecodeError):
            # If decoding or JSON parsing fails, move to the next encoding
            continue
        except Exception as e:
            # Catch any other exceptions and continue to the next encoding
            print(f"Error with encoding {encoding}: {e}")
            continue

    # If all attempts fail, raise an error
    raise ValueError("Failed to decode JSON with all attempted strategies.")


def getSpeckleStream(stream_id,
                     branch_name,
                     client,
                     commit_id=""
                     ):
    """
    Retrieves data from a specific branch of a speckle stream.

    Args:
        stream_id (str): The ID of the speckle stream.
        branch_name (str): The name of the branch within the speckle stream.
        client (specklepy.api.client.Client, optional): A speckle client. Defaults to a global `client`.
        commit_id (str): id of a commit, if nothing is specified, the latest commit will be fetched

    Returns:
        dict: The speckle stream data received from the specified branch.

    This function retrieves the last commit from a specific branch of a speckle stream.
    It uses the provided speckle client to get the branch and commit information, and then 
    retrieves the speckle stream data associated with the last commit.
    It prints out the branch details and the creation dates of the last three commits for debugging purposes.
    """


    # set stream and branch
    try:
        branch = client.branch.get(stream_id, branch_name, 3)
        print(branch)
    except:
        branch = client.branch.get(stream_id, branch_name, 1)
        print(branch)

    print("last three commits:")
    [print(ite.createdAt) for ite in branch.commits.items]

    if commit_id == "":
        latest_commit = branch.commits.items[0]
        choosen_commit_id = latest_commit.id
        commit = client.commit.get(stream_id, choosen_commit_id)
        print("latest commit ", branch.commits.items[0].createdAt, " was choosen")
    elif type(commit_id) == type("s"): # string, commit uuid
        choosen_commit_id = commit_id
        commit = client.commit.get(stream_id, choosen_commit_id)
        print("provided commit ", choosen_commit_id, " was choosen")
    elif type(commit_id) == type(1): #int 
        latest_commit = branch.commits.items[commit_id]
        choosen_commit_id = latest_commit.id
        commit = client.commit.get(stream_id, choosen_commit_id)


    print(commit)
    print(commit.referencedObject)
    # get transport
    transport = ServerTransport(client=client, stream_id=stream_id)
    #speckle stream
    res = operations.receive(commit.referencedObject, transport)

    return res



def update_graph_data():
    """
    Function to update all charts dynamically by fetching new data and updating ColumnDataSources and Divs.
    """
    global data_source_mode, client

    # 1. DETERMINE DATA SOURCE MODE AND AUTHENTICATE IF NECESSARY -------------------
    print (data_source_mode)
    if data_source_mode == "speckle":
        # Read values from the Speckle widgets
        speckle_token = speckle_token_input.value.strip()
        speckle_stream_id = speckle_stream_id_input.value.strip()
        speckle_branch_name = speckle_branch_name_input.value.strip()
        speckle_commit_id_cur = speckle_commit_id_current.value.strip()
        speckle_commit_id_prev = speckle_commit_id_previous.value.strip()
        print("speckle stream id",speckle_stream_id)
        # Check if essential fields are provided    
        if not speckle_token or not speckle_stream_id or not speckle_branch_name:
            print("Error: Speckle token, stream ID, and branch name must be provided.")
            return

        # Authenticate client with provided token
        client = SpeckleClient(host="https://speckle.xyz")
        client.authenticate(token=speckle_token)

        # Default to using commit 0 and commit 1 if commit IDs are not specified
        commit_id_current = speckle_commit_id_cur if speckle_commit_id_cur else 0
        commit_id_previous = speckle_commit_id_prev if speckle_commit_id_prev else 1

        # Fetch data from Speckle
        current_graph = get_gh_graph(speckle_stream_id, speckle_branch_name, client, commit_id_current)
        previous_graph = get_gh_graph(speckle_stream_id, speckle_branch_name, client, commit_id_previous)

    elif data_source_mode == "json":
        # Parse uploaded JSON files
        if not json_upload_current.value or not json_upload_previous.value:
            print("Error: Both current and previous JSON files must be uploaded.")
            return

        try:
            # Decode the base64-encoded JSON strings+
            current_graph = robust_json_decode(json_upload_current.value)
            previous_graph = robust_json_decode(json_upload_previous.value)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return
        except (KeyError, TypeError) as e:
            print(f"Error: Invalid JSON structure: {e}")
            return


    else:
        print("Error: Unknown data source mode.")
        return

    # 2. PROCESS AND UPDATE THE GRAPHS ------------------------------------------------
    # Generate NetworkX graphs
    global cur_graphX
    global prev_graphX
    global cur_graphX_reverse
    global prev_graphX_reverse
    cur_graphX, cur_graphX_reverse = generateGraph(current_graph)
    prev_graphX, prev_graphX_reverse = generateGraph(previous_graph)

    # Compute differences between versions
    added, removed, changed = node_comparison(previous_graph, current_graph, attributes_to_track=attribute_change_tracker)

    # 3. UPDATE MAIN CHARTS (Current and Previous Graphs) --------------------------
    # Update data sources for current and previous graphs
    new_node_source_current, new_edge_source_current, new_highlighted_edges_source_current = graph_visualization_datasources(
        cur_graphX, attributes_to_include, added, removed, changed
    )
    new_node_source_previous, new_edge_source_previous, new_highlighted_edges_source_previous = graph_visualization_datasources(
        prev_graphX, attributes_to_include, added, removed, changed
    )

    # Update the data for the Current Graph
    node_source_current.data = dict(new_node_source_current.data)
    edge_source_current.data = dict(new_edge_source_current.data)
    highlighted_edges_source_current.data = dict(new_highlighted_edges_source_current.data)

    # Update the data for the Previous Graph
    node_source_previous.data = dict(new_node_source_previous.data)
    edge_source_previous.data = dict(new_edge_source_previous.data)
    highlighted_edges_source_previous.data = dict(new_highlighted_edges_source_previous.data)

    # 4. UPDATE MORPH VIEW CHART ---------------------------------------------------
    # Compute new data for morph view chart
    new_interpolated_node_source, new_unchanged_node_source, new_interpolated_edge_source, new_unchanged_edge_source = morphViewCDS(
        node_source_current.data, node_source_previous.data
    )
    global transition_states 
    transition_states = compute_transition_states(
        node_source_previous.data, node_source_current.data, new_interpolated_node_source.data["instanceUUID"]
    )

    # Update the morph view data sources
    interpolated_node_source.data = dict(new_interpolated_node_source.data)
    unchanged_node_source.data = dict(new_unchanged_node_source.data)
    interpolated_edge_source.data = dict(new_interpolated_edge_source.data)
    unchanged_edge_source.data = dict(new_unchanged_edge_source.data)

    # 5. STATS EXTRACTION AND UPDATES ----------------------------------------------
    # Extract statistics for bar chart and other potential uses
    stats_current = extractStats(node_source_current.data, node_source_previous.data)
    stats_previous = extractStats(node_source_previous.data)  # Only the original stats
    total_nodes = len(node_source_current.data["kind"])

    # 6. BAR CHART UPDATE ----------------------------------------------------------
    # Create new bar chart data and update the source
    new_bar_chart_source = create_bar_chart_data(stats_current, total_nodes)
    bar_chart_source.data = dict(new_bar_chart_source.data)

    # 7. RANK VALUE PLOT UPDATE ----------------------------------------------------
    # Create new data for rank value plot and update the source
    new_rank_value_source = create_rank_value_data(node_source_current, node_source_previous)
    rank_value_source.data = dict(new_rank_value_source.data)

    # Update the rank slider's attributes based on new data
    max_rank = len(new_rank_value_source.data['computiationTime'])
    rank_slider.start = 1
    rank_slider.end = max_rank
    rank_slider.value = max_rank  # Reset the slider to max value to show full range

    # 8. DIV UPDATES ---------------------------------------------------------------
    # Update change_stats_div content
    change_stats_div.text = create_stats_div_text(stats_current)

    # Update comparison Divs for each attribute
    for key, attr_name in attribute_names.items():
        cur_value = stats_current[key]
        prev_value = stats_previous[key]
        update_comparison_div(attr_name, cur_value, prev_value)








# Speckle Tab Widgets
speckle_token_input = PasswordInput(title="Speckle Token", placeholder="Enter your token...")
speckle_stream_id_input = TextInput(title="Speckle Stream ID", placeholder="Enter your stream ID")
speckle_branch_name_input = TextInput(title="Speckle Branch Name", placeholder="Enter your branch name")
speckle_commit_id_current = TextInput(title="Commit ID (Current, optional)",placeholder="optional")
speckle_commit_id_previous = TextInput(title="Commit ID (Previous, optional)",placeholder="optional")
speckle_fetch_button = Button(label="Generate Graph", width=100)


speckle_fetch_button.on_click(update_graph_data)

# Layout for Speckle Tab
speckle_layout = column(
    speckle_token_input,
    speckle_stream_id_input,
    speckle_branch_name_input,
    speckle_commit_id_current,
    speckle_commit_id_previous,
    
)

# JSON Tab Widgets with Divs as titles
json_upload_current_title = Div(text="JSON of Current Graph")
json_upload_current = FileInput(accept=".json")

json_upload_previous_title = Div(text="JSON of Previous Graph")
json_upload_previous = FileInput(accept=".json")

# Layout for JSON Tab
json_layout = column(
    json_upload_current_title,
    json_upload_current,
    json_upload_previous_title,
    json_upload_previous,
)

# UUID Syncing Widgets
uuid_sync_stream_id_input = TextInput(title="Speckle Stream ID for UUID Syncing (Optional)", placeholder="optional")
uuid_sync_branch_name_input = TextInput(title="Speckle Branch Name for UUID Syncing (Optional)", placeholder="optional")

uuid_sync_layout = column(
    uuid_sync_stream_id_input,
    uuid_sync_branch_name_input,
    speckle_fetch_button
)



# Function to update data source mode when JSON is uploaded
def on_json_upload(attr, old, new):
    global data_source_mode
    if json_upload_current.value or json_upload_previous.value:
        data_source_mode = "json"
        
# Function to update data source mode when Speckle fields are filled
def on_speckle_input(attr, old, new):
    global data_source_mode
    if speckle_token_input.value or speckle_stream_id_input.value or speckle_branch_name_input.value:
        data_source_mode = "speckle"

# Attach event handlers to widgets
json_upload_current.on_change('value', on_json_upload)
json_upload_previous.on_change('value', on_json_upload)
speckle_token_input.on_change('value', on_speckle_input)
speckle_stream_id_input.on_change('value', on_speckle_input)
speckle_branch_name_input.on_change('value', on_speckle_input)

# ===== SPECKLE Authentication and data retrieval =====

attributes_to_include = [
    'instanceGuid', 'pivot', 'computiationTime', 'subgraphID', 
    'bounds', 'name', 'description', 'kind', 'sources', 
    'pathCount', 'componentGuid', 'endPoint', 'subCategory', 
    'targets', 'dataType', 'category', 'nickName', 'simplify', 
    'dataMapping', 'startPoint', 'dataCount'
]



# generate differences between t and t-1 (-> as attributes on the graph) ====
attribute_change_tracker = [
     'pivot', 'name', 'description',  'sources', 
     'targets', 'dataType', 'nickName', 'simplify', 
    'dataMapping', 
]


# we could compute relative differences for the following attributes
# computiationTime, pathCount, dataCount,  targets, sources




def get_gh_graph(streamID, branchName, client, commitID=0):
    res = getSpeckleStream(streamID,
                        branchName,
                        client,commitID)

    # get gh graph data 
    res_new = copy.deepcopy(res)
    main_graph_new = res_new["@main_graph"]["@{0}"][0]["@main_graph"]


    main_graph_json = json.loads(main_graph_new)
    #cluster_graph_json = json.loads(cluster_graph_new)     
    print(next(iter(main_graph_json.values())).keys())


    return main_graph_json


# Get the absolute path to the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def robust_json_load_from_file(file_path):
    """
    Loads and decodes a JSON file, handling various encodings and potential BOMs.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Loaded JSON object.
        
    Raises:
        ValueError: If the JSON loading fails after all attempts.
    """
    # Define potential encodings to try
    encodings = ['utf-8-sig', 'utf-8', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # If decoding or JSON parsing fails, move to the next encoding
            continue
        except Exception as e:
            # Catch any other exceptions and print error, then continue to the next encoding
            print(f"Error with encoding {encoding}: {e}")
            continue

    # If all attempts fail, raise an error
    raise ValueError(f"Failed to load JSON file with all attempted strategies: {file_path}")

# Function to load JSON data from files
def load_initial_json_data():
    """
    Load initial graph data from JSON files.
    
    Returns:
    - Tuple of two dictionaries representing the current and previous graphs.
    """
    # Get the absolute path to the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the absolute paths to the JSON files
    current_graph_path = os.path.join(script_dir, "graph.json")
    previous_graph_path = os.path.join(script_dir, "graph_before.json")

    try:
        # Use the robust function to load JSON data
        current_graph = robust_json_load_from_file(current_graph_path)
        previous_graph = robust_json_load_from_file(previous_graph_path)
        
    except FileNotFoundError as e:
        print(f"Error: JSON file not found: {e}")
        return None, None
    except ValueError as e:
        print(f"Error decoding JSON file: {e}")
        return None, None

    return current_graph, previous_graph

# Load initial data
current_graph, previous_graph = load_initial_json_data()



# generate networkX graph
# Initialize a directed graph
def generateGraph(data):
    G = nx.DiGraph()

    # Add nodes to the graph
    for key, value in data.items():
        G.add_node(key, **value)

    # Add edges based on the 'sources' and 'targets' attributes
    for key, value in data.items():
        #for source in value['sources']:
        #    G.add_edge(source, key)
        for target in value['targets']:
            G.add_edge(key, target)

    # Return basic information about the graph
    graph_info = nx.info(G)
    G_reversed = G.reverse(copy=True)

    return G, G_reversed


def graph2CDS(G, attributes_to_include, added=None, removed=None, changed=None):
    nodes = list(G.nodes(data=True))

    def reorder_bounds(bounds):
        return [bounds[0], bounds[2], bounds[3], bounds[1], bounds[0]]

    node_data = {
        'index': [node[0] for node in nodes],
        'x': [node[1]['pivot'][0] for node in nodes],
        'y': [node[1]['pivot'][1] for node in nodes],
        'bounds_x': [[coord[0] for coord in reorder_bounds(node[1]['bounds'])] for node in nodes],
        'bounds_y': [[coord[1] for coord in reorder_bounds(node[1]['bounds'])] for node in nodes],
        'change_summary': ['' for _ in nodes],
        'node_status': ['unchanged' for _ in nodes],
        'nickname': [node[1].get('nickname', 'N/A') for node in nodes],  # <-- Added here
        'instanceUUID': [node[1].get('instanceGuid', 'N/A') for node in nodes]  # <-- Added here
    }


    for attribute in attributes_to_include:
        node_data[attribute] = [node[1].get(attribute, None) for node in nodes]
        node_data[f'{attribute}_changed'] = [False for _ in nodes]

    # Add ranks
    # Sort based on computation time and get ranks
    computation_times = node_data['computiationTime']
    ranks = sorted(range(len(computation_times)), key=lambda k: computation_times[k], reverse=True)
    
    # Adjust the ranks as sorted() returns indices of sorted elements and not the rank itself
    adjusted_ranks = [0] * len(ranks)
    for idx, rank in enumerate(ranks):
        adjusted_ranks[rank] = idx + 1  # Adding 1 because ranks typically start from 1, not 0

    # Add ranks to node_data
    node_data['rank'] = adjusted_ranks

    for i, node in enumerate(nodes):
        node_id = node[0]
        change_list = []
        if added and node_id in added:
            node_data['node_status'][i] = 'added'
            change_list.append("<div style='font-weight: bold; padding: 5px;'>Added</div>")
        elif removed and node_id in removed:
            node_data['node_status'][i] = 'removed'
            change_list.append("<div style='font-weight: bold; padding: 5px;'>Removed</div>")
        elif changed and node_id in changed:
            is_only_position_changed = 'pivot' in changed[node_id] and len(changed[node_id]) == 1
            if is_only_position_changed:
                node_data['node_status'][i] = 'position_changed'
            else:
                node_data['node_status'][i] = 'changed'
            
            for attr in changed[node_id]:  # Iterating through changed attributes of the current node
                if f'{attr}_changed' in node_data:
                    node_data[f'{attr}_changed'][i] = True

            #change_list = []

            # For "added" and "removed" statuses, we can check directly in the node_data
            status = node_data['node_status'][i]
            #if status == 'added':
            #    change_list.append("<div style='font-weight: bold; padding: 5px;'>Added</div>")
            #elif status == 'removed':
            #    change_list.append("<div style='font-weight: bold; padding: 5px;'>Removed</div>")
            if status == 'position_changed':
                change_list.append("<div style='font-weight: bold; padding: 5px;'>Location changed</div>")
            else:
                # Check if there are any changed attributes
                has_changed_attributes = any(attr in changed[node_id] for attr in attributes_to_include if changed.get(node_id))

                if has_changed_attributes:
                    # Add column names only if there are changes
                    change_list.append("<div style='display: flex; justify-content: space-between; padding: 2px 0; color: lightgray;'>"
                                    "<span><b>Attribute</b></span>"
                                    "<span><b>Current</b></span>"
                                    "<span><b>Previous</b></span></div>")

                for attr in attributes_to_include:
                    if attr in changed[node_id]:
                        change_row = f"<div style='display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 2px 0;'>"
                        change_row += f"<span><b>{attr}</b></span>"
                        change_row += f"<span>{changed[node_id][attr]['new']}</span>"
                        change_row += f"<span style='background-color: #FFEBCC;'>{changed[node_id][attr]['old']}</span></div>"
                        change_list.append(change_row)

        node_data['change_summary'][i] = ''.join(change_list)
        


    node_source = ColumnDataSource(data=node_data)
    #node_source.add(data={'visibility_helper': ['1' for _ in range(len(node_source.data['name']))]}, name='visibility_helper')

    # For edges 
    xs = []
    ys = []
    for start, end in G.edges():
        start_node = G.nodes[start]
        end_node = G.nodes[end]
        xs.append([start_node['pivot'][0], end_node['pivot'][0]])
        ys.append([start_node['pivot'][1], end_node['pivot'][1]])

    edge_source = ColumnDataSource(data=dict(xs=xs, ys=ys))
    return node_source, edge_source



def node_comparison(old_graph, new_graph, attributes_to_track):
    # Get sets of node IDs
    old_nodes = set(old_graph.keys())
    new_nodes = set(new_graph.keys())
    
    # Determine added and removed nodes
    added_nodes = new_nodes - old_nodes
    removed_nodes = old_nodes - new_nodes
    
    # Determine changed nodes
    changed_nodes = {}
    for node_id in old_nodes.intersection(new_nodes):
        old_node = old_graph[node_id]
        new_node = new_graph[node_id]
        changed_attributes = {}
        
        for attribute in attributes_to_track:
            if old_node.get(attribute) != new_node.get(attribute):
                changed_attributes[attribute] = {"old": old_node.get(attribute), "new": new_node.get(attribute)}
        
        if changed_attributes:
            changed_nodes[node_id] = changed_attributes
            
    return added_nodes, removed_nodes, changed_nodes


# generate networkX graphs
global cur_graphX
global prev_graphX
global cur_graphX_reverse
global prev_graphX_reverse
cur_graphX, cur_graphX_reverse = generateGraph(current_graph)
prev_graphX, prev_graphX_reverse  = generateGraph(previous_graph)


# compute differences between versions
added, removed, changed = node_comparison(previous_graph, current_graph, attributes_to_track=attribute_change_tracker)

def update_graph_data():
    #TODO 
    pass



def graph_visualization_datasources(G, attributes_to_include, added=None, removed=None, changed=None):
    """
    Generates ColumnDataSource objects for nodes and edges from the graph G.
    """
    node_source, edge_source = graph2CDS(G, attributes_to_include, added=added, removed=removed, changed=changed)
    highlighted_edges_source = ColumnDataSource(data=dict(xs=[], ys=[], color=[]))
    
    return node_source, edge_source, highlighted_edges_source


# generate bokeh graph
def setup_graph_visualization(node_source, edge_source, highlighted_edges_source, title="someGraph"):
    """
    Sets up the Bokeh plot using pre-existing data sources.
    """
    status_colors = {
        'added': 'green',
        'removed': 'red',
        'changed': 'blue',
        'unchanged': 'gray',
        'position_changed': 'purple'
    }

    color_mapper = factor_cmap('node_status', palette=list(status_colors.values()), factors=list(status_colors.keys()))

    p = figure(plot_width=800, plot_height=900, title=title, tools="pan,box_zoom,wheel_zoom,reset,tap,box_select",
               sizing_mode="stretch_both", name=title)
    p.toolbar.logo = None

    # Draw nodes with patches
    node_glyph = p.patches('bounds_x', 'bounds_y', source=node_source, fill_color=color_mapper, 
                           hover_line_color="black", hover_line_width=3, hover_line_alpha=0.8,
                           alpha=0.6, line_color="white", selection_line_color="orange", selection_line_alpha=0.8, selection_line_width=3)

    # Set up legend callback
    callback_legend = CustomJS(args=dict(source=node_source), code="""
        const glyph = cb_obj;  // The dummy glyph that triggered the callback
        const clicked = glyph.name;  // Using the 'name' attribute to identify the status
        const data = source.data;
        const statuses = data['node_status'];
        let selected_indices = [];

        for (let i = 0; i < statuses.length; i++) {
            if (statuses[i] === clicked) {
                selected_indices.push(i);
            }
        }

        source.selected.indices = selected_indices;
        source.change.emit();
        glyph.visible = false;  // Ensure the dummy glyph remains invisible after the callback
    """)

    # Create dummy glyphs for legend
    legend_glyphs = []
    for status, color in status_colors.items():
        dummy_glyph = p.scatter([0], [0], color=color, alpha=0.9, line_color="white", name=status, visible=False, muted_alpha=0.9, muted_color=color)
        dummy_glyph.js_on_change("visible", callback_legend)  # Attach the callback to the 'visible' property change
        legend_glyphs.append(dummy_glyph)

    legend_items = [
        LegendItem(label="added", renderers=[legend_glyphs[0]]),
        LegendItem(label="removed", renderers=[legend_glyphs[1]]),
        LegendItem(label="changed", renderers=[legend_glyphs[2]]),
        LegendItem(label="unchanged", renderers=[legend_glyphs[3]])
    ]
    
    outside_legend = Legend(items=legend_items, click_policy='hide', location="bottom_center", orientation="horizontal")
    p.add_layout(outside_legend, 'below')

    # Draw edges
    p.multi_line('xs', 'ys', source=edge_source, line_width=0.5, color="black", alpha=0.6)
    p.multi_line('xs', 'ys', color='color', source=highlighted_edges_source, line_width=2)

    # Draw text labels
    text_glyph = p.text(x='x', y='y', text='name', source=node_source, text_font_size="7pt", text_align="center", text_baseline="middle")
    text_glyph.visible = False

    # Configure axes
    p.xaxis.major_label_text_font_size = '0pt'
    p.yaxis.major_label_text_font_size = '0pt'
    p.xaxis.minor_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xaxis.major_tick_line_color = None
    p.yaxis.major_tick_line_color = None
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.axis_line_color = "gray"
    p.xaxis.axis_line_alpha = 0.0
    p.yaxis.axis_line_color = "gray"
    p.yaxis.axis_line_alpha = 0.0
    p.outline_line_color = "lightgrey"
    
    return p, node_glyph, text_glyph



def update_data_sources(node_source, edge_source, highlighted_edges_source, G, attributes_to_include, added=None, removed=None, changed=None):
    """
    Updates the existing ColumnDataSource objects with new data from the graph G.
    """
    new_node_source, new_edge_source = graph2CDS(G, attributes_to_include, added=added, removed=removed, changed=changed)

    # Update the data in existing sources
    node_source.data.update(new_node_source.data)
    edge_source.data.update(new_edge_source.data)

    # Clear the highlighted edges
    highlighted_edges_source.data.update(dict(xs=[], ys=[], color=[]))

#=======================================================

# Step 1: Create initial data sources for current and previous graphs
node_source_current, edge_source_current, highlighted_edges_source_current = graph_visualization_datasources(cur_graphX, attributes_to_include, added, removed, changed)
node_source_previous, edge_source_previous, highlighted_edges_source_previous = graph_visualization_datasources(prev_graphX, attributes_to_include, added, removed, changed)

# Step 2: Set up the visualizations using these data sources
p_current, node_glyph_cur, text_glyph_cur = setup_graph_visualization(node_source_current, edge_source_current, highlighted_edges_source_current, title="Current Graph")
p_previous, node_glyph_prev, text_glyph_prev = setup_graph_visualization(node_source_previous, edge_source_previous, highlighted_edges_source_previous, title="Previous Graph")


# Generate visualizations for each graph
#p_current, node_source_current, edge_source_current, highlighted_edges_source_current, node_glyph_cur, text_glyph_cur = setup_graph_visualization(cur_graphX, added, removed, changed, title="Current Graph")
#p_previous, node_source_previous, edge_source_previous, highlighted_edges_source_previous, node_glyph_prev, text_glyph_prev = setup_graph_visualization(prev_graphX, added, removed, changed, title="Previous Graph")

# Link the x and y ranges of the plots
p_previous.x_range = p_current.x_range
p_previous.y_range = p_current.y_range


# Outside the function, create the callback:
callback_text_cur = CustomJS(args=dict(text_glyph=text_glyph_cur), code="""
    var zoom_level = cb_obj.end - cb_obj.start;
    if (zoom_level < 800) {
        text_glyph.visible = true;
    } else {
        text_glyph.visible = false;
    }
""")
                             
# Outside the function, create the callback:
callback_text_prev = CustomJS(args=dict(text_glyph=text_glyph_prev), code="""
    var zoom_level = cb_obj.end - cb_obj.start;
    if (zoom_level < 800) {
        text_glyph.visible = true;
    } else {
        text_glyph.visible = false;
    }
""")                         


p_current.x_range.js_on_change('start', callback_text_cur)
p_current.x_range.js_on_change('end', callback_text_prev)

#=== hover tooltip


num = 1
visibility_formatter = CustomJSHover(code=f"""
    special_vars.indices = special_vars.indices.slice(0, {num});
    return special_vars.indices.includes(special_vars.index) ? "" : "hidden";
""")


hover_html = """
<div style=" background-color: white; border-radius: 8px; box-shadow: 2px 2px 12px #aaa; padding: 10px;" @name{custom}>
    <div style="font-size: 1.1em; font-weight: bold; margin-bottom: 5px;">@name</div>
    <div style="font-size: 0.8em; margin-bottom: 10px;"><span>@nickName</span> | <span>@instanceGuid</span></div>
    <div style="border-top: 1px solid #aaa;">
        @change_summary
    </div>
</div>
"""


hover_tool_current = HoverTool(renderers=[node_glyph_cur], tooltips=hover_html, mode='mouse', formatters={'@name': visibility_formatter})
p_current.add_tools(hover_tool_current)

hover_tool_previous = HoverTool(renderers=[node_glyph_prev], tooltips=hover_html, mode='mouse', formatters={'@name': visibility_formatter})
p_previous.add_tools(hover_tool_previous)



# ================ morph vie figure ==================
def morphViewCDS(node_data_current, node_data_previous):
    #========== morph view ==============
    interpolated_node_data = {
        'index': [],
        'x': [],
        'y': [],
        'bounds_x': [],
        'bounds_y': [],
        'change_summary': [],
        'node_status': [],
        'nickname': [],
        'instanceUUID': [],
        'color': [],
        'alpha': [],
    }
    unchanged_node_data = {
        'index': [],
        'x': [],
        'y': [],
        'bounds_x': [],
        'bounds_y': [],
        'change_summary': [],
        'node_status': [],
        'nickname': [],
        'instanceUUID': [],
    }

    status_colors = {
    'added': 'green',
    'removed': 'red',
    'changed': 'blue',
    'unchanged': 'gray',
    'position_changed': 'purple' 
    }

    # Create sets of uuids from both current and previous node datasets
    current_uuids = set(node_data_current['instanceUUID'])
    previous_uuids = set(node_data_previous['instanceUUID'])

    # Iterate over current nodes
    for i, uuid in enumerate(node_data_current['instanceUUID']):
        if uuid not in previous_uuids:  # The node has been added
            for key in interpolated_node_data:
                if key in node_data_current:
                    interpolated_node_data[key].append(node_data_current[key][i])
        elif node_data_current['node_status'][i] in ['changed', 'position_changed']:  # The node has been changed
            for key in interpolated_node_data:
                if key in node_data_current:
                    interpolated_node_data[key].append(node_data_current[key][i])
        else:  # The node is unchanged
            for key in unchanged_node_data:
                if key in node_data_current:
                    unchanged_node_data[key].append(node_data_current[key][i])

    # Iterate over previous nodes for removed nodes
    for i, uuid in enumerate(node_data_previous['instanceUUID']):
        if uuid not in current_uuids:  # The node has been removed
            for key in interpolated_node_data:
                if key in node_data_previous:
                    interpolated_node_data[key].append(node_data_previous[key][i])
    # add default alpha
    interpolated_node_data['alpha'] = [1] * len(interpolated_node_data['index'])
    interpolated_node_data['color'] = ["grey"] * len(interpolated_node_data['index'])

    # Convert to CDS
    interpolated_node_source = ColumnDataSource(data=interpolated_node_data)
    unchanged_node_source = ColumnDataSource(data=unchanged_node_data)

    
    # ============ For edges =====================
    interpolated_edge_data = {
        'xs': [],
        'ys': [],
        'color': [],
        'alpha': [],
    }
    unchanged_edge_data = {
        'xs': [],
        'ys': [],
    }

    # Helper function to get coordinates for given uuid
    def get_coordinates(node_data, uuid_lookup, uuid):
        index = uuid_lookup[uuid]
        return node_data['x'][index], node_data['y'][index]

    # Flatten a list of lists
    def flatten(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

    # Extract sources and targets
    current_targets = node_data_current['targets']
    previous_targets = node_data_previous['targets']

    # Convert UUIDs to sets for faster lookup
    current_uuids = set(flatten(current_targets))
    previous_uuids = set(flatten(previous_targets))

    # Create a dictionary for faster UUID lookup
    uuid_lookup_current = {uuid: index for index, uuid in enumerate(node_data_current['instanceUUID'])}
    uuid_lookup_previous = {uuid: index for index, uuid in enumerate(node_data_previous['instanceUUID'])}
   
    # Iterate over each node's targets
    for idx, node_targets in enumerate(current_targets):
        node_uuid = node_data_current['instanceUUID'][idx]
        for target in node_targets:
            if target not in uuid_lookup_current:
                #print(f"Warning: UUID {target} not found in current data for target.")
                continue
            
            source_coords = get_coordinates(node_data_current, uuid_lookup_current, node_uuid)
            target_coords = get_coordinates(node_data_current, uuid_lookup_current, target)
            
            source_status = node_data_current['node_status'][idx]
            target_status = node_data_current['node_status'][uuid_lookup_current[target]]

            if source_status == 'unchanged' and target_status == 'unchanged':
                unchanged_edge_data['xs'].append([source_coords[0], target_coords[0]])
                unchanged_edge_data['ys'].append([source_coords[1], target_coords[1]])
            else:
                interpolated_edge_data['xs'].append([source_coords[0], target_coords[0]])
                interpolated_edge_data['ys'].append([source_coords[1], target_coords[1]])

    # Iterate over previous edges for removed nodes
    for idx, previous_targets in enumerate(previous_targets):
        
        # This node's UUID from previous data
        node_uuid = node_data_previous['instanceUUID'][idx]

        # Check each UUID in previous targets
        for target in previous_targets:
            if target not in current_uuids:
                source_coords = get_coordinates(node_data_previous, uuid_lookup_previous, node_uuid)
                target_coords = get_coordinates(node_data_previous, uuid_lookup_previous, target)
                interpolated_edge_data['xs'].append([source_coords[0], target_coords[0]])
                interpolated_edge_data['ys'].append([source_coords[1], target_coords[1]])

    interpolated_edge_data['color'] = ["grey"] * len(interpolated_edge_data['xs'])
    interpolated_edge_data['alpha'] = [1] * len(interpolated_edge_data['xs'])
    # Convert to CDS
    interpolated_edge_source = ColumnDataSource(data=interpolated_edge_data)
    unchanged_edge_source = ColumnDataSource(data=unchanged_edge_data)

    return interpolated_node_source, unchanged_node_source, interpolated_edge_source, unchanged_edge_source

def interpolate_color(start_color, end_color, factor):
    """
    Interpolate between two colors.
    :param start_color: tuple of RGB values for the starting color.
    :param end_color: tuple of RGB values for the ending color.
    :param factor: interpolation factor (0 means start_color, 1 means end_color).
    :return: interpolated RGB color as a tuple.
    """
    return (
        int(start_color[0] + factor * (end_color[0] - start_color[0])),
        int(start_color[1] + factor * (end_color[1] - start_color[1])),
        int(start_color[2] + factor * (end_color[2] - start_color[2]))
    )

def get_edge_id(source_uuid, target_uuid):
    return f"{source_uuid}->{target_uuid}"

def edge_status(previous_node_targets, current_node_targets, target):
    if target not in previous_node_targets:
        return 'added'
    elif target in previous_node_targets and target not in current_node_targets:
        return 'removed'
    elif target in previous_node_targets:
        return 'unchanged'
    else:
        raise ValueError("Unexpected edge status.")


def compute_transition_states(previous_graph, current_graph, interpolated_node_source_instanceUUID, steps=31):
    status_colors = {
        'added': 'green',
        'removed': 'red',
        'changed': 'blue',
        'unchanged': 'gray',
        'position_changed': 'purple'
    }
    status_colors_rgb = {
        'added': (0, 255, 0),        # green
        'removed': (255, 0, 0),     # red
        'changed': (0, 0, 255),     # blue
        'unchanged': (128, 128, 128), # gray
        'position_changed': (128, 0, 128) # purple
    }

    transition_states = []
    uuid_lookup_current = {uuid: index for index, uuid in enumerate(current_graph['instanceUUID'])}
    uuid_lookup_previous = {uuid: index for index, uuid in enumerate(previous_graph['instanceUUID'])}

    def get_coordinates(node_data, uuid_lookup, uuid):
            index = uuid_lookup[uuid]
            return node_data['x'][index], node_data['y'][index]
    
    
    for step in range(steps):
        state = {'nodes': [], 'edges': []}
        # PHASE 1: (0-9)
        if 0 <= step < 10:
            alpha_factor = step / 10.0  # Used to fade in/out elements

            # RGB for white
            white_rgb = (255, 255, 255)

            # Keep track of UUIDs processed
            processed_uuids = set()

            # Handling nodes from previous_graph
            for uuid in previous_graph['instanceUUID']:
                if uuid not in interpolated_node_source_instanceUUID:
                    continue
                
                index = previous_graph['instanceUUID'].index(uuid)
                node = {key: val[index] for key, val in previous_graph.items()}

                # Determine the end color
                end_color_rgb = status_colors_rgb[node['node_status']]
                
                # If the node is "added", interpolate between white and its final color
                if node['node_status'] == 'added':
                    interpolated_color = interpolate_color(white_rgb, end_color_rgb, alpha_factor)
                else:
                    # Otherwise, interpolate between gray (unchanged) and the final color
                    interpolated_color = interpolate_color(status_colors_rgb['unchanged'], end_color_rgb, alpha_factor)
                
                # Convert RGB to hex format for Bokeh
                node['color'] = '#%02x%02x%02x' % interpolated_color
                node['alpha'] = 1  # Set alpha to 1 since we're only dealing with colors now
                state['nodes'].append(node)

                # Mark this UUID as processed
                processed_uuids.add(uuid)

            # Handling nodes from current_graph
            for uuid in current_graph['instanceUUID']:
                if uuid not in interpolated_node_source_instanceUUID or uuid in processed_uuids:
                    continue

                index = current_graph['instanceUUID'].index(uuid)
                node = {key: val[index] for key, val in current_graph.items()}

                # Determine the end color
                end_color_rgb = status_colors_rgb[node['node_status']]
                
                # If the node is "added", interpolate between white and its final color
                if node['node_status'] == 'added':
                    interpolated_color = interpolate_color(white_rgb, end_color_rgb, alpha_factor)
                else:
                    # Otherwise, interpolate between gray (unchanged) and the final color
                    interpolated_color = interpolate_color(status_colors_rgb['unchanged'], end_color_rgb, alpha_factor)
                
                # Convert RGB to hex format for Bokeh
                node['color'] = '#%02x%02x%02x' % interpolated_color
                node['alpha'] = 1  # Keep alpha at 1 for these nodes as well
                state['nodes'].append(node)

            # Handling edges in Phase 1 =========================================
            # Use unchanged_edge_source for the base
            """
            for xs, ys in zip(unchanged_edge_source.data['xs'], unchanged_edge_source.data['ys']):
                edge = {
                    'xs': xs,
                    'ys': ys,
                    'color': status_colors['unchanged'],
                    'alpha': 1
                }
                state['edges'].append(edge)
            """

            # handle new edges, fade in new edges 
            for idx, node_targets in enumerate(current_graph['targets']):
                node_uuid = current_graph['instanceUUID'][idx]

                # Get the corresponding targets from the previous graph (if the node exists)
                previous_node_targets = []
                if node_uuid in uuid_lookup_previous:
                    previous_idx = uuid_lookup_previous[node_uuid]
                    previous_node_targets = previous_graph['targets'][previous_idx]

                for target in node_targets:
                    if target not in previous_node_targets:
                        try:
                            source_coords = get_coordinates(previous_graph, uuid_lookup_previous, node_uuid)
                            target_coords = get_coordinates(previous_graph, uuid_lookup_previous, target)
                        except:
                            source_coords = get_coordinates(current_graph, uuid_lookup_current, node_uuid)
                            target_coords = get_coordinates(current_graph, uuid_lookup_current, target)

                        state['edges'].append({
                            'xs': [source_coords[0], target_coords[0]],
                            'ys': [source_coords[1], target_coords[1]],
                            'color': status_colors['added'],
                            'alpha': alpha_factor # Fade in
                        })

            
            # Handle removed edges (color fade from gray to red)
            for idx, previous_targets in enumerate(previous_graph['targets']):
                node_uuid = previous_graph['instanceUUID'][idx]
                
                current_node_targets = []
                if node_uuid in uuid_lookup_current:
                    current_idx = uuid_lookup_current[node_uuid]
                    current_node_targets = current_graph['targets'][current_idx]
                for target in previous_targets:
                    if target not in current_node_targets:
                    
                        source_coords = get_coordinates(previous_graph, uuid_lookup_previous, node_uuid)
                        target_coords = get_coordinates(previous_graph, uuid_lookup_previous, target)
                        interpolated_edge_color = interpolate_color(status_colors_rgb['unchanged'], status_colors_rgb['removed'], alpha_factor)
                        
                        state['edges'].append({
                            'xs': [source_coords[0], target_coords[0]],
                            'ys': [source_coords[1], target_coords[1]],
                            'color': '#%02x%02x%02x' % interpolated_edge_color,
                            'alpha': 1
                        })



            # Handle edges that have changed positions in the previous_graph state
            for idx, current_targets in enumerate(current_graph['targets']):
                node_uuid = current_graph['instanceUUID'][idx]
                if node_uuid in uuid_lookup_previous:
                    previous_idx = uuid_lookup_previous[node_uuid]
                    previous_node_targets = previous_graph['targets'][previous_idx]

                    for target in current_targets:
                        #if target in previous_node_targets:
                        try:
                            source_coords_previous = get_coordinates(previous_graph, uuid_lookup_previous, node_uuid)
                            target_coords_previous = get_coordinates(previous_graph, uuid_lookup_previous, target)

                            source_coords_current = get_coordinates(current_graph, uuid_lookup_current, node_uuid)
                            target_coords_current = get_coordinates(current_graph, uuid_lookup_current, target)
                            
                            # === filter for nodes that actually need to be updated ====
                            coord_change_flag = False
                            for i, tcp in enumerate(target_coords_previous):
                                if tcp != target_coords_current[i]:
                                    coord_change_flag = True
                            if node_uuid not in interpolated_node_source_instanceUUID and not coord_change_flag:
                                continue
                            #=========================================================== 
                            
                            state['edges'].append({
                                'xs': [source_coords_previous[0], target_coords_previous[0]],
                                'ys': [source_coords_previous[1], target_coords_previous[1]],
                                'color': status_colors['unchanged'],
                                'alpha': 1
                            })
                        except:
                            pass

                    
            
        # PHASE 2: (10-19)
        elif 10 <= step < 20:
            morph_factor = (step - 10) / 9.3

            for uuid in previous_graph['instanceUUID']:
                if uuid not in interpolated_node_source_instanceUUID:
                    continue
                
                index = previous_graph['instanceUUID'].index(uuid)
                node = {key: val[index] for key, val in previous_graph.items()}

                try:
                    index_current = current_graph['instanceUUID'].index(uuid)
                    corresponding_node = {key: val[index_current] for key, val in current_graph.items()}
                except ValueError:
                    corresponding_node = None

                if corresponding_node:
                    node['x'] += morph_factor * (corresponding_node['x'] - node['x'])
                    node['y'] += morph_factor * (corresponding_node['y'] - node['y'])
                    
                    # Adjust bounds_x and bounds_y morphing
                    node['bounds_x'] = [x1 + morph_factor * (x2 - x1) for x1, x2 in zip(node['bounds_x'], corresponding_node['bounds_x'])]
                    node['bounds_y'] = [y1 + morph_factor * (y2 - y1) for y1, y2 in zip(node['bounds_y'], corresponding_node['bounds_y'])]
                    
                    node['color'] = status_colors[node["node_status"]]
                    node['alpha'] = 1
                    state['nodes'].append(node)

            # Handle edges that have changed positions - Interpolate between previous and current =======================
            lerp_factor = (step - 10) / 10.0  # Ranges from 0 to 1 as step goes from 10 to 20

            for idx, current_targets in enumerate(current_graph['targets']):
                node_uuid = current_graph['instanceUUID'][idx]
 
                if node_uuid in uuid_lookup_previous:
                    previous_idx = uuid_lookup_previous[node_uuid]
                    previous_node_targets = previous_graph['targets'][previous_idx]

                    for target in current_targets:
                        e_status = edge_status(previous_node_targets, current_targets, target)
                
                        try:
                            #if target in previous_node_targets:
                            source_coords_previous = get_coordinates(previous_graph, uuid_lookup_previous, node_uuid)
                            target_coords_previous = get_coordinates(previous_graph, uuid_lookup_previous, target)
                            
                            source_coords_current = get_coordinates(current_graph, uuid_lookup_current, node_uuid)
                            target_coords_current = get_coordinates(current_graph, uuid_lookup_current, target)
                            # Lerp the coordinates
                            interpolated_xs = [
                                source_coords_previous[0] + lerp_factor * (source_coords_current[0] - source_coords_previous[0]),
                                target_coords_previous[0] + lerp_factor * (target_coords_current[0] - target_coords_previous[0])
                            ]
                            interpolated_ys = [
                                source_coords_previous[1] + lerp_factor * (source_coords_current[1] - source_coords_previous[1]),
                                target_coords_previous[1] + lerp_factor * (target_coords_current[1] - target_coords_previous[1])
                            ]

                            # === filter for nodes that actually need to be updated ====
                            coord_change_flag = False
                            for i, tcp in enumerate(target_coords_previous):
                                if tcp != target_coords_current[i]:
                                    coord_change_flag = True
                            if node_uuid not in interpolated_node_source_instanceUUID and not coord_change_flag:
                                continue
                            #=========================================================== 
                            state['edges'].append({
                                'xs': interpolated_xs,
                                'ys': interpolated_ys,
                                'color': status_colors[e_status],
                                'alpha': 1
                            })
                        except:
                            pass

            # handle new edges, fade in new edges ------------------------------------------------------------
            for idx, node_targets in enumerate(current_graph['targets']):
                node_uuid = current_graph['instanceUUID'][idx]

                # Get the corresponding targets from the previous graph (if the node exists)
                previous_node_targets = []
                if node_uuid in uuid_lookup_previous:
                    previous_idx = uuid_lookup_previous[node_uuid]
                    previous_node_targets = previous_graph['targets'][previous_idx]

                for target in node_targets:
                    # === filter for nodes that actually need to be updated ====
                    coord_change_flag = False
                    for i, tcp in enumerate(target_coords_previous):
                        if tcp != target_coords_current[i]:
                            coord_change_flag = True
                    if node_uuid not in interpolated_node_source_instanceUUID and not coord_change_flag:
                        continue
                    #=========================================================== 
                    if target not in previous_node_targets:
                        
                        source_coords_current = get_coordinates(current_graph, uuid_lookup_current, node_uuid)
                        target_coords_current = get_coordinates(current_graph, uuid_lookup_current, target)
                        #if target in previous_node_targets:
                        try:
                            source_coords_previous = get_coordinates(previous_graph, uuid_lookup_previous, node_uuid)
                            target_coords_previous = get_coordinates(previous_graph, uuid_lookup_previous, target)
                        except:
                            source_coords_previous = source_coords_current
                            target_coords_previous = target_coords_current
                        
                        # Lerp the coordinates
                        interpolated_xs = [
                            source_coords_previous[0] + lerp_factor * (source_coords_current[0] - source_coords_previous[0]),
                            target_coords_previous[0] + lerp_factor * (target_coords_current[0] - target_coords_previous[0])
                        ]
                        interpolated_ys = [
                            source_coords_previous[1] + lerp_factor * (source_coords_current[1] - source_coords_previous[1]),
                            target_coords_previous[1] + lerp_factor * (target_coords_current[1] - target_coords_previous[1])
                        ]
                        
                        state['edges'].append({
                            'xs': interpolated_xs,
                            'ys': interpolated_ys,
                            'color': status_colors["added"],
                            'alpha': 1
                        })
            
            # Handle removed edges (color fade from gray to red)------------------------------------------------------------------------
            for idx, previous_targets in enumerate(previous_graph['targets']):
                node_uuid = previous_graph['instanceUUID'][idx]
           
                current_node_targets = []
                if node_uuid in uuid_lookup_current:
                    current_idx = uuid_lookup_current[node_uuid]
                    current_node_targets = current_graph['targets'][current_idx]

       
                #=========================================================== 
                for target in previous_targets:
                    if target not in current_node_targets:
                        try:
                            #if target in previous_node_targets:
                            source_coords_previous = get_coordinates(previous_graph, uuid_lookup_previous, node_uuid)
                            target_coords_previous = get_coordinates(previous_graph, uuid_lookup_previous, target)
                            
                            try:
                                source_coords_current = get_coordinates(current_graph, uuid_lookup_current, node_uuid)
                                target_coords_current = get_coordinates(current_graph, uuid_lookup_current, target)
                                #target_coords_current = target_coords_previous
                            except:
                                source_coords_current = source_coords_previous
                                target_coords_current = target_coords_previous
                            
                          
                            # Lerp the coordinates
                            interpolated_xs = [
                                source_coords_previous[0] + lerp_factor * (source_coords_current[0] - source_coords_previous[0]),
                                target_coords_previous[0] + lerp_factor * (target_coords_current[0] - target_coords_previous[0])
                            ]
                            interpolated_ys = [
                                source_coords_previous[1] + lerp_factor * (source_coords_current[1] - source_coords_previous[1]),
                                target_coords_previous[1] + lerp_factor * (target_coords_current[1] - target_coords_previous[1])
                            ]
                            state['edges'].append({
                                'xs': interpolated_xs,
                                'ys': interpolated_ys,
                                'color': status_colors["removed"],
                                'alpha': 1
                            })
                        except:
                            pass
                

        # PHASE 3: (20-29)
        elif 20 <= step < 31:
            fade_factor = (step - 20) / 10.0  # Factor used for fading colors

            for uuid in previous_graph['instanceUUID']:
                if uuid not in interpolated_node_source_instanceUUID:
                    continue
                
                index = previous_graph['instanceUUID'].index(uuid)
                node = {key: val[index] for key, val in previous_graph.items()}

                # Determine the start color
                start_color_rgb = status_colors_rgb[node['node_status']]
                
                # Interpolate between the initial color and gray (unchanged)
                interpolated_color = interpolate_color(start_color_rgb, status_colors_rgb['unchanged'], fade_factor)
                
                # Convert RGB to hex format for Bokeh
                node['color'] = '#%02x%02x%02x' % interpolated_color

                if node['node_status'] == 'removed':
                    node['alpha'] = 1 - fade_factor  # Fade out
                else:
                    node['alpha'] = 1

                state['nodes'].append(node)

            for uuid in current_graph['instanceUUID']:
                if uuid not in interpolated_node_source_instanceUUID:
                    continue
                
                index = current_graph['instanceUUID'].index(uuid)
                node = {key: val[index] for key, val in current_graph.items()}

                # Determine the start color
                start_color_rgb = status_colors_rgb[node['node_status']]
                
                # Interpolate between the initial color and gray (unchanged)
                interpolated_color = interpolate_color(start_color_rgb, status_colors_rgb['unchanged'], fade_factor)
                
                # Convert RGB to hex format for Bokeh
                node['color'] = '#%02x%02x%02x' % interpolated_color

                #if node['node_status'] in ['added', 'changed', 'position_changed']:
                #    node['alpha'] = 1

                if node['node_status'] == 'removed':
                    node['alpha'] = 1 - fade_factor  # Fade out
                else:
                    node['alpha'] = 1

                state['nodes'].append(node)


            # Handling edges in Phase 3 =========================================
            # Use unchanged_edge_source for the base
        
            # handle new edges, fade in new edges 
            for idx, node_targets in enumerate(current_graph['targets']):
                node_uuid = current_graph['instanceUUID'][idx]

                # Get the corresponding targets from the previous graph (if the node exists)
                previous_node_targets = []
                if node_uuid in uuid_lookup_previous:
                    previous_idx = uuid_lookup_previous[node_uuid]
                    previous_node_targets = previous_graph['targets'][previous_idx]

                for target in node_targets:
                    if target not in previous_node_targets:
                        try:
                            source_coords = get_coordinates(previous_graph, uuid_lookup_previous, node_uuid)
                            target_coords = get_coordinates(previous_graph, uuid_lookup_previous, target)
                        except:
                            source_coords = get_coordinates(current_graph, uuid_lookup_current, node_uuid)
                            target_coords = get_coordinates(current_graph, uuid_lookup_current, target)
                        
                        interpolated_color = interpolate_color(start_color_rgb, status_colors_rgb['unchanged'], fade_factor)
                
                        

                        state['edges'].append({
                            'xs': [source_coords[0], target_coords[0]],
                            'ys': [source_coords[1], target_coords[1]],
                            'color': '#%02x%02x%02x' % interpolated_color,
                            'alpha': 1 # Fade in
                        })

            
            # Handle removed edges (color fade from gray to red)
            for idx, previous_targets in enumerate(previous_graph['targets']):
                node_uuid = previous_graph['instanceUUID'][idx]
                
                current_node_targets = []
                if node_uuid in uuid_lookup_current:
                    current_idx = uuid_lookup_current[node_uuid]
                    current_node_targets = current_graph['targets'][current_idx]
                for target in previous_targets:
                    if target not in current_node_targets:
                    
                        source_coords = get_coordinates(previous_graph, uuid_lookup_previous, node_uuid)
                        try:
                            source_coords = get_coordinates(current_graph, uuid_lookup_current, node_uuid)
                            target_coords = get_coordinates(current_graph, uuid_lookup_current, target)
                        except:
                            target_coords = get_coordinates(previous_graph, uuid_lookup_previous, target)
                            source_coords = get_coordinates(previous_graph, uuid_lookup_previous, node_uuid)
                        interpolated_edge_color = interpolate_color(status_colors_rgb['unchanged'], status_colors_rgb['removed'], alpha_factor)
                        
                        state['edges'].append({
                            'xs': [source_coords[0], target_coords[0]],
                            'ys': [source_coords[1], target_coords[1]],
                            'color': 'red',
                            'alpha': 1- fade_factor
                        })


            # Handle edges that have changed positions in the previous_graph state
            for idx, current_targets in enumerate(current_graph['targets']):
                node_uuid = current_graph['instanceUUID'][idx]
                if node_uuid in uuid_lookup_previous:
                    previous_idx = uuid_lookup_previous[node_uuid]
                    previous_node_targets = previous_graph['targets'][previous_idx]

                    for target in current_targets:
                        if target in previous_node_targets:
                        
                            source_coords_previous = get_coordinates(previous_graph, uuid_lookup_previous, node_uuid)
                            target_coords_previous = get_coordinates(previous_graph, uuid_lookup_previous, target)

                            source_coords_current = get_coordinates(current_graph, uuid_lookup_current, node_uuid)
                            target_coords_current = get_coordinates(current_graph, uuid_lookup_current, target)
                            
                            # === filter for nodes that actually need to be updated ====
                            coord_change_flag = False
                            for i, tcp in enumerate(target_coords_previous):
                                if tcp != target_coords_current[i]:
                                    coord_change_flag = True
                            if node_uuid not in interpolated_node_source_instanceUUID and not coord_change_flag:
                                continue
                            #=========================================================== 
                            
                            state['edges'].append({
                                'xs': [source_coords_current[0], target_coords_current[0]],
                                'ys': [source_coords_current[1], target_coords_current[1]],
                                'color': status_colors['unchanged'],
                                'alpha': 1
                            })

        
        transition_states.append(state)
    
    return transition_states






interpolated_node_source, unchanged_node_source, interpolated_edge_source, unchanged_edge_source = morphViewCDS(node_source_current.data, node_source_previous.data)
global transition_states
transition_states = compute_transition_states(node_source_previous.data, node_source_current.data, interpolated_node_source.data["instanceUUID"])



# FIGURE
p_morphView = figure(plot_width=1300, plot_height=900, 
               title="", tools="pan,box_zoom,wheel_zoom,reset, tap, box_select, hover",
                sizing_mode="stretch_both", name ="morphView" )
p_morphView.toolbar.logo = None

p_morphView.xaxis.major_label_text_font_size = '0pt'
p_morphView.yaxis.major_label_text_font_size = '0pt'
p_morphView.xaxis.minor_tick_line_color = None
p_morphView.yaxis.minor_tick_line_color = None
p_morphView.xaxis.major_tick_line_color = None
p_morphView.yaxis.major_tick_line_color = None

# Removing the grid
p_morphView.xgrid.visible = False
p_morphView.ygrid.visible = False

p_morphView.xaxis.axis_line_color = "gray"
p_morphView.xaxis.axis_line_alpha = 0.0

p_morphView.yaxis.axis_line_color = "gray"
p_morphView.yaxis.axis_line_alpha = 0.0

p_morphView.outline_line_color = "lightgrey"

# Capture the patches in a variable
node_glyph_static = p_morphView.patches('bounds_x', 'bounds_y', source=unchanged_node_source, fill_color="grey", 
                        hover_line_color = "black", hover_line_width =3, hover_line_alpha=0.8,
                        alpha=0.6, line_color="white", selection_line_color="orange", 
                        selection_line_alpha=0.8, selection_line_width=3)

node_glyph_dynamic = p_morphView.patches('bounds_x', 'bounds_y', source=interpolated_node_source, fill_color="color", 
                        hover_line_color = "black", hover_line_width =3, hover_line_alpha=0.8,
                        fill_alpha="alpha", line_color="grey", selection_line_color="orange", 
                        selection_line_alpha=0.8, selection_line_width=3)


# Draw edges using multi_line glyph
p_morphView.multi_line('xs', 'ys', source=unchanged_edge_source, line_width=0.1, color="black", alpha=0.6)
p_morphView.multi_line('xs', 'ys', source=interpolated_edge_source, line_width=0.5, color="color", alpha="alpha")


# Create a slider to control the transition
morph_slider = Slider(start=0, end=30, value=0, step=1, title="Transition Step", sizing_mode="fixed", width=200, align="center", name="morph_slider_layout")
morph_slider_layout = morph_slider # column(morph_slider, name="morph_slider_layout")  # Center the slider horizontally




def morph_callback(attr, old, new):
    
    step = morph_slider.value

    # Extract node and edge data
    node_data = copy.deepcopy(interpolated_node_source.data)
    edge_data = copy.deepcopy(interpolated_edge_source.data)
    global transition_states
    transition_data = transition_states[step]  # Get the transition state for the current step


    

    # Create a color mapper for node statuses
    status_colors = {
        'added': 'green',
        'removed': 'red',
        'changed': 'blue',
        'unchanged': 'gray',
        'position_changed': 'purple' 
    }


    # Node handling logic
    for idx, uuid in enumerate(interpolated_node_source.data["instanceUUID"]):
        node = next((n for n in transition_data['nodes'] if n['instanceUUID'] == uuid), None)
        if not node:
            continue
        
        if 0 <= step < 10:
            # Phase 1: Introduction of changes
            node_data['alpha'][idx] = node['alpha']
            node_data['color'][idx] = node["color"]

        elif 10 <= step < 20:
            # Phase 2: Morph positions and bounds
            node_data['x'][idx] = node['x']
            node_data['y'][idx] = node['y']
            node_data['bounds_x'][idx] = node['bounds_x']
            node_data['bounds_y'][idx] = node['bounds_y']
            # Keep the end color of Phase 1
            node_data['color'][idx] = node["color"]

        else:  # 20 <= step <= 29
            # Phase 3: Reversion to original states
            node_data['color'][idx] = node["color"]
            node_data['alpha'][idx] = node["alpha"]

    # Edge handling logic

     # Edge handling logic
    edge_data['xs'] = []
    edge_data['ys'] = []
    edge_data['color'] = []
    edge_data['alpha'] = []

    for edge in transition_data['edges']:
        edge_data['xs'].append(edge['xs'])
        edge_data['ys'].append(edge['ys'])
        edge_data['color'].append(edge["color"])
        edge_data['alpha'].append(edge['alpha'])
    
    interpolated_node_source.data = dict(node_data)
    interpolated_edge_source.data = dict(edge_data)




morph_slider.on_change('value', morph_callback)
morphViewLayout = column(p_morphView, name="morphViewLayout", sizing_mode="stretch_both")
curdoc().add_root(morphViewLayout)
curdoc().add_root(morph_slider_layout)


#------ Extract Stats -----------------
def extractStats(dataDict, previousDataDict=None):
    stats = {}
    
    compCount = len([x for x in dataDict["kind"] if x == "component"])
    startPts = len([x for x in dataDict["startPoint"] if x == True])
    endPts = len([x for x in dataDict["endPoint"] if x == True])
    
    try:
        compTime = str(round(sum(dataDict["computiationTime"]) / 1000,1)) + "s"
    except:
        compTime = "0s"
    
    stats.update({"compTime": compTime, "compCount": compCount, "startPts": startPts, "endPts": endPts})
    
    # If previousDataDict is provided, we compute differences.
    if previousDataDict:
        added = len([x for x in dataDict["node_status"] if x == "added"])
        removed = len([x for x in dataDict["node_status"] if x == "removed"])
        changed = len([x for x in dataDict["node_status"] if x == "changed"])
        
        # Count changed connections based on the flags for source and target attributes.
        #changed_conn = sum(1 for src_changed, tgt_changed in zip(dataDict["source_changed"], dataDict["target_changed"]) if src_changed or tgt_changed)
        
        stats.update({"added": added, "removed": removed, "changed": changed})
    else:
        stats.update({"added": 0, "removed": 0, "changed": 0})
    
    return stats



def create_bar_chart_data(stats_current, total_nodes):
    categories = ["added", "removed", "changed"]
    percentages = [(stats_current[cat] / total_nodes) * 100 for cat in categories]

    source = ColumnDataSource(data=dict(categories=categories, percentages=percentages))
    return source


def create_bar_chart(source):
    categories = source.data['categories']
    p = figure(x_range=categories, width=200, height=180, title="", toolbar_location=None, tools="")
    colors = ['green', 'red', 'blue']

    # Create a color map that maps categories to their respective colors
    color_map = factor_cmap('categories', palette=colors, factors=categories)

    p.vbar(x='categories', top='percentages', width=0.7, source=source, color=color_map)
    
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.toolbar.logo = None
    p.toolbar_location = None
    
    # Removing minor ticks
    p.xaxis.minor_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    
    # Removing the grid
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.major_label_text_font_size = '8pt'
    p.yaxis.major_label_text_font_size = '5pt'

    hover = HoverTool()
    hover.tooltips = [("Percentage", "@percentages{0.2f}%")]
    p.add_tools(hover)

    p.toolbar.logo = None
    p.toolbar_location = None

    # Activating the hover tool by default
    p.toolbar.active_inspect = [hover]
    
    return p




# Now to create the div:
#change_stats_div = create_stats_div(stats_current)



def create_rank_value_data(node_source_current, node_source_previous):
    # Extract data from the provided node sources
    current_data = {
        'rank': node_source_current.data['rank'],
        'computiationTime': node_source_current.data['computiationTime'],
        'type': ['Current'] * len(node_source_current.data['rank'])
    }

    previous_data = {
        'rank': node_source_previous.data['rank'],
        'computiationTime': node_source_previous.data['computiationTime'],
        'type': ['Previous'] * len(node_source_previous.data['rank'])
    }

    # Merge current and previous data
    rank_data = {
        'rank': current_data['rank'] + previous_data['rank'],
        'computiationTime': current_data['computiationTime'] + previous_data['computiationTime'],
        'type': current_data['type'] + previous_data['type']
    }

    # Create a new ColumnDataSource with combined data
    rank_source = ColumnDataSource(data=rank_data)
    return rank_source



def create_rank_value_plot(rank_source):
    plot = figure(width=200, height=200, title="", tools="box_select", x_axis_label="Rank", y_axis_label="Comp. Time")

    # Scatter plot for the current and previous data with dynamic legend
    plot.scatter(
        'rank', 'computiationTime',
        source=rank_source,
        color=factor_cmap('type', palette=['blue', 'red'], factors=['Current', 'Previous']),
        size=3, alpha=0.5,
        legend_field='type'  # Correctly set legend field
    )

    # Set up legend and styling
    plot.legend.label_text_font_size = '5pt'
    plot.legend.orientation = "vertical"
    plot.legend.location = "top_right"
    plot.legend.border_line_width = 1
    plot.legend.border_line_alpha = 0
    plot.legend.background_fill_alpha = 0.3
    plot.legend.padding = 0

    # Styling
    plot.xaxis.major_label_text_font_size = '5pt'
    plot.yaxis.major_label_text_font_size = '5pt'
    plot.xaxis.major_label_text_color = "black"
    plot.yaxis.major_label_text_color = "black"
    plot.yaxis.major_label_text_alpha = 0.4
    plot.xaxis.major_label_text_alpha = 0.4
    plot.xaxis.axis_label_text_color = 'lightgrey'
    plot.yaxis.axis_label_text_color = 'lightgrey'
    plot.xaxis.axis_line_color = "gray"
    plot.xaxis.axis_line_alpha = 0.2
    plot.yaxis.axis_line_color = "gray"
    plot.yaxis.axis_line_alpha = 0.2
    plot.outline_line_color = "white"
    plot.toolbar.logo = None
    plot.toolbar_location = None

    # Removing minor ticks
    plot.xaxis.minor_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None

    # Removing the grid
    plot.xgrid.visible = False
    plot.ygrid.visible = False

    # Adjust x-axis ticks to show only min and max values
    max_rank = max(rank_source.data['rank'])
    plot.xaxis.ticker = [0, max_rank]

    # Move axis labels closer to the plot
    plot.xaxis.axis_label_standoff = 20
    plot.yaxis.axis_label_standoff = 16
    plot.xaxis.axis_label_text_font_size = '6pt'
    plot.yaxis.axis_label_text_font_size = '6pt'

    # Shorten the tick lines
    plot.xaxis.major_tick_out = 0
    plot.xaxis.major_tick_in = 2
    plot.yaxis.major_tick_out = 0
    plot.yaxis.major_tick_in = 2

    # Move tick labels inside the plot
    plot.xaxis.major_label_standoff = -24
    plot.yaxis.major_label_standoff = -24

    return plot





stats_current = extractStats(node_source_current.data, node_source_previous.data)
stats_previous = extractStats(node_source_previous.data) # This doesn't compare against anything, so only the original stats.
total_nodes = len(node_source_current.data["kind"])


bar_chart_source = create_bar_chart_data(stats_current, total_nodes)
p_bar_chart = create_bar_chart(bar_chart_source)

rank_value_source = create_rank_value_data(node_source_current, node_source_previous)
rank_value_plot= create_rank_value_plot(rank_value_source)

# Slider to set the threshold for computation time
max_rank = max(len(node_source_current.data['computiationTime']), len(node_source_previous.data['computiationTime']))



rank_slider = Slider(start=1, end=max_rank, value=max_rank, step=1, title="zoom", 
                     sizing_mode="stretch_width")
rank_callback = CustomJS(args=dict(x_range=rank_value_plot.x_range, slider=rank_slider), code="""
    x_range.start = 0;
    x_range.end = slider.value;
""")
rank_slider.js_on_change('value', rank_callback)

curdoc().add_root(column(rank_value_plot, rank_slider, width=200, name="rank_value_plot"))
p_bar_chart.name = "bar_chart"
curdoc().add_root(p_bar_chart)




# Create Divs for all attributes
attribute_names = {
    "compTime": "Computation Time",
    "compCount": "Component Count",
    "startPts": "Start Points",
    "endPts": "End Points"
}




# Function to create and add Divs for each attribute comparison
def create_comparison_div(attr_name, current_value, previous_value):
    """
    Creates Divs for each attribute comparison and adds them to curdoc.
    """
    # Ensure the attribute name is sanitized for use in names
    attr_name_sanitized = attr_name.replace(" ", "_")
    
    # Create Divs for current and previous values
    current_value_div = Div(
        text=f"{current_value}",
        name=f"{attr_name_sanitized}_current_value",
        width_policy="min",
        width=20
    )
    previous_value_div = Div(
        text=f"{previous_value}",
        name=f"{attr_name_sanitized}_previous_value",
        width_policy="min",
        width=20
    )
    
    # Add them to the document for reference in the HTML template
    curdoc().add_root(current_value_div)
    curdoc().add_root(previous_value_div)

    # Return Div objects for further manipulation if needed
    return current_value_div, previous_value_div

# Initialize comparison Divs based on current and previous stats
def initialize_comparison_divs(stats_current, stats_previous):
    """
    Initializes all comparison Divs for current and previous values of each attribute.
    """
    divs = {}  # Dictionary to store Div objects for later updates
    for key, attr_name in attribute_names.items():
        cur_div, prev_div = create_comparison_div(attr_name, stats_current[key], stats_previous[key])
        divs[f"{attr_name}_current"] = cur_div
        divs[f"{attr_name}_previous"] = prev_div
    return divs

# Assume these stats are calculated elsewhere in your code
# stats_current = extractStats(node_source_current.data, node_source_previous.data)
# stats_previous = extractStats(node_source_previous.data)



# Function to create the stats Div text content
def create_stats_div_text(stats_current):
    total_nodes = stats_current['compCount']
    
    # Calculate percentages
    changed_percent = round((stats_current['changed'] / total_nodes) * 100, 1)
    added_percent = round((stats_current['added'] / total_nodes) * 100, 1)
    removed_percent = round((stats_current['removed'] / total_nodes) * 100, 1)
    
    # Convert to text
    changed_text = f"Changed Nodes: {changed_percent:.2f}%"
    added_text = f"Nodes Added: {added_percent}%"
    removed_text = f"Nodes Removed: {removed_percent}%"
    
    # Combine all stats into one HTML content
    html_content = f"""
    <div style="display: flex; flex-direction: column; align-items: center; width: 200px; padding: 8px; border: 1px solid #ccc; border-radius: 8px;">
        <div style="font-size: 15px; margin-bottom: 2px; text-align: center;">{changed_text}</div>
        <div style="font-size: 15px; margin-bottom: 2px; text-align: center;">{added_text}</div>
        <div style="font-size: 15px; margin-bottom: 2px; text-align: center;">{removed_text}</div>
    </div>
    """
    return html_content


def update_comparison_div(attr_name, current_value, previous_value):
    """
    Updates the text of existing Div elements for a given attribute.
    """
    attr_name_sanitized = attr_name.replace(" ", "_")
    
    # Try to get the Div by its name and update it
    current_div = curdoc().get_model_by_name(f"{attr_name_sanitized}_current_value")
    previous_div = curdoc().get_model_by_name(f"{attr_name_sanitized}_previous_value")
    
    if current_div:
        current_div.text = f"{current_value}"
    else:
        print(f"Warning: {attr_name_sanitized}_current_value Div not found.")

    if previous_div:
        previous_div.text = f"{previous_value}"
    else:
        print(f"Warning: {attr_name_sanitized}_previous_value Div not found.")


# Initialize Div elements and store them in a dictionary

comparison_divs = initialize_comparison_divs(stats_current, stats_previous)

# Create and add the change_stats_div to the document
change_stats_div = Div(text=create_stats_div_text(stats_current), width=200, height=130, sizing_mode="stretch_width" )
#curdoc().add_root(change_stats_div)




# highlight and sync edges  =====================================================
#   =============================================================================
#   =============================================================================


def updateSpeckleStream(stream_id,
                        branch_name,
                        client,
                        data_object,
                        commit_message="Updated the data object",
                        ):
    """
    Updates a speckle stream with a new data object.

    Args:
        stream_id (str): The ID of the speckle stream.
        branch_name (str): The name of the branch within the speckle stream.
        client (specklepy.api.client.Client): A speckle client.
        data_object (dict): The data object to send to the speckle stream.
        commit_message (str): The commit message. Defaults to "Updated the data object".
    """
    # set stream and branch
    branch = client.branch.get(stream_id, branch_name)
    # Get transport
    transport = ServerTransport(client=client, stream_id=stream_id)
    # Send the data object to the speckle stream
    object_id = operations.send(data_object, [transport])

    # Create a new commit with the new object
    commit_id = client.commit.create(
        stream_id,
        object_id= object_id,
        message=commit_message,
        branch_name=branch_name,
    )

    return commit_id


LABELS = ["sync previous", "sync current"]
checkbox_button_group = CheckboxButtonGroup(labels=LABELS, active=[])

# Create the mappings
id_mapping_cur = {str(i): component_id for i, component_id in enumerate(node_source_current.data['instanceUUID'])}
uuid_mapping_cur = {v: k for k, v in id_mapping_cur.items()}

id_mapping_prev = {str(i): component_id for i, component_id in enumerate(node_source_previous.data['instanceUUID'])}
uuid_mapping_prev = {v: k for k, v in id_mapping_prev.items()}


def getStream():
    """Fetch data from Speckle stream using user-provided inputs."""
    # Read the values from the widgets
    speckle_token = speckle_token_input.value.strip()
    uuid_sync_stream_id = uuid_sync_stream_id_input.value.strip()
    uuid_sync_branch_name = uuid_sync_branch_name_input.value.strip()

    # Check if all required inputs are provided
    if not speckle_token or not uuid_sync_stream_id or not uuid_sync_branch_name:
        print("Speckle token, stream ID, and branch name are required to fetch data.")
        return None, None

    # Authenticate client with provided token
    client = SpeckleClient(host="https://speckle.xyz")
    client.authenticate(token=speckle_token)

    # Fetch data from the specified stream and branch
    res = getSpeckleStream(uuid_sync_stream_id, uuid_sync_branch_name, client)
    
    # Extract data
    uuid_data = res["@uuid_data"]["@{0}"][0]
    res_copy = copy.deepcopy(res)

    return uuid_data, res_copy



def updateStream(res_new, data_str):
    """Update data on Speckle stream using user-provided inputs."""
    # Read the values from the widgets
    speckle_token = speckle_token_input.value.strip()
    uuid_sync_stream_id = uuid_sync_stream_id_input.value.strip()
    uuid_sync_branch_name = uuid_sync_branch_name_input.value.strip()

    # Check if all required inputs are provided
    if not speckle_token or not uuid_sync_stream_id or not uuid_sync_branch_name:
        print("Speckle token, stream ID, and branch name are required to update data.")
        return

    # Authenticate client with provided token
    client = SpeckleClient(host="https://speckle.xyz")
    client.authenticate(token=speckle_token)

    # Update the stream data
    res_new["@uuid_data"]["@{0}"][0] = data_str
    commit_id = updateSpeckleStream(uuid_sync_stream_id, uuid_sync_branch_name, client, res_new)
    print("Stream updated, commit id", commit_id)



def process_data_to_indices(data, uuid_mapping):
    """Convert a list of UUIDs to their corresponding indices in the graph."""
    try:
        data = data.replace("'", '"')
        uuid_list = json.loads(data)
        if isinstance(uuid_list, list):
            return [int(uuid_mapping[uuid]) for uuid in uuid_list if uuid in uuid_mapping and uuid != "no_id"]
    except json.JSONDecodeError:
        pass
    return []

update_from_button = False


def on_button_click():
    speckle_ids, res_new = getStream()
    
    # For current graph
    indices_cur = process_data_to_indices(speckle_ids, uuid_mapping_cur)
    
    # For previous graph
    indices_prev = process_data_to_indices(speckle_ids, uuid_mapping_prev)

    node_source_current.selected.indices = indices_cur
    node_source_previous.selected.indices = indices_prev




def on_selection_change_current(attr, old, new):
    selected_uuids = [node_source_current.data['instanceGuid'][i] for i in new]
    speckle_id, res_new = getStream()
    updateStream(res_new, str(selected_uuids))

def on_selection_change_previous(attr, old, new):
    selected_uuids = [node_source_previous.data['instanceGuid'][i] for i in new]
    speckle_id, res_new = getStream()
    updateStream(res_new, str(selected_uuids))

# Define the function to push selected UUIDs
def push_selected():
    # Gather selected UUIDs from current source
    selected_uuids_cur = [node_source_current.data['instanceUUID'][i] for i in node_source_current.selected.indices]

    # Gather selected UUIDs from previous source
    selected_uuids_prev = [node_source_previous.data['instanceUUID'][i] for i in node_source_previous.selected.indices]

    # Combine both lists
    combined_selected_uuids = list(set(selected_uuids_cur + selected_uuids_prev))

    # Fetch existing data (if you need to compare or merge)
    _, res_new = getStream()

    # Update the stream
    updateStream(res_new, str(combined_selected_uuids))

# Create the "Push Selected" button and attach the function
button_push_selected = Button(label="Push Selected")
button_push_selected.on_click(push_selected)



button_fetch = Button(label="Fetch Selected")
button_fetch.on_click(on_button_click)

fetch_push_selected_layout = column(button_fetch, button_push_selected, sizing_mode="stretch_width", name="fetch_push_selected_layout")
curdoc().add_root(fetch_push_selected_layout)
#   =============================================================================
#   =============================================================================
#   =============================================================================

slider_upstream = Slider(start=0, end=60, value=0, step=1, title="Upstream Depth", sizing_mode="stretch_width")
slider_downstream = Slider(start=0, end=60, value=0, step=1, title="Downstream Depth", sizing_mode="stretch_width")
select_button = Button(label="Add to Selection", sizing_mode="stretch_width")
select_button_flip = Button(label="Select Connected",sizing_mode="stretch_width")

graph_selection_widgest = column(slider_upstream, slider_downstream, select_button, select_button_flip, sizing_mode="stretch_width")

global global_connected_nodes
global_connected_nodes = {"current": [], "previous": []}

def update_highlighted_edges(source_from, source_to, G, G_reversed, depth_upstream, depth_downstream, highlighted_edges_source, mode="current"):

    selected_uuids = [source_from.data['instanceGuid'][i] for i in source_from.selected.indices]
    
    upstream_nodes, downstream_nodes = set(), set()

    local_connected_nodes = {"current": [], "previous": []}
    for uuid in selected_uuids:
        # Find nodes downstream (successors)
        length_downstream = nx.single_source_shortest_path_length(G, uuid, cutoff=depth_downstream)
        downstream_nodes_current = [node for node, depth in length_downstream.items() if depth <= depth_downstream and depth > 0]
        downstream_nodes.update(downstream_nodes_current)

        local_connected_nodes[mode].extend(downstream_nodes_current)
        
        

        # Find nodes upstream (predecessors) using the reversed graph
        length_upstream = nx.single_source_shortest_path_length(G_reversed, uuid, cutoff=depth_upstream)
        upstream_nodes_current = [node for node, depth in length_upstream.items() if depth <= depth_upstream]
        upstream_nodes.update(upstream_nodes_current)

        local_connected_nodes[mode].extend(upstream_nodes_current)

    #update global
    global global_connected_nodes
    global_connected_nodes[mode] = local_connected_nodes[mode]

    # Nodes between the selected node and the upstream/downstream nodes
    between_nodes = upstream_nodes.intersection(downstream_nodes)

    xs, ys, colors = [], [], []

    # Handle downstream edges
    for uuid in downstream_nodes:
        for neighbor in G.successors(uuid):
            if neighbor in downstream_nodes:
                xs.append([G.nodes[uuid]['pivot'][0], G.nodes[neighbor]['pivot'][0]])
                ys.append([G.nodes[uuid]['pivot'][1], G.nodes[neighbor]['pivot'][1]])
                colors.append("red")

    # Handle upstream edges
    for uuid in upstream_nodes:
        for neighbor in G.successors(uuid):
            if neighbor in upstream_nodes:
                xs.append([G.nodes[uuid]['pivot'][0], G.nodes[neighbor]['pivot'][0]])
                ys.append([G.nodes[uuid]['pivot'][1], G.nodes[neighbor]['pivot'][1]])
                colors.append("blue")

    # Handle between edges
    for uuid in between_nodes:
        for neighbor in G.successors(uuid):
            if neighbor in between_nodes:
                xs.append([G.nodes[uuid]['pivot'][0], G.nodes[neighbor]['pivot'][0]])
                ys.append([G.nodes[uuid]['pivot'][1], G.nodes[neighbor]['pivot'][1]])
                colors.append("green")

    highlighted_edges_source.data = dict(xs=xs, ys=ys, color=colors)

def callback_current_edges(attr, old, new):
    global cur_graphX
    global prev_graphX
    global cur_graphX_reverse
    global prev_graphX_reverse
    selected_uuids = [node_source_current.data['instanceGuid'][i] for i in new]
    target_indices = [i for i, guid in enumerate(node_source_previous.data['instanceGuid']) if guid in selected_uuids]
    node_source_previous.selected.indices = target_indices
    update_highlighted_edges(node_source_current, node_source_previous, cur_graphX, 
                             cur_graphX_reverse, slider_upstream.value, slider_downstream.value, 
                             highlighted_edges_source_current, "current")

def callback_previous_edges(attr, old, new):
    global cur_graphX
    global prev_graphX
    global cur_graphX_reverse
    global prev_graphX_reverse
    selected_uuids = [node_source_previous.data['instanceGuid'][i] for i in new]
    target_indices = [i for i, guid in enumerate(node_source_current.data['instanceGuid']) if guid in selected_uuids]
    node_source_current.selected.indices = target_indices

    update_highlighted_edges(node_source_previous, node_source_current, prev_graphX, 
                             prev_graphX_reverse, slider_upstream.value, slider_downstream.value, 
                             highlighted_edges_source_previous, "previous")

def on_slider_change(attr, old, new):
    global cur_graphX
    global prev_graphX
    global cur_graphX_reverse
    global prev_graphX_reverse
    update_highlighted_edges(node_source_current, node_source_previous, cur_graphX, 
                             cur_graphX_reverse, slider_upstream.value, slider_downstream.value, 
                             highlighted_edges_source_current, "current")
    update_highlighted_edges(node_source_previous, node_source_current,prev_graphX, 
                             prev_graphX_reverse, slider_upstream.value, slider_downstream.value, 
                             highlighted_edges_source_previous, "previous")
    
def select_components(node_source_current=node_source_current, node_source_previous=node_source_previous):
    # Get the connected nodes based on the current selection
    global global_connected_nodes
    connected_nodes_current = copy.deepcopy(set(global_connected_nodes["current"]))
    connected_nodes_previous = copy.deepcopy(set(global_connected_nodes["previous"]))

    # reset before updating the selected components
    global_connected_nodes = {"current": [], "previous": []}


    # Update the selected indices of the data sources
    node_source_current.selected.indices = [i for i, guid in enumerate(node_source_current.data['instanceGuid']) if guid in connected_nodes_current]
    node_source_previous.selected.indices = [i for i, guid in enumerate(node_source_previous.data['instanceGuid']) if guid in connected_nodes_previous]

def select_components_flip(node_source_current=node_source_current, node_source_previous=node_source_previous):
    # Access the global variable
    global global_connected_nodes
    
    # Get the set of nodes that are currently selected
    current_selected_nodes_current = set([node_source_current.data['instanceGuid'][i] for i in node_source_current.selected.indices])
    current_selected_nodes_previous = set([node_source_previous.data['instanceGuid'][i] for i in node_source_previous.selected.indices])

    # Get the connected nodes based on the current selection
    connected_nodes_current = set(global_connected_nodes["current"])
    connected_nodes_previous = set(global_connected_nodes["previous"])

    # Find nodes that are in connected nodes but not currently selected
    switch_nodes_current = connected_nodes_current - current_selected_nodes_current
    switch_nodes_previous = connected_nodes_previous - current_selected_nodes_previous

    # Reset the global variable
    global_connected_nodes = {"current": [], "previous": []}

    # Update the selected indices of the data sources
    node_source_current.selected.indices = [i for i, guid in enumerate(node_source_current.data['instanceGuid']) if guid in switch_nodes_current]
    node_source_previous.selected.indices = [i for i, guid in enumerate(node_source_previous.data['instanceGuid']) if guid in switch_nodes_previous]

slider_upstream.on_change('value', on_slider_change)
slider_downstream.on_change('value', on_slider_change)
node_source_current.selected.on_change('indices', callback_current_edges)
node_source_previous.selected.on_change('indices', callback_previous_edges)

select_button_flip.on_click(select_components_flip)
select_button.on_click(select_components)

# sync cds callbacks ========================
def callback_current(attr, old, new):
    selected_uuids = [node_source_current.data['instanceGuid'][i] for i in new]
    target_indices = [i for i, guid in enumerate(node_source_previous.data['instanceGuid']) if guid in selected_uuids]
    node_source_previous.selected.indices = target_indices

def callback_previous(attr, old, new):
    selected_uuids = [node_source_previous.data['instanceGuid'][i] for i in new]
    target_indices = [i for i, guid in enumerate(node_source_current.data['instanceGuid']) if guid in selected_uuids]
    node_source_current.selected.indices = target_indices

           
node_source_current.selected.on_change('indices', callback_current)
node_source_previous.selected.on_change('indices', callback_previous)


split_view_layout = column(gridplot([[p_current, p_previous]],toolbar_location='below', sizing_mode="stretch_both"), name = "split_view_layout", sizing_mode="stretch_both")

graph_selection_widgest.name = "graph_selection_widgest"



# create layout
curdoc().add_root(split_view_layout)
curdoc().add_root(graph_selection_widgest)

# Add layouts to the current document so they can be referenced in the HTML
curdoc().add_root(column(speckle_layout, name="speckle_layout"))
curdoc().add_root(column(json_layout, name="json_layout"))
curdoc().add_root(column(uuid_sync_layout, name="uuid_sync_layout"))
