from pathlib import Path
import shutil


def find_connected_nodes(node, graph, visited=None):
    if visited is None:
        visited = set()

    # Check if the node is in the graph keys or values
    if node not in visited:
        visited.add(node)
        if node in graph:
            for connected_node in graph[node]:
                find_connected_nodes(connected_node, graph, visited)
        for key, value in graph.items():
            if node in value:
                find_connected_nodes(key, graph, visited)
    return list(visited)


def remove_elements(list1, list2):
    return [item for item in list1 if item not in list2]


def get_groups_alt(my_nodes_dict, verbose = False):

    candidates_nodes = list(my_nodes_dict.keys())
    my_groups = []

    for current_node in my_nodes_dict.keys():

        if verbose:
            print(f'len: {len(candidates_nodes)}')

        if current_node not in candidates_nodes:
            if verbose:
                print(f'\tNode {current_node} already processed')
            continue

        connected_nodes = find_connected_nodes(current_node, my_nodes_dict)

        candidates_nodes = remove_elements(candidates_nodes, connected_nodes)   
        candidates_nodes.insert(0, current_node)

        my_groups.append((current_node, len(connected_nodes)))

    return my_groups


def copy_arrays_to_folder(arrays, indices, folder_path, verbose=False):
    """
    Copy a list of arrays to a folder, renaming them with the corresponding probability.
    """

    # Create the subfolder using pathlib
    folder_path.mkdir(parents=True, exist_ok=True)

    # Loop over the indices and copy each WAV file
    for idx in indices:
        if idx < len(arrays):

            file_path = arrays[idx]
            new_filename = Path(file_path).stem + Path(file_path).suffix

            destination_path = folder_path / new_filename

            shutil.copy(file_path, destination_path)
            if verbose:
                print(f"{folder_path.name}: Copied {new_filename}")
        else:
            print(f"!!!!! \t{folder_path.name}: Index {idx} is out of range.")
