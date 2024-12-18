import numpy as np
import pandas as pd
from GraphTsetlinMachine.graphs import Graphs
from tqdm import tqdm
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import itertools
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx  # for centrality measures
import os
import pandas as pd
import itertools
import numpy as np
import time  # For generating unique timestamps
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_neighbors(q, r, node_dim):
    directions = [
        (1, 0),  # right
        (1, -1),  # top right
        (0, -1),  # top left
        (-1, 0),  # left
        (-1, 1),  # bottom left
        (0, 1)  # bottom right
    ]
    neighbors = []

    # loop neighbours and calc coordinates (q: horisontal axis, r: diagonal axis)
    for dq, dr in directions:
        nq, nr = q + dq, r + dr
        if 0 <= nq < node_dim and 0 <= nr < node_dim:
            neighbors.append(((dq, dr), (nq, nr)))
    return neighbors


def is_edge_cell(q, r, node_dim):
    return q == 0 or r == 0 or q == node_dim - 1 or r == node_dim - 1


def is_corner_cell(q, r, node_dim):
    return (q == 0 and r == 0) or \
        (q == 0 and r == node_dim - 1) or \
        (q == node_dim - 1 and r == 0) or \
        (q == node_dim - 1 and r == node_dim - 1)


def find_winning_path(board, player, node_dim):
    visited = set()
    predecessor = {}
    queue = deque()

    # initialize the queue based on the players starting edge
    if player == 1:  # black from the left
        for q in range(node_dim):
            if board[q, 0] == player:
                queue.append((q, 0))
                visited.add((q, 0))
    elif player == -1:  # white from the top
        for r in range(node_dim):
            if board[0, r] == player:
                queue.append((0, r))
                visited.add((0, r))

    while queue:
        current_q, current_r = queue.popleft()

        # check if the current node has reached the opposite edge
        if (player == 1 and current_r == node_dim - 1) or (player == -1 and current_q == node_dim - 1):
            # reconstruct the winning path
            path = [(current_q, current_r)]
            while (current_q, current_r) in predecessor:
                current_q, current_r = predecessor[(current_q, current_r)]
                path.append((current_q, current_r))
            return True, path

        # explore neighbors
        neighbors = get_neighbors(current_q, current_r, node_dim)
        for _, (neighbor_q, neighbor_r) in neighbors:
            if (neighbor_q, neighbor_r) not in visited and board[neighbor_q, neighbor_r] == player:
                visited.add((neighbor_q, neighbor_r))
                predecessor[(neighbor_q, neighbor_r)] = (current_q, current_r)
                queue.append((neighbor_q, neighbor_r))

    return False, []


'''
methods for using networkx to calc betweeness (often in the shortest win path), 
closeness (closeness to other nodes of same player) 
and centrality of nodes on the board
'''


def calculate_betweenness_centrality(board, player, node_dim):
    G = nx.Graph()
    for q in range(node_dim):
        for r in range(node_dim):
            if board[q, r] != player:
                continue
            node = (q, r)
            G.add_node(node)
            for _, neighbor in get_neighbors(q, r, node_dim):
                neighbor_q, neighbor_r = neighbor
                if board[neighbor_q, neighbor_r] == player:
                    G.add_edge(node, neighbor)
    if len(G) == 0:
        return {}
    centrality = nx.betweenness_centrality(G)
    return centrality


def calculate_closeness_centrality(board, player, node_dim):
    G = nx.Graph()
    for q in range(node_dim):
        for r in range(node_dim):
            if board[q, r] != player:
                continue
            node = (q, r)
            G.add_node(node)
            for _, neighbor in get_neighbors(q, r, node_dim):
                neighbor_q, neighbor_r = neighbor
                if board[neighbor_q, neighbor_r] == player:
                    G.add_edge(node, neighbor)
    if len(G) == 0:
        return {}
    centrality = nx.closeness_centrality(G)
    return centrality


def calculate_eigenvector_centrality(board, player, node_dim):
    G = nx.Graph()
    for q in range(node_dim):
        for r in range(node_dim):
            if board[q, r] != player:
                continue
            node = (q, r)
            G.add_node(node)
            for _, neighbor in get_neighbors(q, r, node_dim):
                neighbor_q, neighbor_r = neighbor
                if board[neighbor_q, neighbor_r] == player:
                    G.add_edge(node, neighbor)
    if len(G) == 0:
        return {}
    try:
        centrality = nx.eigenvector_centrality_numpy(G)
    except nx.NetworkXException:
        centrality = {}
    return centrality


def generate_proximity_features(node_dim):
    features = []
    for player in ['black', 'white']:
        for side in ['prox_left', 'prox_right'] if player == 'black' else ['prox_top', 'prox_bottom']:
            for distance in range(node_dim):  # 0 to node_dim-1
                feature = f'{player}_{side}_{distance}'
                features.append(feature)
    return features


def generate_pattern_features(board, q, r, node_dim):
    # check for patterns like consecutive stones
    patterns = []
    directions = {
        'east': (1, 0),
        'north_east': (1, -1),
        'north_west': (0, -1),
        'west': (-1, 0),
        'south_west': (-1, 1),
        'south_east': (0, 1)
    }

    current_player = board[q, r]
    if current_player == 0:
        return patterns  # no pattern, empty

    for dir_name, (dq, dr) in directions.items():
        consecutive = 1
        for step in range(1, node_dim):
            nq, nr = q + dq * step, r + dr * step
            if 0 <= nq < node_dim and 0 <= nr < node_dim:
                if board[nq, nr] == current_player:
                    consecutive += 1
                else:
                    break
            else:
                break
        if consecutive >= 2:
            patterns.append(f'pattern_{dir_name}_{consecutive}')

    return patterns


'''
Start: strategy patterns from http://www.mseymour.ca/hex_book/ 
'''


def is_critical_node(board, player, q, r, node_dim):
    # checks if a node is critical to block the opponents win

    opponent = -player
    if board[q, r] != 0:
        return False
    temp_board = board.copy()
    temp_board[q, r] = player
    opponent_has_path_before = find_winning_path(board, opponent, node_dim)[0]
    opponent_has_path_after = find_winning_path(temp_board, opponent, node_dim)[0]
    return opponent_has_path_before and not opponent_has_path_after


def detect_ladder(board, q, r, player, node_dim):
    ladder_forms = []
    if board[q, r] != player:
        return ladder_forms

    directions = [(1, -1), (0, 1), (-1, 1)]
    for dq, dr in directions:
        if (0 <= q + dq < node_dim) and (0 <= r + dr < node_dim):
            if board[q + dq, r + dr] == 0:
                if (0 <= q - dq < node_dim) and (0 <= r - dr < node_dim and board[q - dq, r - dr] == player):
                    ladder_forms.append(f'ladder_{q}_{r}')
    return ladder_forms


def detect_switchback(board, q, r, player, node_dim):
    switchbacks = []
    if board[q, r] == player:
        neighbors = get_neighbors(q, r, node_dim)
        connected_ladder = False
        for _, (nq, nr) in neighbors:
            if board[nq, nr] == player:
                connected_ladder = True
                break
        if connected_ladder:
            switchbacks.append(f'switchback_{q}_{r}')
    return switchbacks


def detect_double_threat(board, q, r, player, node_dim):
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    threats = []

    for i, (dq1, dr1) in enumerate(directions):
        nq1, nr1 = q + dq1, r + dr1
        if 0 <= nq1 < node_dim and 0 <= nr1 < node_dim and board[nq1, nr1] == player:
            for j, (dq2, dr2) in enumerate(directions):
                if i != j:
                    nq2, nr2 = q + dq2, r + dr2
                    if 0 <= nq2 < node_dim and 0 <= nr2 < node_dim and board[nq2, nr2] == player:
                        threats.append(f'double_threat_{q}_{r}')
                        break
    return threats


def detect_bridge(board, q, r, player, node_dim):
    bridges = []
    if board[q, r] != player:
        return bridges

    # Check for bridge patterns
    # Define the relative positions that form a bridge
    bridge_patterns = [
        [(1, 0), (0, 1)],  # East and South-East
        [(0, 1), (-1, 1)],  # South-East and South-West
        [(-1, 1), (-1, 0)],  # South-West and West
        [(-1, 0), (0, -1)],  # West and North-West
        [(0, -1), (1, -1)],  # North-West and North-East
        [(1, -1), (1, 0)],  # North-East and East
    ]

    for pattern in bridge_patterns:
        has_bridge = True
        for dq, dr in pattern:
            nq, nr = q + dq, r + dr
            if not (0 <= nq < node_dim and 0 <= nr < node_dim and board[nq, nr] == player):
                has_bridge = False
                break
        if has_bridge:
            bridges.append(f'bridge_{q}_{r}')
    return bridges


def add_expert_endgame_features(board, q, r, player, node_dim, node_properties):
    ladders = detect_ladder(board, q, r, player, node_dim)
    node_properties.extend(ladders)

    switchbacks = detect_switchback(board, q, r, player, node_dim)
    node_properties.extend(switchbacks)

    double_threats = detect_double_threat(board, q, r, player, node_dim)
    node_properties.extend(double_threats)

    bridges = detect_bridge(board, q, r, player, node_dim)
    node_properties.extend(bridges)


'''
End: Expert strategy patterns from http://www.mseymour.ca/hex_book/ 
'''


def calculate_stone_count(board, player):
    # checks for the amount of stones the player has on the board
    return np.sum(board == player)


def calculate_empty_cell_count(board):
    return np.sum(board == 0)


def generate_all_symbols(node_dim):
    # generates all symbols used for properties to the nodes

    symbols = [
        'black',
        'white',
        'empty',
        'edge',
        'corner',
        'black_high_betweenness',
        'black_low_betweenness',
        'black_high_closeness',
        'black_low_closeness',
        'black_high_eigenvector',
        'black_low_eigenvector',
        'white_high_betweenness',
        'white_low_betweenness',
        'white_high_closeness',
        'white_low_closeness',
        'white_high_eigenvector',
        'white_low_eigenvector',
        'black_neighbor_0',
        'black_neighbor_1',
        'black_neighbor_2',
        'black_neighbor_3',
        'black_neighbor_4',
        'black_neighbor_5',
        'black_neighbor_6',
        'white_neighbor_0',
        'white_neighbor_1',
        'white_neighbor_2',
        'white_neighbor_3',
        'white_neighbor_4',
        'white_neighbor_5',
        'white_neighbor_6',
        'empty_neighbor_0',
        'empty_neighbor_1',
        'empty_neighbor_2',
        'empty_neighbor_3',
        'empty_neighbor_4',
        'empty_neighbor_5',
        'empty_neighbor_6',
    ]

    # Generate proximity features based on node_dim size
    proximity_features = generate_proximity_features(node_dim)
    symbols.extend(proximity_features)

    directions = ['east', 'north_east', 'north_west', 'west', 'south_west', 'south_east']
    for dir_name in directions:
        for consecutive in range(2, node_dim + 1):
            symbols.append(f'pattern_{dir_name}_{consecutive}')

    for stone_count in range(node_dim * node_dim + 1):
        symbols.append(f'black_stone_count_{stone_count}')
        symbols.append(f'white_stone_count_{stone_count}')

    max_empty_cells = node_dim * node_dim
    for empty_count in range(max_empty_cells + 1):
        symbols.append(f'empty_cell_count_{empty_count}')

    symbols.extend([f'ladder_{q}_{r}' for q in range(node_dim) for r in range(node_dim)])
    symbols.extend([f'switchback_{q}_{r}' for q in range(node_dim) for r in range(node_dim)])
    symbols.extend([f'double_threat_{q}_{r}' for q in range(node_dim) for r in range(node_dim)])
    symbols.extend([f'bridge_{q}_{r}' for q in range(node_dim) for r in range(node_dim)])

    return symbols


def load_graph(file_path, hypervector_size, hypervector_bits, init_with=None):
    data = pd.read_csv(file_path, header=0, low_memory=False)
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    print(f"unique labels in Y: {np.unique(Y)}")

    number_of_examples = X.shape[0]
    node_dim = 7
    total_nodes = node_dim * node_dim

    column_names = data.columns.tolist()
    position_map = {}
    for idx, col_name in enumerate(column_names[:-1]):
        if col_name.startswith('cell'):
            q, r = map(int, col_name[4:].split('_'))
            position_map[idx] = (q, r)

    X_grids = np.zeros((number_of_examples, node_dim, node_dim))
    for idx in range(len(column_names) - 1):
        q, r = position_map[idx]
        X_grids[:, q, r] = X[:, idx]

    X_reshaped = X_grids

    symbols = generate_all_symbols(node_dim)

    graphs = Graphs(
        number_of_graphs=number_of_examples,
        symbols=symbols,
        hypervector_size=hypervector_size,
        hypervector_bits=hypervector_bits,
        init_with=init_with
    )

    for graph_id in tqdm(range(number_of_examples), desc="adding nodes to graphs"):
        graphs.set_number_of_graph_nodes(graph_id, total_nodes)

    graphs.prepare_node_configuration()

    for graph_id in tqdm(range(number_of_examples), desc="adding nodes to graphs", leave=False):
        for q in range(node_dim):
            for r in range(node_dim):
                node_name = f'node_{q}_{r}'
                neighbors = get_neighbors(q, r, node_dim)
                number_of_outgoing_edges = len(neighbors)
                graphs.add_graph_node(graph_id, node_name, number_of_outgoing_edges)

    graphs.prepare_edge_configuration()

    centrality_betweenness_black = []
    centrality_closeness_black = []
    centrality_eigenvector_black = []
    centrality_betweenness_white = []
    centrality_closeness_white = []
    centrality_eigenvector_white = []
    for graph_id in tqdm(range(number_of_examples), desc="calculating centrality measures", leave=False):
        board = X_reshaped[graph_id]
        betweenness_black = calculate_betweenness_centrality(board, 1, node_dim)
        closeness_black = calculate_closeness_centrality(board, 1, node_dim)
        eigenvector_black = calculate_eigenvector_centrality(board, 1, node_dim)
        betweenness_white = calculate_betweenness_centrality(board, -1, node_dim)
        closeness_white = calculate_closeness_centrality(board, -1, node_dim)
        eigenvector_white = calculate_eigenvector_centrality(board, -1, node_dim)
        centrality_betweenness_black.append(betweenness_black)
        centrality_closeness_black.append(closeness_black)
        centrality_eigenvector_black.append(eigenvector_black)
        centrality_betweenness_white.append(betweenness_white)
        centrality_closeness_white.append(closeness_white)
        centrality_eigenvector_white.append(eigenvector_white)

    for graph_id in tqdm(range(number_of_examples), desc="processing examples", leave=True):
        board = X_reshaped[graph_id]

        node_properties = {}
        for q in range(node_dim):
            for r in range(node_dim):
                node_name = f'node_{q}_{r}'
                node_properties[node_name] = []

        betweenness_black = centrality_betweenness_black[graph_id]
        closeness_black = centrality_closeness_black[graph_id]
        eigenvector_black = centrality_eigenvector_black[graph_id]
        betweenness_white = centrality_betweenness_white[graph_id]
        closeness_white = centrality_closeness_white[graph_id]
        eigenvector_white = centrality_eigenvector_white[graph_id]

        black_stone_count = calculate_stone_count(board, 1)
        white_stone_count = calculate_stone_count(board, -1)
        empty_cell_count = calculate_empty_cell_count(board)

        for q in range(node_dim):
            for r in range(node_dim):
                node_name = f'node_{q}_{r}'
                cell_value = board[q, r]

                if cell_value == 1:
                    node_properties[node_name].append('black')
                elif cell_value == -1:
                    node_properties[node_name].append('white')
                else:
                    node_properties[node_name].append('empty')

                if is_edge_cell(q, r, node_dim):
                    node_properties[node_name].append('edge')
                if is_corner_cell(q, r, node_dim):
                    node_properties[node_name].append('corner')

                if betweenness_black.get((q, r), 0) >= 0.1:
                    node_properties[node_name].append('black_high_betweenness')
                else:
                    node_properties[node_name].append('black_low_betweenness')

                if betweenness_white.get((q, r), 0) >= 0.1:
                    node_properties[node_name].append('white_high_betweenness')
                else:
                    node_properties[node_name].append('white_low_betweenness')

                if closeness_black.get((q, r), 0) >= 0.2:
                    node_properties[node_name].append('black_high_closeness')
                else:
                    node_properties[node_name].append('black_low_closeness')

                if closeness_white.get((q, r), 0) >= 0.2:
                    node_properties[node_name].append('white_high_closeness')
                else:
                    node_properties[node_name].append('white_low_closeness')

                if eigenvector_black.get((q, r), 0) >= 0.1:
                    node_properties[node_name].append('black_high_eigenvector')
                else:
                    node_properties[node_name].append('black_low_eigenvector')

                if eigenvector_white.get((q, r), 0) >= 0.1:
                    node_properties[node_name].append('white_high_eigenvector')
                else:
                    node_properties[node_name].append('white_low_eigenvector')

                neighbors = get_neighbors(q, r, node_dim)
                black_neighbors = sum(1 for (dir_offset, (nq, nr)) in neighbors if board[nq, nr] == 1)
                white_neighbors = sum(1 for (dir_offset, (nq, nr)) in neighbors if board[nq, nr] == -1)
                empty_neighbors = sum(1 for (dir_offset, (nq, nr)) in neighbors if board[nq, nr] == 0)

                black_neighbors = min(black_neighbors, 6)
                white_neighbors = min(white_neighbors, 6)
                empty_neighbors = min(empty_neighbors, 6)

                node_properties[node_name].append(f'black_neighbor_{black_neighbors}')
                node_properties[node_name].append(f'white_neighbor_{white_neighbors}')
                node_properties[node_name].append(f'empty_neighbor_{empty_neighbors}')

                prox_left = r  # distance to left edge
                prox_right = node_dim - 1 - r  # distance to right edge
                node_properties[node_name].append(f'black_prox_left_{prox_left}')
                node_properties[node_name].append(f'black_prox_right_{prox_right}')

                prox_top = q  # distance to top edge
                prox_bottom = node_dim - 1 - q  # distance to bottom edge
                node_properties[node_name].append(f'white_prox_top_{prox_top}')
                node_properties[node_name].append(f'white_prox_bottom_{prox_bottom}')

                patterns = generate_pattern_features(board, q, r, node_dim)
                for pattern in patterns:
                    node_properties[node_name].append(pattern)

                add_expert_endgame_features(board, q, r, cell_value, node_dim, node_properties[node_name])

                node_properties[node_name].append(f'black_stone_count_{black_stone_count}')
                node_properties[node_name].append(f'white_stone_count_{white_stone_count}')
                node_properties[node_name].append(f'empty_cell_count_{empty_cell_count}')

        # assign properties to graph nodes
        for node_name, properties in node_properties.items():
            for prop in properties:
                if prop in symbols:
                    graphs.add_graph_node_property(graph_id, node_name, prop)
                else:
                    # incase non-defined symbols are added
                    print(f"undefined symbol '{prop}' encountered")
                    symbols.append(prop)
                    graphs.add_graph_node_property(graph_id, node_name, prop)

    for graph_id in tqdm(range(number_of_examples), desc="adding edges", leave=True):
        for q in range(node_dim):
            for r in range(node_dim):
                node_name = f'node_{q}_{r}'
                neighbors = get_neighbors(q, r, node_dim)
                for _, (nq, nr) in neighbors:
                    neighbor_name = f'node_{nq}_{nr}'
                    edge_type = 'Plain'
                    try:
                        graphs.add_graph_node_edge(graph_id, node_name, neighbor_name, edge_type)
                    except Exception as e:
                        print(f"error adding edge {node_name} -> {neighbor_name} in graph {graph_id}: {e}")
                        continue

    graphs.encode()
    return graphs, Y, symbols, node_dim


def main():
    # Generate Clauses with a factor of 2 for 10 iterations
    clauses_options = [100 * (2 ** i) for i in range(10)]
    # Calculate T as 80% of each corresponding Clauses value
    T_options = [round(c * 0.8) for c in clauses_options]
    # Calculate S as 1% of each corresponding Clauses value
    s_options = [round(c * 0.01) for c in clauses_options]

    print("Clauses options:", clauses_options)
    print("T options (80% of Clauses):", T_options)
    print("S options (1% of Clauses):", s_options)

    epochs_options = [100]  # Adjust to a reasonable number for testing
    hypervector_size = 64
    hypervector_bits = 2

    results_df = pd.DataFrame(columns=['clauses', 'T', 's', 'epochs', 'train_accuracy', 'test_accuracy',
                                       'train_precision', 'test_precision', 'train_recall', 'test_recall',
                                       'train_f1_score', 'test_f1_score'])

    graphs_train, Y_train, symbols, node_dim = load_graph("dataset/hex_train.csv", hypervector_size, hypervector_bits)
    graphs_test, Y_test, _, _ = load_graph("dataset/hex_test.csv", hypervector_size, hypervector_bits,
                                           init_with=graphs_train)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparameter_grid = list(itertools.product(clauses_options, T_options, s_options, epochs_options))

    # Initialize variables to store the best results
    best_test_f1_score = 0
    best_result = None

    # Main progress bar for grid search and epoch tracking
    with tqdm(total=len(hyperparameter_grid) * epochs_options[0], desc="Grid Search", leave=True) as pbar:
        for idx, (clauses, T, s, epochs) in enumerate(hyperparameter_grid, 1):
            tsetlin_machine = MultiClassGraphTsetlinMachine(
                number_of_clauses=clauses,
                T=T,
                s=s,
                grid=(16 * 13, 1, 1),
                block=(128, 1, 1)
            )

            train_metrics = {'accuracy': [], 'recall': [], 'precision': [], 'f1_score': []}
            test_metrics = {'accuracy': [], 'recall': [], 'precision': [], 'f1_score': []}

            for epoch in range(1, epochs + 1):
                tsetlin_machine.fit(graphs_train, Y_train, epochs=1, incremental=True)

                train_predictions = tsetlin_machine.predict(graphs_train)
                accuracy_train = accuracy_score(Y_train, train_predictions)
                recall_train = recall_score(Y_train, train_predictions, average='weighted', zero_division=0)
                precision_train = precision_score(Y_train, train_predictions, average='weighted', zero_division=0)
                f1_train = f1_score(Y_train, train_predictions, average='weighted', zero_division=0)

                test_predictions = tsetlin_machine.predict(graphs_test)
                accuracy_test = accuracy_score(Y_test, test_predictions)
                recall_test = recall_score(Y_test, test_predictions, average='weighted', zero_division=0)
                precision_test = precision_score(Y_test, test_predictions, average='weighted', zero_division=0)
                f1_test = f1_score(Y_test, test_predictions, average='weighted', zero_division=0)

                train_metrics['accuracy'].append(accuracy_train)
                train_metrics['recall'].append(recall_train)
                train_metrics['precision'].append(precision_train)
                train_metrics['f1_score'].append(f1_train)

                test_metrics['accuracy'].append(accuracy_test)
                test_metrics['recall'].append(recall_test)
                test_metrics['precision'].append(precision_test)
                test_metrics['f1_score'].append(f1_test)

                # Update the progress bar for each epoch
                pbar.set_postfix({
                    "Clauses": clauses, "T": T, "s": s,
                    "Epoch": epoch, "Train Acc": accuracy_train, "Test Acc": accuracy_test
                })
                pbar.update(1)

            # After completing all epochs, check the final test f1 score for this configuration
            final_train_accuracy = train_metrics['accuracy'][-1]
            final_test_accuracy = test_metrics['accuracy'][-1]
            final_train_precision = train_metrics['precision'][-1]
            final_test_precision = test_metrics['precision'][-1]
            final_train_recall = train_metrics['recall'][-1]
            final_test_recall = test_metrics['recall'][-1]
            final_train_f1_score = train_metrics['f1_score'][-1]
            final_test_f1_score = test_metrics['f1_score'][-1]

            # Update the best result based on the final test f1 score
            if final_test_f1_score > best_test_f1_score:
                best_test_f1_score = final_test_f1_score
                best_result = {
                    'clauses': clauses, 'T': T, 's': s, 'epochs': epochs,
                    'train_accuracy': final_train_accuracy, 'test_accuracy': final_test_accuracy,
                    'train_precision': final_train_precision, 'test_precision': final_test_precision,
                    'train_recall': final_train_recall, 'test_recall': final_test_recall,
                    'train_f1_score': final_train_f1_score, 'test_f1_score': final_test_f1_score
                }

            # Store the final epoch results for each configuration
            results_df = pd.concat([results_df, pd.DataFrame([{
                'clauses': clauses,
                'T': T,
                's': s,
                'epochs': epochs,
                'train_accuracy': final_train_accuracy,
                'test_accuracy': final_test_accuracy,
                'train_precision': final_train_precision,
                'test_precision': final_test_precision,
                'train_recall': final_train_recall,
                'test_recall': final_test_recall,
                'train_f1_score': final_train_f1_score,
                'test_f1_score': final_test_f1_score
            }])], ignore_index=True)

    # Save all results to a CSV file and best result to a separate file
    results_filename = f"grid_search_results_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"Results saved to {results_filename}")

    # Save the best result to CSV
    if best_result:
        best_result_filename = f"best_result_{timestamp}.csv"
        pd.DataFrame([best_result]).to_csv(best_result_filename, index=False)
        print(f"Best result saved to {best_result_filename}")


if __name__ == "__main__":
    main()