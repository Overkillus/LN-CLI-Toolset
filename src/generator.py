from random import random
import click
import pandas
import numpy as np
from tqdm import tqdm
from common import get_truncated_normal, modified_power_law, dampened_power_law
import networkx as nx
import pandas as pd


@click.group()
def generator():
    pass

# (python .\__main.py timemachine restore .\gossip-20220823.gsp.bz2 1657843200 60 --normalise)
# https://storage.googleapis.com/lnresearch/gossip-20220823.gsp.bz2
@generator.command()
@click.argument("seed", type=int)
@click.option("--fee_base", nargs=2, type=(int, int), default=(100, 1000), help="fee_base (average, deviation)")
@click.option("--fee_base_default", type=(int, float), default=(0, 0.45), help="fee_base_default (value, percentage of users), rest uses normal distribution")
@click.option("--fee_proportional", nargs=2, type=(int, int), default=(1, 250), help="fee_proportional (average, deviation)")
@click.option("--fee_proportional_default", type=(int, float), default=(1, 0.3), help="fee_proportional_default (value, percentage of users), rest uses normal distribution")
@click.option("--min_htlc", nargs=2, type=(int, int), default=(1000, 2_000), help="min_htlc (average, deviation)")
@click.option("--min_htlc_default", type=(int, float), default=(1000, 0.8), help="min_htlc_default (value, percentage of users), rest uses normal distribution")
@click.option("--timelock", nargs=2, type=(int, int), default=(40, 80), help="timelock (average, deviation)")
@click.option("--timelock_default", type=(int, float), default=(40, 0.73), help="timelock_default (value, percentage of users), rest uses normal distribution")
@click.option("--path", type=str, default="..\output") # path leading to a folder contating the topology files used
def Parameters(seed=None, path=None, fee_base=None, fee_base_default=None, fee_proportional=None, fee_proportional_default=None, min_htlc=None, min_htlc_default=None, timelock=None, timelock_default=None):
    """
    Parameters generates artificial transaction edge parameters (for balances see Generate Balances).

    Generate Parameters createss random edge parameters and adds them to the network topology file.
    Generated parameters are:
    - edge fee_base
    - edge fee_proportional
    - edge min_htlc
    - edge timelock
    
    The parameters are added to the topology files of the network:
    - edges.json: a list of edges
    """

    # ----- LOAD DATA -----
    fee_base_avg, fee_base_dev = fee_base
    fee_base_default_value, fee_base_default_percentage = fee_base_default
    fee_proportional_avg, fee_proportional_dev = fee_proportional
    fee_proportional_default_value, fee_proportional_default_percentage = fee_proportional_default
    min_htlc_avg, min_htlc_dev = min_htlc
    min_htlc_default_value, min_htlc_default_percentage = min_htlc_default
    timelock_avg, timelock_dev = timelock
    timelock_default_value, timelock_default_percentage = timelock_default

    np.random.seed(seed)
    # Load data from files
    with open(path + "\edges.csv", "r") as f:
        edges = pandas.read_csv(f, index_col=0)

    # ----- GENERATION -----
    # Initialize parameters
    edges["fee_base(millisat)"] = 0
    edges["fee_proportional"] = 0
    edges["min_htlc(millisat)"] = 0
    edges["timelock"] = 0

    for i in tqdm(range(len(edges)), desc="Generating parameters"):
        # Generate edge fee_base
        if random() < fee_base_default_percentage:
            fee_base = fee_base_default_value
        else:
            fee_base = int(get_truncated_normal(fee_base_avg, fee_base_dev, 0, 100_000_000))
        edges["fee_base(millisat)"][i] = fee_base

        # Generate edge fee_proportional
        if random() < fee_proportional_default_percentage:
            fee_proportional = fee_proportional_default_value
        else:
            fee_proportional = int(get_truncated_normal(fee_proportional_avg, fee_proportional_dev, 0, 100_000_000))
        edges["fee_proportional"][i] = fee_proportional

        # Generate edge min_htlc
        if random() < min_htlc_default_percentage:
            min_htlc = min_htlc_default_value
        else:
            min_htlc = int(get_truncated_normal(min_htlc_avg, min_htlc_dev, 0, 100_000_000))
        edges["min_htlc(millisat)"][i] = min_htlc

        # Generate edge timelock
        if random() < timelock_default_percentage:
            timelock = timelock_default_value
        else:
            timelock = int(get_truncated_normal(timelock_avg, timelock_dev, 0, 1_000))
        edges["timelock"][i] = timelock

    # ----- OUTPUT -----
    print(edges.describe())
    print(edges.head())

    # Save data to files
    edges = edges.reindex(columns=["channel_id", "counter_edge_id", "from_node_id", "to_node_id", "balance(millisat)", "fee_base(millisat)", "fee_proportional", "min_htlc(millisat)", "timelock"])
    with open("..\output\edges.csv", "w", newline='') as f:
        edges.to_csv(f)

@generator.command()
@click.argument("seed", type=int)
@click.argument("average", type=int)
@click.argument("channel_deviation", type=int) 
@click.argument("edge_deviation", type=int) # deviation in percent, e.g. 20 for 20%
@click.option("--min", type=int, default=100_000)
@click.option("--max", type=int, default=10e15)
@click.option("--path", type=str, default="..\output") # path leading to a folder contating the topology files used
@click.option("--describe", is_flag=True, default=False, help="Prints the description of the generated data.")
def Balances(path=None, seed=None, average=None, channel_deviation=None, edge_deviation=None, min=None, max=None, describe=None):
    """
    Generate Balances generates artificial channel and edge balances.

    Generate Balances creates sudo-random channel and edge balances and adds them to the network topology file.
    Generated balances are:
    - channel capacity
    - edge balance
    
    The balances are added to the topology files of the network:
    - nodes.csv: a list of nodes
    - channels.csv: a list of channels
    - edges.csv: a list of edges
    """

    # ----- LOAD DATA -----
    np.random.seed(seed)
    # Load data from files
    with open(path + "\\nodes.csv", "r") as f:
        nodes = pandas.read_csv(f, index_col=0)
    with open(path + "\channels.csv", "r") as f:
        channels = pandas.read_csv(f, index_col=0)
    with open(path + "\edges.csv", "r") as f:
        edges = pandas.read_csv(f, index_col=0)

    # ----- GENERATION -----
    # Initialize parameters
    channels["capacity(millisat)"] = 0
    edges["balance(millisat)"] = 0

    for i in tqdm(range(len(channels)), desc="Generating balances"):
        # Generate channel capacity
        capacity = int(get_truncated_normal(average, channel_deviation, min, max))
        channels["capacity(millisat)"][i] = capacity

        # Generate edge balance distribution deviation (0-1)
        balance_distribution = float(get_truncated_normal(0.5, edge_deviation/100, 0, 1))
        edges["balance(millisat)"][channels["edge1_id"][i]] = int(capacity * balance_distribution)
        edges["balance(millisat)"][channels["edge2_id"][i]] = capacity - edges["balance(millisat)"][channels["edge1_id"][i]]

    # ----- OUTPUT -----
    # Reorder columns
    channels = channels.reindex(columns=["edge1_id", "edge2_id", "node1_id", "node2_id", "capacity(millisat)"])
    edges = edges.reindex(columns=["channel_id", "counter_edge_id", "from_node_id", "to_node_id", "balance(millisat)", "fee_base(millisat)", "fee_proportional", "min_htlc(millisat)", "timelock"])

    # Save data to files
    if describe:
        print(channels.describe())
        print(edges.describe())

    with open("..\output\\nodes.csv", "w", newline='') as f:
        nodes.to_csv(f)
    with open("..\output\channels.csv", "w", newline='') as f:
        channels.to_csv(f)
    with open("..\output\edges.csv", "w", newline='') as f:
        edges.to_csv(f)

@generator.command()
@click.argument("seed", type=int)
@click.argument("n_nodes", type=int)
@click.option("--multiplier", type=float, default=1, help="Streches (X-axis) the fitting curve. Lower values result in higher average node degree (more equal distribution). Very high values will fail due to unconnectedness of the network. (Default: 1)")
@click.option("--attempts", type=int, default=100, help="Number of attempts to generate a network with the given parameters as the process can rarely fail. (Default: 100)")
@click.option("--stats", is_flag=True, help="Prints statistics about the generated network. (Default: False)")
@click.option("--runs", type=int, default=1, help="Number of network generations to run. (Default: 1)")
@click.option("--no_save", is_flag=True, help="Disables saving the generated network. (Default: False)")
def network(n_nodes, seed=None, attempts=None, stats=None, runs=None, multiplier=None, no_save=None):
    """
    Network creates a network with n nodes based on the fitted degree distribution of LN snapshots using a modified Havel-Hakimi algorithm.
    Particularly relevant settings are n_nodes settign the size of the network and the multiplier adjusting the degree distribution (average degree).
    """

    # ----- INPUT VALIDATION -----
    if multiplier < 0:
        print("ERROR: multiplier must be non-negative.")
        return

    # ----- DEGREE DISTRIBUTION -----
    degree_distribution = _generate_degree_distribution(n_nodes, multiplier)

    # ----- GENERATION -----
    G = set()
    for i in tqdm(range(runs), desc="Generating networks", position=0, leave=True):
        g = _generate_network(n_nodes, attempts, degree_distribution, seed+i)
        if g is not None:
            G.add(g)

    # ----- OUTPUT -----
    if stats:
        _stats(G)
    if len(G) > 0:
        if not no_save:
            _save_network(G)
    else:
        print("Failed to generate network in ", attempts, " attempts.")

# Generates a degree distribution based on the LN snapshot from epoch time 1657843200 with 16900 nodes
def _generate_degree_distribution(n_nodes, multiplier):
    # params_no_dampening = [108.02193399,  1.78110723,  0.9284658 ] # parameters fitted to LN snapshot from epoch time 1657843200 with 16900 nodes - power law
    params = [1.07293038e+02, 1.77732194e+00, 9.23805145e-01, 1.84942930e-03] # parameters fitted to LN snapshot from epoch time 1657843200 with 16900 nodes - power law + dampening
    # degree_distribution = [modified_power_law(x*multiplier, *params_no_dampening) for x in range(1, n_nodes)]
    degree_distribution = [dampened_power_law(x*multiplier, *params) for x in range(1, n_nodes)]
    total = sum(degree_distribution)
    degree_distribution = [x / total for x in degree_distribution]
    return degree_distribution

# Repeatedly attempts to create a network with the given degree distribution. Returns None if it fails.
def _generate_network(n_nodes, attempts, degree_distribution, seed):
    rng = np.random.default_rng(seed)
    # Generation attempts
    for _ in range(attempts):
        # Generate a degree sequence
        degrees = list(rng.choice(range(1, n_nodes), size=n_nodes, p=degree_distribution))

        # Attempt graph generation
        g = _realize_degrees_connected(degrees)
        if g is not None:
            return g
    return None

# Tries to generate a connected loopless multigraph the degree sequence. Return None if it fails.
def _realize_degrees_connected(degrees):
    # Initial checks
    total = sum(degrees)
    # Ensure the summ is even
    if total % 2 != 0: 
        return None
    # Ensure the degree sequence is multigraphical
    if not (total >= 2 * max(degrees)):
        return None

    # Generate graph
    g = nx.MultiGraph()
    degrees = sorted(degrees, reverse=True)
    for __ in tqdm(range(total // 2), desc="Generating edges", position=1, leave=False):
        # Decrease the degree of the node with the smallest degree
        degrees[-1] -= 1

        # Decrease the degree of the node with the highest degree
        # find highest index with maximum degree [4, 4, (4), 3, 3, 2, 1, 1]
        h = 0
        for h in range(len(degrees)-1):
            if degrees[h] != degrees[h+1]:
                degrees[h] -= 1
                break

        # Save channel
        g.add_edge(len(degrees)-1, h)

        # Remove nodes with degree 0
        if degrees[-1] == 0:
            degrees.pop(len(degrees)-1)
        if degrees[h] == 0:
            degrees.pop(h)

        # Success check
        if len(degrees) == 0 and nx.is_connected(g) and nx.number_of_selfloops(g) == 0:
            return g
    print("Failed due to potential unconectedness of the degree sequence.")
    return None

# Tries to generate a connected loopless multigraph the degree sequence using a preferential connection method. Return None if it fails.
def _realize_degrees_connected_preferential(degrees, rng):
        # Initial checks
        total = sum(degrees)
        # Ensure the summ is even
        if total % 2 != 0: 
            return None
        # Ensure the degree sequence is multigraphical
        if not (total >= 2 * max(degrees)):
            return None

        # Generate graph
        g = nx.MultiGraph()
        # For every edge to be generated connect the smallest remaining degree node to a random node weighted by it's remaining degree
        for e in tqdm(range(total // 2), desc="Generating edges", position=1, leave=False):
            # Find the smallest remaining degree node
            min_degree = min(filter(lambda d: d > 0, degrees))
            min_index = degrees.index(min_degree)

            # slice of degrees with remaining degrees
            if len([d for d in degrees if d > 0]) <= 1:
                print([d for d in degrees if d > 0])
                return None

            # Weight nodes based on their remaining degrees
            dd = [d*d*(rng.random()/5) for d in degrees]
            s = sum(dd)
            weights = [d/s for d in dd]

            # Choose a random node weighted by it's remaining degree
            selected_index = rng.choice(range(len(degrees)), p=weights)
            while selected_index == min_index:
                selected_index = rng.choice(range(len(degrees)), p=weights)

            # Connect the nodes
            g.add_edge(min_index, selected_index)

            # Update the degrees of the connected nodes
            degrees[min_index] -= 1
            degrees[selected_index] -= 1
        
        if sum(degrees) == 0 and nx.is_connected(g):
            return g

        return None

# Prints statistics about the generated networks
def _stats(G):
    print("Network statistics:")

    num_nodes = np.mean([g.number_of_nodes() for g in G])
    edges = [g.number_of_edges() for g in G]
    degrees = [np.array([x[1] for x in g.degree()]) for g in G]
    nodes_degree_gt_100 = [np.sum(d > 100) for d in degrees]
    highest_degrees = [np.max(d) for d in degrees]
    avg_degrees = [np.mean(d) for d in degrees]

    neighbours_degrees = [
        np.mean([x[1] for x in g.degree(g.neighbors(node))])
        for g, d in zip(G, degrees)
        for node in g.nodes()
        if g.degree(node) == 1
    ]

    neighbours_degrees = [np.mean(neighbours_degrees[i : i + len(d)]) for i, d in enumerate(degrees)]

    print("Average number of nodes: ", num_nodes)
    print("Average number of edges: ", np.mean(edges), " Standard Deviation: ", np.std(edges))
    print("Average degree: ", np.mean(avg_degrees), " Standard Deviation: ", np.std(avg_degrees))
    print("Average highest degree: ", np.mean(highest_degrees), " Standard Deviation: ", np.std(highest_degrees))
    print("Average number of nodes with degree > 100:", np.mean(nodes_degree_gt_100), " Standard Deviation: ", np.std(nodes_degree_gt_100))
    print("Average degree of neighbours of nodes with degree 1: ", np.mean(neighbours_degrees), " Standard Deviation: ", np.std(neighbours_degrees))

# Saves the generated networks to a csv file
def _save_network(G):
    for i, g in enumerate(G):
                # Nodes
                nodes = pd.DataFrame({"id": range(g.number_of_nodes())})

                # Channels
                channels = nx.to_pandas_edgelist(g)
                channels.rename(columns={"source": "node1_id", "target": "node2_id"}, inplace=True)
                channels["edge1_id"] = 0
                channels["edge2_id"] = 0
                channels = channels[["edge1_id", "edge2_id", "node1_id", "node2_id"]]

                # Edges
                edges_list = []  # list to store new edge rows
                for j, row in channels.iterrows():
                    edge1_id = 2 * j
                    edge2_id = 2 * j + 1
                    edge1 = {"channel_id": j, "counter_edge_id": edge2_id, "from_node_id": row["node1_id"], "to_node_id": row["node2_id"]}
                    edge2 = {"channel_id": j, "counter_edge_id": edge1_id, "from_node_id": row["node2_id"], "to_node_id": row["node1_id"]}
                    edges_list.append(edge1)
                    edges_list.append(edge2)
                    # Update channels DataFrame with edge IDs
                    channels.at[j, "edge1_id"] = edge1_id
                    channels.at[j, "edge2_id"] = edge2_id
                edges = pd.DataFrame(edges_list, columns=["channel_id", "counter_edge_id", "from_node_id", "to_node_id"])

                # Write
                numbering = "" if i == 0 else "_" + str(i)
                nodes_name = "nodes" + numbering + ".csv"
                channels_name = "channels" + numbering + ".csv"
                edges_name = "edges" + numbering + ".csv"
                with open("..\output\\" + nodes_name, "w", newline='') as f:
                    nodes.to_csv(f, index=False)
                with open("..\output\\" + channels_name, "w", newline='') as f:
                    channels.to_csv(f, index_label="id")
                with open("..\output\\" + edges_name, "w", newline='') as f:
                    edges.to_csv(f, index_label="id")

# ----------- Standard Graphs -----------

@generator.command()
@click.argument("seed", type=int)
@click.argument("n_nodes", type=int)
def gn_graph(n_nodes, seed=None):
    """
    """
    k = lambda x: x * np.e ** (-x)
    g = nx.gn_graph(n_nodes, seed=seed, kernel=k)
    g = g.to_undirected()

    channels = nx.to_pandas_edgelist(g)
    channels.rename(columns={"source": "node1_id", "target": "node2_id"}, inplace=True)

    with open("..\output\channels.csv", "w", newline='') as f:
        channels.to_csv(f)

@generator.command()
@click.argument("seed", type=int)
@click.argument("n_nodes", type=int)
@click.argument("n_initial_edges", type=int)
def BarabasiAlbertGraph(n_nodes, n_initial_edges, seed=None):
    """
    BarabasiAlbertGraph generates a network based on the Barabasi-Albert graph.

    BarabasiAlbertGraph creates a Barabasi-Albert graph with n nodes and m edges per node.
    """
    g = nx.barabasi_albert_graph(n_nodes, n_initial_edges, seed=seed)

    channels = nx.to_pandas_edgelist(g)
    channels.rename(columns={"source": "node1_id", "target": "node2_id"}, inplace=True)

    with open("..\output\channels.csv", "w", newline='') as f:
        channels.to_csv(f)

@generator.command()
@click.argument("seed", type=int)
@click.argument("n_nodes", type=int)
@click.argument("p", type=float)
def gnp(n_nodes, p, seed=None):
    """
    """
    g = nx.gnp_random_graph(n_nodes, p, seed=seed)

    channels = nx.to_pandas_edgelist(g)
    channels.rename(columns={"source": "node1_id", "target": "node2_id"}, inplace=True)

    with open("..\output\channels.csv", "w", newline='') as f:
        channels.to_csv(f)