import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import click
from scipy.optimize import curve_fit
from common import dampened_power_law, modified_power_law, dampening


@click.group()
def grapher():
    pass

@grapher.command()
@click.option("--path", type=str, default="..\output", help="Path leading to a folder containing the topology files used.")
@click.option("--cropx", type=float, default=100, help="Excludes the x% of the highest values from the histogram.") 
@click.option("--percentage", is_flag=True, help="If set, the y-axis will be in percentage.")
@click.option("--fit", is_flag=True, help="If set, a curve will be fitted to the histogram.")
@click.option("--describe", is_flag=True, help="If set, the histogram will be described with statistics.")
def node_degrees(path=None, cropx=None, percentage=None, fit=None, describe=None):
    """
    Plots a histogram of node degree distribution.

    """

    # ----- LOAD DATA -----
    channels_df = pd.read_csv(path + '\channels.csv')

    # ----- CALCULATE NODE DEGREES -----
    # Count occurrences for each node in node1_id and node2_id columns separately
    node1_degrees = channels_df['node1_id'].value_counts()
    node2_degrees = channels_df['node2_id'].value_counts()
    # Combine the counts for each node and create a numpy array of the counts
    node_degrees = node1_degrees.add(node2_degrees, fill_value=0).astype(int).values

    # ----- DESCRIBE HISTOGRAM -----
    if describe:
        print("Number of nodes: ", len(node_degrees))
        print("Number of edges(bidirectional): ", sum(node_degrees)/2)
        print("Average degree: ", np.mean(node_degrees))
        print("Highest degree: ", max(node_degrees))
        print("Average number of nodes with degree > 100:", len(node_degrees[node_degrees > 100]))
        print("Percentage of nodes under 10:", len(node_degrees[node_degrees < 10])/len(node_degrees)*100)

    # ----- CREATE HISTOGRAM -----
    # Calculate the histogram of node degrees
    x_max = int(np.max(node_degrees))
    hist, bin_edges = np.histogram(node_degrees, bins=np.arange(0, x_max))
    if percentage:
        hist = (hist / len(node_degrees)) * 100

    # ----- FIT CURVE -----
    if fit:
        bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2 ) - 0.5
        popt, _ = curve_fit(dampened_power_law, bin_centers[1:], hist[1:], p0=[62.84231549,  1.60461977,  0.3809621, 0.01]) 
        print("Fitted function parameters: ", popt)

    # ----- PLOT HISTOGRAM AND FITTED CURVE -----
    plt.figure(figsize=(6, 5))
    # Bar histogram
    plt.bar(bin_edges[:-1], hist, width=1, edgecolor="k", align="center")
    # Fitted curve
    if fit:
        x_fit = np.linspace(0, x_max, x_max*10)
        y_fit = dampened_power_law(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r-', label="Fitted Curve", alpha=0.75)

    # ----- LABELING & DISPLAY -----
    title = "Node Degree Distribution " 
    if percentage:
        title += " (%)"
    plt.title(title)

    # X
    plt.xlabel("Number of Edges")
    if cropx < 100:
        x_max = int(np.percentile(node_degrees, cropx))
        plt.xlim(0, x_max)
    x_steps = max(5, x_max // 25)
    x_ticks = np.concatenate([np.arange(0, 10, 1), np.arange(10, x_max + 1, x_steps)])
    plt.xticks(x_ticks)

    # Y
    plt.ylim(0, hist.max()*1.1)
    if percentage:
        plt.ylabel("Percentage of Nodes")
        y_ticks = np.arange(0, 55, 5)
        y_tick_labels = [f"{tick:.0f}%" for tick in y_ticks]
        plt.yticks(y_ticks, y_tick_labels)
    else:
        plt.ylabel("Number of Nodes")
        y_ticks = np.arange(0, hist.max() + 1, 500)
        plt.yticks(y_ticks)
    plt.grid(axis='y', linestyle='--', linewidth=0.25)

    # ----- MISCELLANEOUS -----
    plt.show()

@grapher.command()
@click.option("--path", type=str, default="..\output", help="Path leading to a folder containing the topology files used.")
@click.option("--cropx", type=float, default=100, help="Excludes the x% of the highest values from the histogram.")
@click.option("--fit", is_flag=True, help="If set, a curve will be fitted to the data.")
def neighbour_node_degrees(path=None, cropx=None, fit=None):
    """
    Plots the average neighbour node degree per node degree.

    Example: 10 nodes with 1 edge each. 1 node with 10 edges. The 10 nodes connect to the 1 hub node.
    The average neighbour node degree for degree 1 nodes is 10 and for degree 10 nodes is 1.
    """

    # ----- LOAD DATA -----
    channels_df = pd.read_csv(path + '\channels.csv')

    # ----- CALCULATE NODE DEGREES -----
    node1_degrees = channels_df['node1_id'].value_counts()
    node2_degrees = channels_df['node2_id'].value_counts()
    node_degrees = node1_degrees.add(node2_degrees, fill_value=0).astype(int)
    x_max = int(np.max(node_degrees))

    # ----- CALCULATE AVERAGE NEIGHBOUR NODE DEGREE PER NODE DEGREE -----
    channels_df['node1_degree'] = channels_df['node1_id'].map(node_degrees)
    channels_df['node2_degree'] = channels_df['node2_id'].map(node_degrees)
    channels_df['degree_sum'] = channels_df['node1_degree'] + channels_df['node2_degree']

    channels_df_reversed = channels_df.rename(columns={'node1_id': 'node2_id', 'node2_id': 'node1_id', 'node1_degree': 'node2_degree', 'node2_degree': 'node1_degree'})
    bidir_channels_df = pd.concat([channels_df, channels_df_reversed], ignore_index=True)

    avg_dest_degree = bidir_channels_df.groupby('node1_degree')['degree_sum'].mean() - bidir_channels_df.groupby('node1_degree')['node1_degree'].mean()

    # ----- FIT CURVE -----
    if fit:
        x_data = avg_dest_degree.index
        y_data = avg_dest_degree.values
        popt, _ = curve_fit(modified_power_law, x_data, y_data, p0=[1.97479384e+03, 4.48052247e-01, 1.06724498e+01])
        # Error
        y_pred = modified_power_law(x_data, *popt)
        squared_diff = (y_data - y_pred)**2
        mse = np.mean(squared_diff)
        print(f"Mean Squared Error: {mse}")

    # ----- SCATTER PLOT & FIT CURVE -----
    plt.figure(figsize=(12, 5))
    plt.plot(avg_dest_degree.index, avg_dest_degree.values, marker='o', markersize=2, linestyle='', label="Data")
    if fit:
        x_fit = np.linspace(0, x_max, x_max*20)
        y_fit = modified_power_law(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r-', label="Fitted Curve", alpha=0.75)

    # ----- LABELING & DISPLAY -----
    plt.title("Neighbour Node Degree")

    # X
    plt.xlabel("Node Degree")
    if cropx < 100:
        x_max = int(np.percentile(node_degrees, cropx))
        plt.xlim(0, x_max)
    x_steps = max(5, x_max // 10)
    x_ticks = np.concatenate([np.arange(0, x_max + 1, x_steps)])
    plt.grid(axis='x', linestyle='--', linewidth=0.25)
    plt.xticks(x_ticks)

    # Y
    plt.ylabel("Average Neighbour Node Degree")
    y_ticks = np.arange(0, int(avg_dest_degree.max()) + 1, 50)
    plt.grid(axis='y', linestyle='--', linewidth=0.25)
    plt.yticks(y_ticks)
    plt.ylim(0, int(avg_dest_degree.max()*1.1))

    # ----- MISCELLANEOUS -----
    print("Average nehighbour node degree for degree 1 nodes:", avg_dest_degree[1])
    plt.show()

@grapher.command()
@click.argument("d", type=float, default=1.84942930e-03)
def plot_dampening(d=None):
    x_start = 0
    x_end = 5000
    points = 1000
    x_values = np.linspace(x_start, x_end, points)
    y_values = dampening(x_values, d)
    
    plt.plot(x_values, y_values)
    plt.xticks(np.arange(x_start, x_end+1, step=1000))
    plt.xlabel('x')
    plt.ylabel('Dampening function value')
    plt.title(f'Dampening function with d = {d}')

    for x in np.arange(1, x_end+1, step=1000):
        line_opacity = 0.05
        if x == 1:
            line_opacity = 0.3
        plt.axvline(x, linestyle='-', color='gray', alpha=line_opacity)

    for y in np.arange(1, 1.1, step=1000):
        line_opacity = 0.05
        if y == 1:
            line_opacity = 0.4
        plt.axhline(y, linestyle='-', color='gray', alpha=line_opacity)

    plt.show()