import sys
from common import DatasetFile
import click
from parser import ChannelAnnouncement, ChannelUpdate, NodeAnnouncement
from tqdm import tqdm


@click.group()
def timemachine():
    pass

# (python .\__main.py timemachine restore .\gossip-20220823.gsp.bz2 1657843200 60 --normalise)
# https://storage.googleapis.com/lnresearch/gossip-20220823.gsp.bz2
@timemachine.command()
@click.argument("dataset", type=DatasetFile())
@click.argument("timestamp", type=int, required=False)
@click.argument("period", type=int, required=False)
@click.option("--normalise", is_flag=True, default=False)
@click.option("--policies", is_flag=True, default=False)
def restore(dataset, timestamp=None, period=None, normalise=False, policies=False):
    """
    Restore reconstructs the network topology at a specific time in the past.

    Restore replays gossip messages from a dataset and reconstructs
    the network as it would have looked like at the specified
    timestamp in the past. Discards channels and nodes that weren't updated in the last period days.

    Usage Template: python .\__main__.py timemachine restore <dataset> <optional: timestamp> <optional: period> <optional flag: --normalise> <optional flag: --policies>
    Usage Example:  python .\__main__.py timemachine restore ..\data\gossip-20220823.gsp.bz2 1657843200 60 --normalise --policies
                    Reconstructs the network topology with policies (fee and timelock policies) at Friday, July 15, 2022 12:00:00 AM,
                    taking into account last 60 days and normalizing all ids. Balances are NOT included but populated with zeroes.
    
    The network is then printed to 3 files:
    - nodes.json: a list of nodes
    - channels.json: a list of channels
    - edges.json: a list of edges"""

    # ----- SETUP -----
    # Defaults
    if timestamp is None:
        timestamp = 1657843200
    if period is None:
        period = 60
    cutoff = timestamp - period * 24 * 3600      # period in days
    edges = {}
    channels = {}
    nodes = {}

    # ----- REPLAY -----
    for m in tqdm(dataset, desc="Replaying gossip messages"):
        if isinstance(m, ChannelAnnouncement):
            edges[f"{m.short_channel_id}/0"] = {
                "id" : f"{m.short_channel_id}/0",
                "channel_id": m.short_channel_id,
                "counter_edge_id": f"{m.short_channel_id}/1",
                "source": m.node_ids[0].hex(),
                "destination": m.node_ids[1].hex(),
                "timestamp": 0,
                "fee_base(millisat)": 0,
                "fee_proportional": 0,
                "min_htlc": 0,
                "timelock": 0,
            }
            edges[f"{m.short_channel_id}/1"] = {
                "id" : f"{m.short_channel_id}/1",
                "channel_id": m.short_channel_id,
                "counter_edge_id": f"{m.short_channel_id}/0",
                "source": m.node_ids[1].hex(),
                "destination": m.node_ids[0].hex(),
                "timestamp": 0,
                "fee_base(millisat)": 0,
                "fee_proportional": 0,
                "min_htlc": 0,
                "timelock": 0,
            }
            channels[f"{m.short_channel_id}"] = {
                "id": m.short_channel_id,
                "edge1_id": f"{m.short_channel_id}/0",
                "edge2_id": f"{m.short_channel_id}/1",
                "node1_id": m.node_ids[0].hex(),
                "node2_id": m.node_ids[1].hex(),
            }

        elif isinstance(m, ChannelUpdate): # NOTE: It's an edge update (channel has 2 edges)
            scid = f"{m.short_channel_id}/{m.direction}"
            edge = edges.get(scid, None)
            ts = m.timestamp

            # Skip this update, it's in the future
            if ts > timestamp:
                continue
            # Skip updates that cannot possibly keep this channel alive
            if ts < cutoff:
                continue
            # Skip updates of edges that don't exist
            if edge is None:
                raise ValueError(
                    f"Could not find channel with short_channel_id {scid}"
                )
            # Skip this update, it's outdated (there was a different newer update)
            if edge["timestamp"] > ts:
                continue

            # Update the channel
            edge["timestamp"] = ts
            if policies:
                edge["fee_base(millisat)"] = m.fee_base_msat
                edge["fee_proportional"] = m.fee_proportional_millionths
                edge["min_htlc"] = m.htlc_minimum_msat
                edge["timelock"] = m.cltv_expiry_delta


        elif isinstance(m, NodeAnnouncement):
            node_id = m.node_id.hex()
            old = nodes.get(node_id, None)
            if old is not None and old["timestamp"] > m.timestamp:
                continue

            nodes[node_id] = {
                "id": node_id,
                "degree": 0,
                "timestamp": m.timestamp,
            }
    
    print(f"Number of nodes before pruning: {len(nodes)}")
    print(f"Number of channels before pruning: {len(channels)}")
    print(f"Number of edgesbefore pruning: {len(edges)}")
    print("Pruning...")

    # ----- PRUNING -----
    toDelete = set()
    for chan in channels.values():
        edgeA = edges.get(chan["edge1_id"])
        edgeB = edges.get(chan["edge2_id"])

        # If both edges are outdated, delete the channel
        if edgeA["timestamp"] < cutoff and edgeB["timestamp"] < cutoff:
            toDelete.add(chan["id"])
        # If nodes don't exist, delete the channel
        elif nodes.get(chan["node1_id"], None) is None or nodes.get(chan["node2_id"], None) is None:
            toDelete.add(chan["id"])
        else:
            # Update the node degrees
            nodes[chan["node1_id"]]["degree"] += 1
            nodes[chan["node2_id"]]["degree"] += 1

    # Delete the channels and their edges
    for scid in toDelete:
        del edges[channels[scid]["edge1_id"]]
        del edges[channels[scid]["edge2_id"]]
        del channels[scid]
    nodes = [n for n in nodes.values() if n["degree"] > 0]

    # ----- DATA NORMALIZATION -----
    if normalise:
        # normalise the node ids to be consecutive integers
        nodeToId = {}
        for i, n in enumerate(nodes):
            nodeToId[n["id"]] = i
            n["id"] = i
        for c in channels.values():
            c["node1_id"] = nodeToId[c["node1_id"]]
            c["node2_id"] = nodeToId[c["node2_id"]]
        for e in edges.values():
            e["source"] = nodeToId[e["source"]]
            e["destination"] = nodeToId[e["destination"]]

        # normalise channel ids to be consecutive integers
        channelToId = {}
        for i, c in enumerate(channels.values()):
            channelToId[c["id"]] = i
            c["id"] = i
        for e in edges.values():
            e["channel_id"] = channelToId[e["channel_id"]]

        # Normialize edge ids to be consecutive integers
        edgeToId = {}
        for i, e in enumerate(edges.values()):
            edgeToId[e["id"]] = i
            e["id"] = i
        for e in edges.values():
            e["counter_edge_id"] = edgeToId[e["counter_edge_id"]]
        for c in channels.values():
            c["edge1_id"] = edgeToId[c["edge1_id"]]
            c["edge2_id"] = edgeToId[c["edge2_id"]]

    # ----- OUTPUT -----
    if len(channels) == 0:
        print(
            "ERROR: no channels are left after pruning, make sure to select a timestamp that is covered by the dataset."
        )
        sys.exit(1)

    print(f"Number of nodes after pruning: {len(nodes)}")
    print(f"Number of channels after pruning: {len(channels)}")
    print(f"Number of edges after pruning: {len(edges)}")

    prefix = "../output/"

    # Save nodes to csv file
    with open(prefix + 'nodes.csv', 'w') as f:
        f.write('id' + '\n')
        for node in nodes:
            f.write(str(node["id"]) + '\n')

    # Save channels to csv file
    if policies:
        with open(prefix + 'channels.csv', 'w') as f:
            f.write('id,edge1_id,edge2_id,node1_id,node2_id,capacity(millisat)' + '\n')
            for channel in channels.values():
                f.write(str(channel["id"]) + ',' + str(channel["edge1_id"]) + ',' + str(channel["edge2_id"]) + ',' + str(channel["node1_id"]) + ',' + str(channel["node2_id"]) + ',0' + '\n')
    else:
        with open(prefix + 'channels.csv', 'w') as f:
            f.write('id,edge1_id,edge2_id,node1_id,node2_id' + '\n')
            for channel in channels.values():
                f.write(str(channel["id"]) + ',' + str(channel["edge1_id"]) + ',' + str(channel["edge2_id"]) + ',' + str(channel["node1_id"]) + ',' + str(channel["node2_id"]) + '\n')

    # Save edges to csv file
    if policies:
        with open(prefix + 'edges.csv', 'w') as f:
            f.write('id,channel_id,counter_edge_id,from_node_id,to_node_id,balance(millisat),fee_base(millisat),fee_proportional,min_htlc(millisat),timelock' + '\n')
            for edge in edges.values():
                f.write(str(edge["id"]) + ',' + str(edge["channel_id"]) + ',' + str(edge["counter_edge_id"]) + ',' + str(edge["source"]) + ',' + str(edge["destination"]) + ',0,' + str(edge["fee_base(millisat)"]) + ',' + str(edge["fee_proportional"]) + ',' + str(edge["min_htlc"]) + ',' + str(edge["timelock"]) + '\n')
    else:    
        with open(prefix + 'edges.csv', 'w') as f:
            f.write('id,channel_id,counter_edge_id,from_node_id,to_node_id' + '\n')
            for edge in edges.values():
                f.write(str(edge["id"]) + ',' + str(edge["channel_id"]) + ',' + str(edge["counter_edge_id"]) + ',' + str(edge["source"]) + ',' + str(edge["destination"]) + '\n')
