import pandas as pd
import math
import heapq
from itertools import pairwise
from collections import defaultdict, deque
import dash_leaflet as dl

import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go

# Constants
Vc = 90 # Cruise speed in Knots
Vw = -10 # Wind speed in Knots
h = 4000 # Altitude in feet
hm = h/3.2808399 # Convert feet to meters

# Get atmospheric data
from ambiance import Atmosphere
sea_level = Atmosphere(0)
rho0_a = sea_level.density
rho0 = rho0_a[0]

# Calculate the density at altitude
a = Atmosphere(hm)
rho_a = a.density
rho  = rho_a[0]

# Calculate the velocity
Vt = Vc * rho0/rho
V = Vt+Vw
Vm = V*1.852/60 # Convert Knots to km/min


# -------------------------------------------------
# Load the coordinate table once at start-up
# -------------------------------------------------
airports = pd.read_csv("airports_coordinates.csv")

coords = airports.set_index("ID")[["Latitude (Decimal)",
                                   "Longitude (Decimal)"]]
KENT = coords.loc["KENT"]

# -------------------------------------------------
# Creates a list of nodes
# -------------------------------------------------
nodes = pd.read_csv("final_recharge_nodes.csv", header=None, names=["ID"])

graph1 = {n: [] for n in nodes["ID"].values} # create empty graph that is bidirectional
graph2 = {n: [] for n in nodes["ID"].values} # create empty graph

# -------------------------------------------------
# Haversine helper
# -------------------------------------------------
EARTH_RADIUS_KM = 6_371.0088       # mean Earth radius

def haversine(lat1, lon1, lat2, lon2, *, r=EARTH_RADIUS_KM) -> float:
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)

    a = (math.sin(dφ/2)**2 +
         math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2)
    return 2 * r * math.asin(math.sqrt(a))

# -------------------------------------------------
# Compute path length + leg table
# -------------------------------------------------
def path_length(path_ids):
    hops, total = [], 0.0
    for a, b in zip(path_ids[:-1], path_ids[1:]):
        lat1, lon1 = coords.loc[a]
        lat2, lon2 = coords.loc[b]
        d_km = haversine(lat1, lon1, lat2, lon2)
        total += d_km
        hops.append((a, b, d_km))
    return total, hops

# -------------------------------------------------
# Create a graph with edges
# -------------------------------------------------
for n in nodes["ID"]:
    for c in coords.iterrows():

        if n == c[0]:
            continue

        ids = c[0]
        path_ids = [n, ids]
        total_km, legs = path_length(path_ids)

        if total_km > 105: # skip long hops; there is no need to add them
            continue

        # Handle special cases for specific nodes that needs priority
        # Such as the nodes across the border, which needs to pass
        # through border control such as CYPT and TOL.
        # The code below ensures that the routes when crossing the border
        # pass through the correct nodes and are added to the graph
        if n == "TOL" and ids == "CYPT":
            graph1[n].append((ids, total_km))
            continue
        if n == "CYPT" and ids == "TOL":
            continue
        if ids == "CYPT":
            graph2[n].append((ids, total_km))
            continue

        if n == "CYPT" and ids == "CLM2":
            graph1[n].append((ids, total_km))
        elif n == "CYPT":
            continue
        if ids == "CLM2":
            continue
        if n == "CLM2" and ids == "3W2" or n == "CLM2" and ids == "BASS" or n == "CLM2" and ids == "89D" or n == "CLM2" and ids == "LPR" or n == "CLM2" and ids == "PHN" or n == "CLM2" and ids == "ARB" or n == "ARB" and ids == "CLM2":
            continue
        if n == "CYPT" and ids == "ONZ" or n == "CLM2" and ids == "ONZ" or n == "DET" and ids == "ONZ" or n == "DET" and ids == "CGL2" or n == "PHN" and ids == "CYZR":
            continue
        if n == "NZ3" and ids == "PHN" or n == "PHN" and ids == "NZ3" or n == "NZ3" and ids == "LPR" or n == "LPR" and ids == "NZ3" or n == "CYTZ" and ids == "63NY":
            continue
        if n == "IAG" and ids == "CNF9" or n == "IAG" and ids == "CNQ3" or n == "IAG" and ids == "NY39" or n == "NY39" and ids == "IAG":
            continue
        if n == "CYSN" and ids == "IAG" or n == "CYSN" and ids == "CNF9" or n == "CYSN" and ids == "CNQ3" or n == "CYSN" and ids == "CYTZ":
            graph1[n].append((ids, total_km))
            continue
        elif n == "CYSN" or ids == "CYSN":
            continue
        if n == "CSX7" and ids == "Y83" or n == "Y83" and ids == "CSX7" or n == "CSX7" and ids == "PHN" or n == "PHN" and ids == "CSX7" or n == "CYTZ" and ids == "NY06" or n == "NY06" and ids == "CYTZ":
            continue
        if n == "CYPQ" and ids == "NY06" or n == "NY06" and ids == "CYPQ" or n == "89NY" and ids == "CNL3" or n == "CNU4" and ids == "SDC" or n == "SDC" and ids == "CNU4":
            continue
        if n == "CRL2" and ids == "89NY" or n == "89NY" and ids == "CRL2":
            continue
        if ids == "CYAM" or ids == "CPF2":
            continue
        if n == "CYAG" and ids == "13Y" or n == "CYAG" and ids == "43Y" or n == "43Y" and ids == "CYAG" or n == "CYAG" and ids == "ORB" or n == "ORB" and ids == "CYAG":
            continue 
        
        # Add the edges to the graph based on the total distance
        if total_km <= 52:
            graph1[n].append((ids, total_km))
        elif ids in nodes["ID"].values:  # if the other ID is a node
            graph2[n].append((ids, total_km))
            continue
        elif total_km <= 55:
            graph1[n].append((ids, total_km))

# Make sure the graph1 (<50km) is bidirectional
for u in list(graph1.keys()):                                   # iterate over a static copy of the keys
    for v, w in graph1[u]:                                      # for every edge u → v (weight w)
        back_edges = [nbr for nbr, _ in graph1.get(v, [])]
        if u not in back_edges:                                 # if v → u isn’t there yet
            graph1.setdefault(v, []).append((u, w))

# -------------------------------------------------
# Combine the two graphs
# -------------------------------------------------
def combine_keep_all(g1, g2):
    merged = defaultdict(list)

    for g in (g1, g2):           # walk both graphs
        for u, nbrs in g.items():    # every source node
            merged[u].extend(nbrs)   # tack on its edges

    return dict(merged)

graph = combine_keep_all(graph1, graph2)

for u, nbrs in list(graph.items()): # Ensures that every node is a key
    for v, _ in nbrs:
        graph.setdefault(v, [])

# -------------------------------------------------
# Dijkstra’s algorithm
# -------------------------------------------------
def dijkstra_all_paths(graph, start, goal):
    pq      = [(0, start)]
    dist    = {start: 0}            # node -> shortest-known distance
    parent  = defaultdict(list)     # node -> [all predecessors]

    while pq:                       # ❶ keep going; no early break
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue                # stale entry

        for v, w in graph.get(u, []):
            nd = d + w              # tentative distance to v
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                parent[v] = [u]     # ❷ new best ⇒ reset predecessor list
                heapq.heappush(pq, (nd, v))
            elif nd == dist[v]:     # ❸ tie ⇒ *add* another predecessor
                parent[v].append(u)

    #  Build all paths from start to goal
    paths, stack = [], deque([[goal]])
    while stack:
        path = stack.pop()
        last = path[-1]
        if last == start:
            paths.append(path[::-1])
        else:
            for p in parent[last]:
                stack.append(path + [p])

    return dist.get(goal, float("inf")), paths

# -------------------------------------------------
# Create markers for each airport
# -------------------------------------------------

ICON_GREEN = dict(
    iconUrl   ="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png",
    shadowUrl ="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-shadow.png",
    iconSize       =[25, 41],
    iconAnchor     =[12, 41],
    shadowSize     =[41, 41],
    shadowAnchor   =[12, 41],
    popupAnchor    =[1, -34]
)

ICON_BLUE = dict(
    iconUrl   ="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png",
    shadowUrl ="https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-shadow.png",
    iconSize       =[25, 41],
    iconAnchor     =[12, 41],
    shadowSize     =[41, 41],
    shadowAnchor   =[12, 41],
    popupAnchor    =[1, -34]
)

node_set = set(nodes["ID"])          # faster lookup than .values each time

all_markers = [
    dl.Marker(
        position=[coords.loc[i]["Latitude (Decimal)"],
                  coords.loc[i]["Longitude (Decimal)"]],
        icon=ICON_GREEN if i in node_set else ICON_BLUE,
        children=dl.Tooltip(i)
    )
    for i in coords.index
]

# -------------------------------------------------
# Create the legends for wind speed and direction
# -------------------------------------------------

# ── Wind-speed colour bar ─────────────────────────────────────────
speed_legend = html.Div(
    [
        html.B("Wind speed (kt)"),
        html.Div([
            html.Div("0",  style={"background":"#d8d8e9"}),
            html.Div("5",  style={"background":"#f3b1f3"}),
            html.Div("10", style={"background":"#ea74ff"}),
            html.Div("15", style={"background":"#a35cff"}),
            html.Div("20", style={"background":"#6175ff"}),
            html.Div("25", style={"background":"#29d2ff"}),
            html.Div("30", style={"background":"#1edd7a"}),
            html.Div("35", style={"background":"#c8f000"}),
            html.Div("40", style={"background":"#ffb200"}),
            html.Div("45", style={"background":"#ff5a36"}),
            html.Div("60", style={"background":"#b20034"})
        ], style={
            "display":"grid",
            "gridTemplateColumns":"repeat(11, 1fr)",
            "gridGap":"0",
            "height":"15px"
        })
    ],
    style={
        "position":"absolute", "bottom":"15px", "left":"10px",
        "background":"rgba(255,255,255,0.8)",
        "padding":"8px 10px", "fontSize":"13px",
        "zIndex":"1000"
    }
)

# ── Wind-direction colour bar ────────────────────────────────────
dir_legend = html.Div(
    [
        html.B("Wind dir (°)"),
        html.Div([
            html.Div("N",   style={"background":"#6e7fc0"}),
            html.Div("NE",  style={"background":"#00a0c2"}),
            html.Div("E",   style={"background":"#009b7a"}),
            html.Div("SE",  style={"background":"#4a7c39"}),
            html.Div("S",   style={"background":"#b96969"}),
            html.Div("SW",  style={"background":"#8a6334"}),
            html.Div("W",   style={"background":"#6e7c32"}),
            html.Div("NW",  style={"background":"#775c8c"})
        ], style={
            "display":"grid",
            "gridTemplateColumns":"repeat(8, 1fr)",
            "gridGap":"0",
            "height":"15px"
        })
    ],
    style={
        "position":"absolute", "bottom":"55px", "left":"10px",
        "background":"rgba(255,255,255,0.8)",
        "padding":"8px 10px", "fontSize":"12px",
        "zIndex":"1000"
    }
)



# -------------------------------------------------
# Create the Dash app
# -------------------------------------------------

app = dash.Dash(__name__)
app.title = "Electric Aircraft Route Map"

# -------------------- TITLE --------------------
title = html.Div(
    [
        html.H1("Electric Aircraft Route Finder", style={"textAlign": "center"}),
    ],
    style={
        "display": "flex",           # put children on one row
        "justifyContent": "center",  # ⤴︎ center horizontally
        "alignItems": "center",      #   center vertically
        "gap": "12px",               # space between blocks
        "padding": "6px 0"
    }   
)

# -------------------- TOP-BAR --------------------
top_bar = html.Div(
    [
        html.Div([
            html.Label("Departure:"),
            dcc.Dropdown(coords.index, "KENT", id="from_id",
                         style={"width": 200})
        ], style={"margin": "0 10px"}),

        html.Div([
            html.Label("Destination:"),
            dcc.Dropdown(coords.index, "KENT", id="to_id",
                         style={"width": 200})
        ], style={"margin": "0 10px"}),

        html.Div(id="output-text", style={"margin": "0 10px",
                                          "fontWeight": "bold"})
    ],
    style={
        "display": "flex",           # put children on one row
        "justifyContent": "center",  # ⤴︎ center horizontally
        "alignItems": "center",      #   center vertically
        "gap": "12px",               # space between blocks
        "padding": "6px 0"
    }
)

app.layout = html.Div([
    title,
    top_bar,
    
    dl.Map(
        id="map",
        center=[coords["Latitude (Decimal)"].mean(), coords["Longitude (Decimal)"].mean()],
        zoom=8,
        style={'width': '100%', 'height': '800px'},
        children=[
            dl.LayersControl([
                dl.BaseLayer(dl.TileLayer(), name="OSM", checked=False),
                dl.BaseLayer(
                    dl.TileLayer(
                        url="https://tiles.arcgis.com/tiles/ssFJjBXIUyZDrSYZ/arcgis/rest/services/VFR_Sectional/MapServer/tile/{z}/{y}/{x}",
                        attribution="FAA VFR Sectional – tiles.arcgis.com"
                    ), name="FAA VFR", checked=True
                ),
                dl.Overlay(dl.WMSTileLayer(
                    url="https://digital.weather.gov/ndfd/wms",
                    layers="ndfd.conus.windspd",
                    format="image/png",
                    transparent=True,
                    attribution="NWS Wind Speed"
                ), name="Wind Speed", checked=False),
                dl.Overlay(dl.WMSTileLayer(
                    url="https://digital.weather.gov/ndfd/wms",
                    layers="ndfd.conus.winddir",
                    format="image/png",
                    transparent=True,
                    attribution="NWS Wind Direction"
                ), name="Wind Direction", checked=False),
                dl.LayerGroup(id="route-layer", children=all_markers)
            ]), speed_legend, dir_legend

        ]
    )

])

@app.callback(
    [Output("route-layer", "children"), Output("output-text", "children")],
    [Input("from_id", "value"), Input("to_id", "value")]
)
def update_route(from_id, to_id):
    output = "Select two different airports."
    children = all_markers.copy()
    warning = "\n WARNING: This route is in the limit of the aircraft range, and although it is possible to fly this route and back, \n it is not recommended. Please check the weather and power before departure."
    if from_id != to_id:
        if from_id not in graph or to_id not in graph:
            return children, f"ERROR: One or both airports not connected in graph."

        dist, paths = dijkstra_all_paths(graph, from_id, to_id)
        if not paths:
            return children, f"No valid route found."

        best = min(paths, key=len)
        total_km, _ = path_length(best)

        best = min(paths, key=len)             # shortest by hops
        positions = [
            [coords.loc[i]["Latitude (Decimal)"],
             coords.loc[i]["Longitude (Decimal)"]]
            for i in best
        ]
        
        line = dl.Polyline(positions=positions,
                           color="red", weight=4, opacity=0.9)
        children.append(line)

        output = f"Route: {' → '.join(best)}\nTotal distance: {dist:.2f} km\nEstimated time: {((dist/Vm)+12*len(best)+70*(len(best)-1))/60:.2f} hrs"

        last_start, last_end = best[-2], best[-1]            # last hop only
        d_last, _ = path_length([last_start, last_end])

        if d_last > 52 and last_end not in node_set:         # node_set = set(nodes["ID"])
            output += "\n" + warning


    return children, output

app = dash.Dash(__name__)
server = app.server

if __name__ == "__main__":
    app.run(debug=False, port=8050, host="0.0.0.0")
