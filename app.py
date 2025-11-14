import sys
import time
from collections import defaultdict
from client import ConsiditionClient

# API Configuration
API_KEY = ""
BASE_URL = "https://api.considition.com"

# Map Configuration
MAP_NAME = ""
MAP_SEED = ""

# Global cache for paths and map data
path_cache = {}
station_index = {}
node_index = {}

# Track which customers we've already routed to charging stations
customer_has_been_routed_set = set()

# Persona weights for scoring
PERSONA_WEIGHTS = {
    "DislikesDriving": {"travel": 5.0, "time": 5.0, "energy": 1.0},
    "Stressed": {"travel": 10.0, "time": 10.0, "energy": 0.5},
    "CostSensitive": {"travel": 1.0, "time": 1.0, "energy": 8.0},
    "EcoConscious": {"travel": 1.0, "time": 1.0, "energy": 10.0},
    "Neutral": {"travel": 3.0, "time": 3.0, "energy": 3.0},
}

def build_graph(map_obj):
    """Build graph structure from map data"""
    global node_index, station_index
    
    nodes = map_obj.get("nodes", [])
    edges = map_obj.get("edges", [])
    
    # Index nodes
    for node in nodes:
        node_id = node.get("id")
        node_index[node_id] = node
        
        # Index charging stations
        target = node.get("target")
        if target and target.get("Type") == "ChargingStation":
            station_index[node_id] = {
                "node_id": node_id,
                "chargeSpeedPerCharger": target.get("chargeSpeedPerCharger", 150),
                "totalAmountOfChargers": target.get("totalAmountOfChargers", 10),
                "amountOfAvailableChargers": target.get("amountOfAvailableChargers", 10),
                "totalAmountOfBrokenChargers": target.get("totalAmountOfBrokenChargers", 0),
                "zone_id": node.get("zoneId"),
            }
    
    # Build adjacency list
    graph = defaultdict(list)
    for edge in edges:
        from_node = edge.get("fromNode")
        to_node = edge.get("toNode")
        length = edge.get("length", 0)
        
        graph[from_node].append({"node": to_node, "distance": length})
        graph[to_node].append({"node": from_node, "distance": length})
    
    return graph

def floyd_warshall(graph, nodes):
    """Compute all-pairs shortest paths"""
    global path_cache
    
    node_list = [n.get("id") for n in nodes]
    n = len(node_list)
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    
    # Initialize distance matrix
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    next_node = [[None] * n for _ in range(n)]
    
    # Distance to self is 0
    for i in range(n):
        dist[i][i] = 0
    
    # Set initial distances from edges
    for node_id, neighbors in graph.items():
        i = node_to_idx.get(node_id)
        if i is None:
            continue
        for neighbor in neighbors:
            j = node_to_idx.get(neighbor["node"])
            if j is not None:
                dist[i][j] = neighbor["distance"]
                next_node[i][j] = neighbor["node"]
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    # Reconstruct paths and store in cache
    def reconstruct_path(start_idx, end_idx):
        if next_node[start_idx][end_idx] is None:
            return []
        path = [node_list[start_idx]]
        while start_idx != end_idx:
            start_idx = node_to_idx[next_node[start_idx][end_idx]]
            path.append(node_list[start_idx])
        return path
    
    # Store in cache
    for i, from_node in enumerate(node_list):
        path_cache[from_node] = {}
        for j, to_node in enumerate(node_list):
            if dist[i][j] != INF:
                path_cache[from_node][to_node] = {
                    "distance": dist[i][j],
                    "path": reconstruct_path(i, j)
                }

def calculate_green_score(zone_id, zones, weather, current_tick):
    """Calculate green energy score for a zone (0-1, higher is greener)"""
    # Find zone
    zone = next((z for z in zones if z.get("id") == zone_id), None)
    if not zone:
        return 0.5
    
    energy_sources = zone.get("energySources", [])
    if not energy_sources:
        return 0.5
    
    green_score = 0.0
    total_capacity = 0.0
    
    hour = (current_tick * 5) // 60  
    is_daytime = 6 <= hour < 18
    
    # Weather data is not in the map_obj, but we use defaults
    cloud_cover = 0.5
    wind_strength = 0.5
    
    for source in energy_sources:
        source_type = source.get("type", "")
        capacity = source.get("generationCapacity", 1.0)
        total_capacity += capacity
        
        if source_type == "Hydro":
            green_score += capacity * 1.0
        elif source_type == "Nuclear":
            green_score += capacity * 0.9
        elif source_type == "Solar":
            if is_daytime:
                green_score += capacity * (1.0 - cloud_cover * 0.7)
            else:
                green_score += 0
        elif source_type == "Wind":
            green_score += capacity * wind_strength
        elif source_type == "NaturalGas":
            green_score += capacity * 0.3
        elif source_type == "Coal":
            green_score += capacity * 0.1
    
    return green_score / total_capacity if total_capacity > 0 else 0.5

def find_best_mandatory_station(customer, map_obj, current_tick):
    """
    Finds the best *reachable* station for a mandatory charge.
    "Best" is defined by the lowest persona-weighted cost, checking
    all stations on the map.
    """
    persona = customer.get("persona", "Neutral")

    customer_node = ""
    if customer.get("state") == "Home":
        customer_node = customer.get("fromNode")
    else:
        found = False
        for node in map_obj.get("nodes", []):
            for c in node.get("customers", []):
                if c.get("id") == customer.get("id"):
                    customer_node = node.get("id")
                    found = True
                    break
            if found: break
        if not found:
            for edge in map_obj.get("edges", []):
                 for c in edge.get("customers", []):
                    if c.get("id") == customer.get("id"):
                        customer_node = edge.get("fromNode")
                        found = True
                        break
                 if found: break
    
    if not customer_node:
        customer_node = customer.get("fromNode")

    to_node = customer.get("toNode", "")
    max_charge = customer.get("maxCharge", 60)
    
    current_charge = customer.get("chargeRemaining", 0)
    consumption = customer.get("energyConsumptionPerKm", 0.2)
    current_range_km = (current_charge / consumption) if consumption > 0 else 0
    
    planned_path = customer.get("path", [])
    if not planned_path:
        if customer_node and to_node and customer_node in path_cache:
            path_info = path_cache[customer_node].get(to_node)
            if path_info:
                planned_path = path_info.get("path", [])

    if customer_node not in path_cache or to_node not in path_cache[customer_node]:
        return None
        
    direct_trip_distance = path_cache[customer_node][to_node].get("distance")
    if direct_trip_distance is None:
        return None

    # Get persona-specific weights
    weights = PERSONA_WEIGHTS.get(persona, PERSONA_WEIGHTS["Neutral"])
    
    zones = map_obj.get("zones", [])
    weather = {}
    
    best_station = None
    best_cost = float('inf')

    # Evaluate all valid stations, letting the cost function naturally favor better options
    for station_node_id, station in station_index.items():
        
        # Skip full or broken stations
        if station.get("amountOfAvailableChargers", 0) <= 0:
            continue
        
        path_to_station = path_cache[customer_node].get(station_node_id)
        if not path_to_station:
            continue
        dist_to_station = path_to_station.get("distance")
        
        path_from_station = path_cache[station_node_id].get(to_node)
        if not path_from_station:
            continue
        dist_from_station = path_from_station.get("distance")

        if dist_to_station is None or dist_from_station is None:
            continue

        detour_distance = (dist_to_station + dist_from_station) - direct_trip_distance
        travel_cost = max(0, detour_distance) 
        
        charge_needed = max_charge * 0.8  
        charge_speed = station.get("chargeSpeedPerCharger", 150)
        charge_time = charge_needed / charge_speed if charge_speed > 0 else 10
        wait_time_proxy = station.get("totalAmountOfBrokenChargers", 0) / (station.get("amountOfAvailableChargers", 1) + 1)
        time_cost = charge_time + wait_time_proxy
        
        green_score = calculate_green_score(station["zone_id"], zones, weather, current_tick)
        energy_cost = 1.0 - green_score 

        total_cost = (
            weights["travel"] * travel_cost +
            weights["time"] * time_cost +
            weights["energy"] * energy_cost
        )
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_station = station_node_id
            
    return best_station

def handle_customer(customer, map_obj, current_tick):
    """
    Handle a single customer, enforcing the "Mandatory-First" charge rule.
    Every customer MUST charge at least once to be eligible for points.
    """
    customer_id = customer.get("id", "")
    state = customer.get("state", "")
    
    if state in ["DestinationReached", "RanOutOfJuice", "FailedToCharge", "Charging", "WaitingForCharger"]:
        return None
        
    if customer_id in customer_has_been_routed_set:
        return None
    
    best_station = find_best_mandatory_station(customer, map_obj, current_tick)
    
    if best_station:
        customer_has_been_routed_set.add(customer_id)
        
        return {
            "customerId": customer_id,
            "chargingRecommendations": [{"nodeId": best_station, "chargeTo": 0.8}]
        }

    return None

def generate_customer_recommendations(map_obj, current_tick):
    """Generate recommendations for all customers"""
    recommendations = []
    
    # Get customers from nodes
    nodes = map_obj.get("nodes", [])
    for node in nodes:
        customers = node.get("customers", [])
        for customer in customers:
            recommendation = handle_customer(customer, map_obj, current_tick)
            if recommendation:
                recommendations.append(recommendation)
    
    # Also check customers on edges (traveling customers)
    edges = map_obj.get("edges", [])
    for edge in edges:
        customers = edge.get("customers", [])
        for customer in customers:
            recommendation = handle_customer(customer, map_obj, current_tick)
            if recommendation:
                recommendations.append(recommendation)
    
    return recommendations

def should_move_on_to_next_tick(response):
    return True

def generate_tick(map_obj, current_tick):
    """
    Generates recommendations for the given tick based on the current map state.
    """

    recommendations = generate_customer_recommendations(map_obj, current_tick)
    
    return {
        "tick": current_tick,
        "customerRecommendations": recommendations,
    }

def main():
    api_key = API_KEY
    base_url = BASE_URL
    map_name = MAP_NAME

    client = ConsiditionClient(base_url, api_key)

    print("Fetching map data...")
    try:
        if base_url == "http://localhost:8080":
            map_obj = client.get_map(map_name)
        else:
            map_obj = client.get_map(map_name, MAP_SEED) # Assumes you modified client.py
            
    except Exception as e:
        print(f"Failed to fetch map: {e}")
        print("Is Docker running? Is client.py correct?")
        sys.exit(1)

    if not map_obj:
        print("Failed to fetch map!")
        sys.exit(1)

    # Build graph and compute shortest paths
    print("Building graph and computing shortest paths...")
    graph = build_graph(map_obj)
    floyd_warshall(graph, map_obj.get("nodes", []))
    print(f"Graph built, {len(station_index)} stations indexed.\n")

    # --- THIS IS THE CORRECT ITERATIVE LOOP ---
    
    current_map_state = map_obj
    total_ticks = int(map_obj.get("ticks", 0))
    all_recommendations = [] # This will store our decision history
    
    print(f"Starting simulation for {total_ticks} ticks...")

    for i in range(total_ticks):
        tick_data = generate_tick(current_map_state, i)
        
        all_recommendations.append(tick_data)


        input_payload = {
            "mapName": map_name,
            "playToTick": i + 1,  # Run the simulation TO this tick
            "ticks": all_recommendations,
        }
        
        if base_url != "http://localhost:8080":
             input_payload.pop("playToTick") # Not allowed on cloud API

        print(f"Playing tick: {i}")
        start = time.perf_counter()
        try:
            game_response = client.post_game(input_payload)
        except Exception as e:
            print(f"Error posting game data at tick {i}: {e}")
            sys.exit(1)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"Tick {i} took: {elapsed_ms:.2f}ms")

        if not game_response:
            print("Got no game response")
            sys.exit(1)

        current_map_state = game_response.get("map", current_map_state)
        
        final_score = game_response.get("score", 0)

        # This logic is for the *final* submission, not local.
        # For local, we must run all ticks.
        if base_url != "http://localhost:8080":
            break # Cloud API runs all ticks at once, so we stop after one post.

    print(f"\n{'='*30}\nSIMULATION COMPLETE\n{'='*30}")
    print(f"Final score: {final_score}")
    print(f"kWh Revenue: {game_response.get('kwhRevenue', 0)}")
    print(f"Customer Score: {game_response.get('customerCompletionScore', 0)}")
    print(f"Game ID (gid): {game_response.get('GameId', 'N/A - Did not beat high score')}")

if __name__ == "__main__":
    main()