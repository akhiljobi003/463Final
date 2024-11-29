import csv
import math
import heapq
from typing import List, Tuple, Dict
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

class Location:
    def __init__(self, name: str, street: str, city: str, state: str, lat: float, lon: float):
        self.name = name
        self.street = street
        self.city = city
        self.state = state
        self.lat = lat
        self.lon = lon

class Graph:
    def __init__(self):
        self.nodes: Dict[str, Location] = {}
        self.edges: Dict[str, List[Tuple[str, float]]] = {}

    def add_node(self, location: Location):
        self.nodes[location.name] = location
        self.edges[location.name] = []

    def add_edge(self, from_node: str, to_node: str, distance: float):
        self.edges[from_node].append((to_node, distance))
        self.edges[to_node].append((from_node, distance))

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def read_csv_data(filename: str) -> List[Location]:
    locations = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            location = Location(
                row['Name'], row['Street'], row['City'], row['State'],
                float(row['Latitude']), float(row['Longitude'])
            )
            locations.append(location)
    return locations

def build_graph(locations: List[Location]) -> Graph:
    graph = Graph()
    for location in locations:
        graph.add_node(location)

    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations[i+1:], start=i+1):
            distance = haversine_distance(loc1.lat, loc1.lon, loc2.lat, loc2.lon)
            graph.add_edge(loc1.name, loc2.name, distance)

    return graph

def dijkstra(graph: Graph, start: str, end: str) -> Tuple[float, List[str]]:
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    pq = [(0, start)]
    previous = {node: None for node in graph.nodes}

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == end:
            path = []
            while current_node:
                path.append(current_node)
                current_node = previous[current_node]
            return current_distance, path[::-1]

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph.edges[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return float('inf'), []

def geocode_city(city: str) -> Tuple[float, float]:
    geolocator = Nominatim(user_agent="police_station_finder")
    try:
        location = geolocator.geocode(city)
        if location:
            return location.latitude, location.longitude
        else:
            raise ValueError(f"Could not find coordinates for {city}")
    except (GeocoderTimedOut, GeocoderServiceError):
        raise RuntimeError("Geocoding service is unavailable. Please try again later.")

def find_nearest_station(user_lat: float, user_lon: float, locations: List[Location]) -> Location:
    nearest_station = min(locations, key=lambda loc: haversine_distance(user_lat, user_lon, loc.lat, loc.lon))
    return nearest_station

def main():
    locations = read_csv_data('locations.csv')
    graph = build_graph(locations)

    user_city = input("Enter your city: ")
    try:
        user_lat, user_lon = geocode_city(user_city)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        return

    nearest_station = find_nearest_station(user_lat, user_lon, locations)
    print(f"\nNearest police station: {nearest_station.name}")
    print(f"Address: {nearest_station.street}, {nearest_station.city}, {nearest_station.state}")
    print(f"Coordinates: ({nearest_station.lat}, {nearest_station.lon})")

    distance = haversine_distance(user_lat, user_lon, nearest_station.lat, nearest_station.lon)
    print(f"Distance: {distance:.2f} km")

    # Example of finding the optimal route between two stations
    start_station = locations[0].name
    end_station = locations[-1].name
    distance, path = dijkstra(graph, start_station, end_station)
    print(f"\nOptimal route from {start_station} to {end_station}:")
    print(" -> ".join(path))
    print(f"Total distance: {distance:.2f} km")

if __name__ == "__main__":
    main()
