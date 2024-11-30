import csv
import math
import heapq
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import difflib
from functools import lru_cache
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Enhanced Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)


@dataclass
class Location:
    name: str
    street: str
    city: str
    state: str
    lat: float
    lon: float

    def __repr__(self) -> str:
        return f"{self.name} ({self.street}, {self.city}, {self.state})"


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


@lru_cache(maxsize=128)
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def read_csv_data(filename: str) -> List[Location]:
    locations = []
    try:
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    location = Location(
                        row['Name'], row['Street'], row['City'], row['State'],
                        float(row['Latitude']), float(row['Longitude'])
                    )
                    locations.append(location)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid row: {e}")
        if not locations:
            raise ValueError("No valid locations found in the CSV file")
        return locations
    except FileNotFoundError:
        logger.error(f"File {filename} not found")
        raise
    except csv.Error as e:
        logger.error(f"CSV reading error: {e}")
        raise


def build_graph(locations: List[Location]) -> Graph:
    graph = Graph()
    for location in locations:
        graph.add_node(location)

    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations[i + 1:], start=i + 1):
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


@lru_cache(maxsize=32)
def geocode_city(city: str, max_attempts: int = 3) -> Tuple[float, float]:
    geolocator = Nominatim(user_agent="police_station_finder")
    for attempt in range(max_attempts):
        try:
            location = geolocator.geocode(city)
            if location:
                return location.latitude, location.longitude
            raise ValueError(f"Could not find coordinates for {city}")
        except (GeocoderTimedOut, GeocoderServiceError):
            if attempt == max_attempts - 1:
                raise RuntimeError("Geocoding service is unavailable")
            logger.warning(f"Geocoding attempt {attempt + 1} failed")


def find_nearest_station(user_lat: float, user_lon: float, locations: List[Location]) -> Location:
    nearest_station = min(locations, key=lambda loc: haversine_distance(user_lat, user_lon, loc.lat, loc.lon))
    return nearest_station


def find_best_match_city(user_input: str, locations: List[Location]) -> Optional[str]:
    cities = {loc.city for loc in locations}
    match = difflib.get_close_matches(user_input, cities, n=1, cutoff=0.6)
    return match[0] if match else None


def main():
    console = Console()
    try:
        locations = read_csv_data('police_stations.csv')
        graph = build_graph(locations)

        user_city = console.input("[bold green]Enter your city: [/]")
        matched_city = find_best_match_city(user_city, locations)
        if not matched_city:
            console.print(f"[bold red]No close match found for {user_city}[/]")
            return

        user_lat, user_lon = geocode_city(matched_city)

        nearest_station = find_nearest_station(user_lat, user_lon, locations)

        table = Table(title=f"Nearest Police Station to {matched_city}")
        table.add_column("Station Name", style="cyan")
        table.add_column("Address", style="magenta")
        table.add_column("Distance (km)", justify="right")

        distance = haversine_distance(user_lat, user_lon, nearest_station.lat, nearest_station.lon)
        table.add_row(
            nearest_station.name,
            f"{nearest_station.street}, {nearest_station.city}, {nearest_station.state}",
            f"{distance:.2f}"
        )

        console.print(table)

        # Example of finding the optimal route between two stations
        start_station = locations[0].name
        end_station = locations[-1].name
        distance, path = dijkstra(graph, start_station, end_station)
        console.print(f"\n[bold green]Optimal route from {start_station} to {end_station}:[/]")
        console.print(" -> ".join(path))
        console.print(f"[bold]Total distance:[/] {distance:.2f} km")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
