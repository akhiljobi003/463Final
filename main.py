import csv
import math
import heapq
import logging
import argparse
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import difflib
import json
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
    """Enhanced Location class using dataclass for cleaner representation"""
    name: str
    street: str
    city: str
    state: str
    lat: float
    lon: float

    def to_dict(self) -> Dict[str, str | float]:
        """Convert location to dictionary"""
        return asdict(self)

    def __repr__(self) -> str:
        """String representation of Location"""
        return f"{self.name} ({self.street}, {self.city}, {self.state})"

class LocationGraph:
    def __init__(self):
        self.nodes: Dict[str, Location] = {}
        self.adjacency_list: Dict[str, List[Tuple[str, float]]] = {}

    def add_location(self, location: Location):
        self.nodes[location.name] = location
        self.adjacency_list[location.name] = []

    def add_connection(self, from_node: str, to_node: str, distance: float):
        self.adjacency_list[from_node].append((to_node, distance))
        self.adjacency_list[to_node].append((from_node, distance))

    def dijkstra(self, start: str) -> Dict[str, float]:
        distances = {node: float('infinity') for node in self.nodes}
        distances[start] = 0
        pq = [(0, start)]

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in self.adjacency_list[current_node]:
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return distances


class LocationFinder:
    def __init__(self, locations_file: str):
        self.locations = self._read_csv_data(locations_file)
        self.graph = self._build_graph()
        self.console = Console()

    def _build_graph(self) -> LocationGraph:
        graph = LocationGraph()
        for location in self.locations:
            graph.add_location(location)
        for i, loc1 in enumerate(self.locations):
            for loc2 in self.locations[i + 1:]:
                distance = self._haversine_distance(loc1.lat, loc1.lon, loc2.lat, loc2.lon)
                graph.add_connection(loc1.name, loc2.name, distance)
        return graph

    def find_nearest_stations(self, user_lat: float, user_lon: float, top_n: int = 3) -> List[Location]:
        # Create a temporary node for the user's location
        user_node = Location("User", "", "", "", user_lat, user_lon)
        self.graph.add_location(user_node)

        # Connect user node to all other nodes
        for location in self.locations:
            distance = self._haversine_distance(user_lat, user_lon, location.lat, location.lon)
            self.graph.add_connection("User", location.name, distance)

        # Run Dijkstra's algorithm from the user's location
        distances = self.graph.dijkstra("User")

        # Sort locations by distance and return top N
        sorted_locations = sorted(
            [(dist, self.graph.nodes[name]) for name, dist in distances.items() if name != "User"])

        # Remove the temporary user node
        self.graph.nodes.pop("User")
        self.graph.adjacency_list.pop("User")

        return [location for _, location in sorted_locations[:top_n]]

    def display_nearest_stations(self, user_lat: float, user_lon: float, top_n: int = 3):
        nearest_stations = self.find_nearest_stations(user_lat, user_lon, top_n)
        table = Table(title=f"Top {top_n} Nearest Stations")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Station Name", style="magenta")
        table.add_column("Address", style="green")
        table.add_column("Distance (km)", justify="right", style="yellow")

        for i, station in enumerate(nearest_stations, 1):
            distance = self._haversine_distance(user_lat, user_lon, station.lat, station.lon)
            table.add_row(
                str(i),
                station.name,
                f"{station.street}, {station.city}, {station.state}",
                f"{distance:.2f}"
            )

        self.console.print(table)

    def _read_csv_data(self, filename: str) -> List[Location]:
        """Enhanced CSV reading with more robust error handling"""
        try:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                locations = []

                for row in reader:
                    # Slight modification to handle the CSV format
                    try:
                        # Split the state to extract the correct state abbreviation
                        state_parts = row['City'].split(',')[-1].strip().split()
                        state = state_parts[0] if state_parts else 'Unknown'

                        location = Location(
                            name=row['Name'],
                            street=row['Street'],
                            city=row['City'].split(',')[0].strip(),
                            state=state,
                            lat=float(row['Latitude']),
                            lon=float(row['Longitude'])
                        )
                        locations.append(location)
                    except (KeyError, ValueError, IndexError) as e:
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

    def _build_graph(self) -> LocationGraph:
        """Construct graph from locations"""
        graph = LocationGraph()
        for location in self.locations:
            graph.add_location(location)

        for i, loc1 in enumerate(self.locations):
            for loc2 in self.locations[i+1:]:
                distance = self._haversine_distance(loc1.lat, loc1.lon, loc2.lat, loc2.lon)
                graph.add_connection(loc1.name, loc2.name, distance)

        return graph

    @staticmethod
    @lru_cache(maxsize=128)
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat/2)**2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def find_nearest_stations(self, user_lat: float, user_lon: float, top_n: int = 3) -> List[Location]:
        stations_with_distance = [
            (self._haversine_distance(user_lat, user_lon, loc.lat, loc.lon), loc)
            for loc in self.locations
        ]
        return [station for _, station in sorted(stations_with_distance)[:top_n]]

    def display_nearest_stations(self, user_lat: float, user_lon: float, top_n: int = 3):
        nearest_stations = self.find_nearest_stations(user_lat, user_lon, top_n)
        table = Table(title=f"Top {top_n} Nearest Stations")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Station Name", style="magenta")
        table.add_column("Address", style="green")
        table.add_column("Distance (km)", justify="right", style="yellow")

        for i, station in enumerate(nearest_stations, 1):
            distance = self._haversine_distance(user_lat, user_lon, station.lat, station.lon)
            table.add_row(
                str(i),
                station.name,
                f"{station.street}, {station.city}, {station.state}",
                f"{distance:.2f}"
            )
        self.console.print(table)

    @staticmethod
    @lru_cache(maxsize=32)
    def geocode_city(city: str, max_attempts: int = 3) -> Tuple[float, float]:
        """Geocode city with retry mechanism"""
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

    def find_best_match_city(self, user_input: str) -> Optional[str]:
        """Find best matching city using fuzzy matching"""
        cities = {loc.city for loc in self.locations}
        match = difflib.get_close_matches(user_input, cities, n=1, cutoff=0.6)
        return match[0] if match else None

def main():
    parser = argparse.ArgumentParser(description="Advanced Location Finder")
    parser.add_argument('--city', type=str, help="City to find nearest locations")
    parser.add_argument('--input-file', default='police_stations.csv', help="Input CSV file with locations")
    parser.add_argument('--top-n', type=int, default=5, help="Number of nearest stations to display")
    args = parser.parse_args()

    console = Console()
    logger = logging.getLogger(__name__)

    try:
        location_finder = LocationFinder(args.input_file)
        user_city = args.city or console.input("[bold green]Enter your city: [/]")
        matched_city = location_finder.find_best_match_city(user_city)

        if not matched_city:
            console.print(f"[bold red]No close match found for {user_city}[/]")
            return

        user_lat, user_lon = location_finder.geocode_city(matched_city)
        console.rule(f"[bold blue]Top {args.top_n} Nearest Locations to {matched_city}[/]")
        location_finder.display_nearest_stations(user_lat, user_lon, args.top_n)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
