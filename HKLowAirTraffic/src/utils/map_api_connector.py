import requests
import json
from typing import Tuple, List

# 地图 API 集成类，负责从 Google Maps 获取地图数据，提供地点查询、路线规划、海拔信息等服务
class MapAPIConnector:
    # 初始化地图 API 集成器，选择地图数据提供商（"google"）并设置 API 密钥
    def __init__(self, api_key: str, service_provider: str = "google"):
        # Google或其他API服务商
        self.api_key = api_key
        self.service_provider = service_provider

    # 获取地点信息，如建筑物、街景等，使用 Google Places API
    def get_place_info(self, location: str) -> dict:
        if self.service_provider == "google":
            url = (
                f"https://maps.googleapis.com/maps/api/place/textsearch/json?"
                f"query={location}&key={self.api_key}"
            )
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to fetch place data: {response.text}")
        else:
            raise ValueError("Currently only Google API is supported for place information.")

    # 获取起点和终点之间的路线信息，使用 Google Directions API
    def get_route_data(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> dict:
        if self.service_provider == "google":
            url = (
                f"https://maps.googleapis.com/maps/api/directions/json?"
                f"origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}&key={self.api_key}"
            )
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to fetch route data: {response.text}")
        else:
            raise ValueError("Currently only Google API is supported for route data.")

    # 获取指定位置的海拔数据，使用 Google Elevation API
    def get_elevation_data(self, location: Tuple[float, float]) -> dict:
        if self.service_provider == "google":
            url = (
                f"https://maps.googleapis.com/maps/api/elevation/json?"
                f"locations={location[0]},{location[1]}&key={self.api_key}"
            )
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to fetch elevation data: {response.text}")
        else:
            raise ValueError("Currently only Google API is supported for elevation data.")
