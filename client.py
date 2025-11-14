import requests

class ConsiditionClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"x-api-key": self.api_key}
    
    def post_game(self, data: object):
        return self.request("POST", "/api/game", json=data)

    def get_map(self, map_name: str, seed: int = None):
        params = {"mapName": map_name}
        if seed is not None:
            params["seed"] = seed
        return self.request("GET", "/api/map", params=params)

    def request(self, method: str, endpoint: str, **kwargs):
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, headers=self.headers, verify=False, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Error response status: {e.response.status_code}")
                print(f"Error response body: {e.response.text}")
            raise
