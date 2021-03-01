import os
import requests
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class DbotApiClient:

    def __init__(self):
        self.refresh = None
        self.access = None
        self.base_url = None
        self._load_tokens()

    def _load_tokens(self):
        load_dotenv()
        self.base_url = os.getenv("base_url")
        if not self.base_url.endswith("/"):
            self.base_url += "/"
        response = requests.post(self.base_url + "token", json={"username": os.getenv("bot_username"),
                                                                "password": os.getenv("bot_password")})
        if response.status_code == 200:
            res_json = response.json()
            self.access = res_json["access"]
            self.refresh = res_json["refresh"]
        else:
            logger.error("Unable to load jwt tokens\n{}".format(response.content))
            raise PermissionError("Unable to fetch jwt tokens")

    def _refresh_access_token(self):
        logger.debug("Refreshing access token")
        response = requests.post(self.base_url + "token/refresh", json={"refresh": self.refresh})
        if response.status_code == 200:
            self.access = response.json()["access"]
            return True

        logger.error("Unable to refresh access token")
        logger.error(response.content)
        return False

    def send_ow_event(self, hero: str, event: str, team: str):
        logger.debug(f"Sending event {event} {team} {hero}")
        response = self._post(self.base_url + "bot/ow_event",
                              json={"hero": hero, "event": event, "team": team})
        if not (200 <= response.status_code < 300):
            logger.error(f"send_ow_event returned response code {response.status_code}")
            logger.error(response.content)

    # smelly. make better
    def _post(self, url, **kwargs):
        response = requests.post(url, headers={'Authorization': 'Bearer {}'.format(self.access)}, **kwargs)
        if response.status_code == 401 and response.json().get("code") == "token_not_valid":
            if self._refresh_access_token():
                response = requests.post(url, headers={'Authorization': 'Bearer {}'.format(self.access)}, **kwargs)
        return response
