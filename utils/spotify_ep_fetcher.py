import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import os
from utils.logger_setup import logger

#SPOTIPY Auth Part
class EpisodeFetcher:

    def __init__(self, uri):
        try:
            client_id = os.environ['SPOTIPY_CLIENT_ID']
            secret = os.environ['SPOTIPY_CLIENT_SECRET']
            auth_manager = SpotifyClientCredentials(client_id, secret)
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            self.uri = uri
        except Exception as e:
            logger.error(f"Exception occurred while authenticating via spotify api: {e}")
            
    def get_episodes(self) -> pd.DataFrame:
        try:
            podcast_uri = 'spotify:show:' + self.uri
            results = self.sp.show_episodes(podcast_uri)
            episodes = results['items']
            while results['next']:
                results = self.sp.next(results)
                episodes.extend(results['items'])    

            df = pd.DataFrame(episodes)
            return df
        except Exception as e:
            logger.error("Error while retreiving episodes: {e}")
