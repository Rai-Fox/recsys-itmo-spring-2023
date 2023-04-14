import itertools
import json
import pickle
from dataclasses import dataclass, field
from typing import List


@dataclass
class Track:
    track: int
    artist: str
    title: str
    recommendations: List[int] = field(default=lambda: [])
    predicted_scores: List[float] = field(default=lambda: [])


class Catalog:
    """
    A helper class used to load track data upon server startup
    and store the data to redis.
    """

    def __init__(self, app):
        self.app = app
        self.tracks = []
        self.my_tracks = []
        self.top_tracks = []

    def load(self, tracks_path, top_tracks_path, my_tracks_path):
        self.app.logger.info(f"Loading tracks from {tracks_path}")
        with open(tracks_path) as catalog_file:
            for j, line in enumerate(catalog_file):
                data = json.loads(line)
                self.tracks.append(
                    Track(
                        data["track"],
                        data["artist"],
                        data["title"],
                        data.get("recommendations", []),
                        data.get("predicted_times", []),
                    )
                )
        self.app.logger.info(f"Loaded {j+1} tracks")

        self.app.logger.info(f"Loading tracks with scores from {my_tracks_path}")
        with open(my_tracks_path) as catalog_file:
            for j, line in enumerate(catalog_file):
                data = json.loads(line)
                self.my_tracks.append(
                    Track(
                        data["track"],
                        data["artist"],
                        data["title"],
                        data.get("recommendations", []),
                        data.get("predicted_times", []),
                    )
                )
        self.app.logger.info(f"Loaded {j + 1} tracks")

        self.app.logger.info(f"Loading top tracks from {top_tracks_path}")
        with open(top_tracks_path) as top_tracks_path_file:
            self.top_tracks = json.load(top_tracks_path_file)
        self.app.logger.info(f"Loaded top tracks {self.top_tracks[:3]} ...")

        return self

    def upload_tracks(self, redis_tracks, redis_my_tracks):
        self.app.logger.info(f"Uploading tracks to redis")
        for track in self.tracks:
            redis_tracks.set(track.track, self.to_bytes(track))

        self.app.logger.info(
            f"Uploaded {len(self.tracks)} tracks"
        )

        self.app.logger.info(f"Uploading tracks with scores to redis")
        for track in self.my_tracks:
            redis_my_tracks.set(track.track, self.to_bytes(track))

        self.app.logger.info(
            f"Uploaded {len(self.my_tracks)} tracks"
        )

    def upload_artists(self, redis):
        self.app.logger.info(f"Uploading artists to redis")
        sorted_tracks = sorted(self.tracks, key=lambda track: track.artist)
        for j, (artist, artist_catalog) in enumerate(
            itertools.groupby(sorted_tracks, key=lambda track: track.artist)
        ):
            artist_tracks = [t.track for t in artist_catalog]
            redis.set(artist, self.to_bytes(artist_tracks))
        self.app.logger.info(f"Uploaded {j + 1} artists")

    def to_bytes(self, instance):
        return pickle.dumps(instance)

    def from_bytes(self, bts):
        return pickle.loads(bts)
