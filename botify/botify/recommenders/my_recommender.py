import numpy as np

from .toppop import TopPop
from .recommender import Recommender
from ..track import Track
import random


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class MyRecommender(Recommender):
    def __init__(
            self,
            app,
            tracks_redis,
            artists_redis,
            user_sessions_redis,
            catalog,
            min_good_time=0.5,
            min_good_predicted_score=0.5,
            log_predict_score=True,
            n_top_tracks=100
    ):
        self.app = app
        self.tracks_redis = tracks_redis
        self.artists_redis = artists_redis
        self.user_sessions_redis = user_sessions_redis

        self.catalog = catalog
        self.top_tracks = catalog.top_tracks[:n_top_tracks]

        self.min_good_time = min_good_time
        self.min_good_predicted_score = min_good_predicted_score
        self.log_predict_score = log_predict_score

    def _get_user_tracks(self, user: int) -> dict:
        user_tracks = self.user_sessions_redis.get(user)

        if user_tracks is not None:
            user_tracks = self.catalog.from_bytes(user_tracks)
        else:
            user_tracks = {'all_tracks': [], 'good_tracks': []}

        return user_tracks

    def _recommend(self, user: int, track: int, prev_track_time: float):
        user_tracks = self._get_user_tracks(user)

        if track not in user_tracks['good_tracks'] and prev_track_time > self.min_good_time:
            user_tracks['good_tracks'].append(track)
        if track not in user_tracks['all_tracks']:
            user_tracks['all_tracks'].append(track)

        self.user_sessions_redis.set(user, self.catalog.to_bytes(user_tracks))

        return track

    def _fallback(self, user_tracks: dict):
        ind = None
        while ind is None or self.top_tracks[ind] in user_tracks['all_tracks']:
            ind = int(np.random.randint(0, len(self.top_tracks)))
        return self.top_tracks[ind]

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        user_tracks = self._get_user_tracks(user)

        if prev_track_time < self.min_good_time and user_tracks['good_tracks']:
            prev_track = int(np.random.choice(user_tracks['good_tracks']))

        previous_track = self.tracks_redis.get(prev_track)
        if previous_track is None:
            return self._recommend(
                user,
                self._fallback(user_tracks),
                prev_track_time
            )

        previous_track = self.catalog.from_bytes(previous_track)

        if not previous_track.recommendations or not previous_track.predicted_scores:
            return self._recommend(
                user,
                self._fallback(user_tracks),
                prev_track_time
            )

        return self._recommend(
            user,
            self._recommend_by_predicted_scores(previous_track, user_tracks),
            prev_track_time
        )

    def _recommend_by_predicted_scores(
            self,
            previous_track: Track,
            user_tracks: dict,
    ):
        recommendations = previous_track.recommendations

        if set(recommendations) - set(user_tracks['all_tracks']) == set():
            return self._fallback(user_tracks)

        recommendations_scores = previous_track.predicted_scores
        if self.log_predict_score:
            recommendations_scores = np.log(np.array(recommendations_scores) + 0.001)
        else:
            recommendations_scores = np.array(recommendations_scores)
        recommendation_probs = softmax(recommendations_scores)

        ind = None
        while ind is None or ind in user_tracks['all_tracks']:
            ind = np.random.choice(np.arange(len(recommendations)), p=recommendation_probs)
        if recommendations_scores[ind] < self.min_good_predicted_score:
            return self._fallback(user_tracks)
        return int(recommendations[ind])

