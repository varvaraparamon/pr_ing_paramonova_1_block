import pandas as pd
from data_manager import DataManager


class TopTracksRecommender:
    def __init__(self, data_manager: DataManager | None = None):
        self.dm = data_manager or DataManager()

    def top(self, n: int = 250) -> pd.DataFrame:
        plays = self.dm.song_plays()
        df = plays.sort_values("play_count", ascending=False).head(n)
        df = df[["artist", "title", "play_count"]].reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "index"
        return df
