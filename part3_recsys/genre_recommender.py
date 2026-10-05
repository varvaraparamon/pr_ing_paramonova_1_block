import pandas as pd
from data_manager import DataManager


class GenreRecommender:
    def __init__(self, data_manager: DataManager | None = None):
        self.dm = data_manager or DataManager()

    def valid_genres(self) -> list[str]:
        return sorted(self.dm.genres()["majority_genre"].dropna().unique())

    def top_by_genre(self, genre: str, n: int = 100) -> pd.DataFrame:
        genres = self.dm.genres()
        mask = genres["majority_genre"].str.lower() == str(genre).lower()
        if not mask.any():
            valid = ", ".join(self.valid_genres())
            raise ValueError(f"Unknown genre '{genre}'. Valid genres: {valid}")
        track_ids = genres.loc[mask, "track_id"]
        tracks = self.dm.tracks()[["track_id", "song_id"]]
        song_ids = tracks.loc[
            tracks["track_id"].isin(set(track_ids)), "song_id"
        ].unique()
        plays = self.dm.song_plays()
        df = plays.loc[plays["song_id"].isin(set(song_ids))]
        df = df.sort_values("play_count", ascending=False).head(n)
        df = df[["artist", "title", "play_count"]].reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "index"
        return df
