from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
CACHE_DIR = DATASETS_DIR / "cache"

TRACKS_FILE = DATASETS_DIR / "p02_unique_tracks.txt"
TRIPLETS_FILE = DATASETS_DIR / "train_triplets.txt"
GENRES_FILE = DATASETS_DIR / "p02_msd_tagtraum_cd2.cls"
LYRICS_FILE = DATASETS_DIR / "mxm_dataset_train.txt"


class DataManager:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tracks = None
        self._triplets = None
        self._song_plays = None
        self._genres = None
        self._lyrics = None

    def _require(self, path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        return path

    def tracks(self) -> pd.DataFrame:
        if self._tracks is not None:
            return self._tracks
        cache = self.cache_dir / "tracks.pkl"
        if cache.exists():
            self._tracks = pd.read_pickle(cache)
            return self._tracks
        self._require(TRACKS_FILE)
        df = pd.read_csv(
            TRACKS_FILE,
            sep="<SEP>",
            engine="python",
            header=None,
            names=["track_id", "song_id", "artist", "title"],
        )
        df.to_pickle(cache)
        self._tracks = df
        return df

    def triplets(self) -> pd.DataFrame:
        if self._triplets is not None:
            return self._triplets
        cache = self.cache_dir / "triplets.pkl"
        if cache.exists():
            self._triplets = pd.read_pickle(cache)
            return self._triplets
        self._require(TRIPLETS_FILE)
        df = pd.read_csv(
            TRIPLETS_FILE,
            sep="\t",
            header=None,
            names=["user_id", "song_id", "play_count"],
            dtype={"user_id": "category", "song_id": "category"},
        )
        df["play_count"] = df["play_count"].astype(np.int32)
        df.to_pickle(cache)
        self._triplets = df
        return df

    def song_plays(self) -> pd.DataFrame:
        if self._song_plays is not None:
            return self._song_plays
        cache = self.cache_dir / "song_plays.pkl"
        if cache.exists():
            self._song_plays = pd.read_pickle(cache)
            return self._song_plays
        triplets = self.triplets()
        plays = (
            triplets.groupby("song_id", observed=True)["play_count"]
            .sum()
            .rename("play_count")
            .reset_index()
        )
        names = self.tracks().drop_duplicates("song_id")[["song_id", "artist", "title"]]
        df = plays.merge(names, on="song_id", how="left")
        df.to_pickle(cache)
        self._song_plays = df
        return df

    def genres(self) -> pd.DataFrame:
        if self._genres is not None:
            return self._genres
        cache = self.cache_dir / "genres.pkl"
        if cache.exists():
            self._genres = pd.read_pickle(cache)
            return self._genres
        self._require(GENRES_FILE)
        df = pd.read_csv(
            GENRES_FILE,
            sep="\t",
            comment="#",
            header=None,
            names=["track_id", "majority_genre", "minority_genre"],
        )
        df.to_pickle(cache)
        self._genres = df
        return df

    def lyrics(self):
        if self._lyrics is not None:
            return self._lyrics
        meta_cache = self.cache_dir / "lyrics_meta.pkl"
        matrix_cache = self.cache_dir / "lyrics_matrix.npz"
        if meta_cache.exists() and matrix_cache.exists():
            meta = pd.read_pickle(meta_cache)
            matrix = sparse.load_npz(matrix_cache)
            self._lyrics = (meta["vocab"], matrix, meta["track_ids"])
            return self._lyrics
        self._require(LYRICS_FILE)
        vocab = []
        track_ids = []
        rows, cols, data = [], [], []
        with open(LYRICS_FILE, encoding="utf-8") as fh:
            for row_idx, line in enumerate(fh):
                if line.startswith("#"):
                    continue
                if line.startswith("%"):
                    vocab = line[1:].strip().split(",")
                    continue
                parts = line.strip().split(",")
                track_ids.append(parts[0])
                for pair in parts[2:]:
                    word_idx, count = pair.split(":")
                    rows.append(len(track_ids) - 1)
                    cols.append(int(word_idx) - 1)
                    data.append(int(count))
        n_words = len(vocab)
        matrix = sparse.csr_matrix(
            (np.asarray(data, dtype=np.int32), (rows, cols)),
            shape=(len(track_ids), n_words),
        )
        pd.to_pickle({"vocab": vocab, "track_ids": np.asarray(track_ids)}, meta_cache)
        sparse.save_npz(matrix_cache, matrix)
        self._lyrics = (vocab, matrix, np.asarray(track_ids))
        return self._lyrics
