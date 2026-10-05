import numpy as np
import pandas as pd
from data_manager import CACHE_DIR, DataManager
from scipy import sparse
from sklearn.preprocessing import normalize

META_PATH = CACHE_DIR / "cf_meta.pkl"
TRAIN_PATH = CACHE_DIR / "cf_train.npz"
TEST_PATH = CACHE_DIR / "cf_test.npz"

N_TEST_USERS = 2000
MIN_USER_SONGS = 10
MIN_SONG_LISTENERS = 100
TEST_SONG_FRACTION = 0.15
SEED = 42


class CollaborativeData:
    def __init__(self, data_manager: DataManager | None = None):
        self.dm = data_manager or DataManager()
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        if META_PATH.exists() and TRAIN_PATH.exists() and TEST_PATH.exists():
            meta = pd.read_pickle(META_PATH)
            self.user_ids = meta["user_ids"]
            self.song_ids = meta["song_ids"]
            self.test_user_codes = meta["test_user_codes"]
            self.test_song_codes = meta["test_song_codes"]
            self.train = sparse.load_npz(TRAIN_PATH).tocsr()
            self.test = sparse.load_npz(TEST_PATH).tocsr()
            self._loaded = True
            return self
        self._build()
        return self

    def _build(self):
        triplets = self.dm.triplets()
        user_codes = triplets["user_id"].cat.codes.to_numpy()
        song_codes = triplets["song_id"].cat.codes.to_numpy()
        plays = triplets["play_count"].to_numpy(dtype=np.float32)
        self.user_ids = triplets["user_id"].cat.categories.to_numpy()
        self.song_ids = triplets["song_id"].cat.categories.to_numpy()
        n_users = len(self.user_ids)
        n_songs = len(self.song_ids)

        rng = np.random.default_rng(SEED)
        songs_per_user = np.bincount(user_codes, minlength=n_users)
        active = np.flatnonzero(songs_per_user >= MIN_USER_SONGS)
        self.test_user_codes = np.sort(
            rng.choice(active, size=min(N_TEST_USERS, len(active)), replace=False)
        )
        listeners_per_song = np.bincount(song_codes, minlength=n_songs)
        popular = np.flatnonzero(listeners_per_song >= MIN_SONG_LISTENERS)
        n_test_songs = int(len(popular) * TEST_SONG_FRACTION)
        weights = listeners_per_song[popular].astype(np.float64)
        weights /= weights.sum()
        self.test_song_codes = np.sort(
            rng.choice(popular, size=n_test_songs, replace=False, p=weights)
        )

        test_user_mask = np.zeros(n_users, dtype=bool)
        test_user_mask[self.test_user_codes] = True
        test_song_mask = np.zeros(n_songs, dtype=bool)
        test_song_mask[self.test_song_codes] = True
        is_test = test_user_mask[user_codes] & test_song_mask[song_codes]

        values = np.log1p(plays).astype(np.float32)
        shape = (n_users, n_songs)
        self.train = sparse.csr_matrix(
            (values[~is_test], (user_codes[~is_test], song_codes[~is_test])),
            shape=shape,
        )
        self.test = sparse.csr_matrix(
            (values[is_test], (user_codes[is_test], song_codes[is_test])),
            shape=shape,
        )
        self.train.sum_duplicates()
        self.test.sum_duplicates()

        pd.to_pickle(
            {
                "user_ids": self.user_ids,
                "song_ids": self.song_ids,
                "test_user_codes": self.test_user_codes,
                "test_song_codes": self.test_song_codes,
            },
            META_PATH,
        )
        sparse.save_npz(TRAIN_PATH, self.train)
        sparse.save_npz(TEST_PATH, self.test)
        self._loaded = True

    def user_code(self, user_id: str) -> int:
        matches = np.flatnonzero(self.user_ids == user_id)
        if len(matches) == 0:
            raise ValueError(f"Unknown user_id '{user_id}'")
        return int(matches[0])

    def song_code(self, song_id: str) -> int:
        matches = np.flatnonzero(self.song_ids == song_id)
        if len(matches) == 0:
            raise ValueError(f"Unknown song_id '{song_id}'")
        return int(matches[0])

    def names(self) -> pd.DataFrame:
        plays = self.dm.song_plays()
        return plays.set_index("song_id")[["artist", "title"]]

    def format_recs(self, song_codes: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
        names = self.names()
        ids = self.song_ids[song_codes]
        df = names.reindex(ids).reset_index(drop=True)
        df["score"] = scores
        df = df.sort_values("score", ascending=False)[["artist", "title"]]
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "index"
        return df


class UserBasedCF:
    def __init__(self, data: CollaborativeData, k_neighbors: int = 50):
        self.data = data.load()
        self.k = k_neighbors
        self._norm = None
        self._norm_t = None

    @property
    def norm(self):
        if self._norm is None:
            self._norm = normalize(self.data.train, norm="l2", axis=1)
        return self._norm

    @property
    def norm_t(self):
        if self._norm_t is None:
            self._norm_t = self.norm.T.tocsr()
        return self._norm_t

    def _recommend_codes(self, uidx: int, n: int = 10):
        train = self.data.train
        sims = (self.norm[uidx] @ self.norm_t).toarray().ravel()
        sims[uidx] = 0.0
        k = min(self.k, max(int((sims > 0).sum()), 1))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[sims[top] > 0]
        if len(top) == 0:
            return np.array([], dtype=int), np.array([])
        scores = np.asarray(sims[top] @ train[top]).ravel()
        seen = train[uidx].indices
        scores[seen] = 0.0
        positive = np.flatnonzero(scores > 0)
        if len(positive) == 0:
            return np.array([], dtype=int), np.array([])
        order = positive[np.argsort(-scores[positive], kind="stable")][:n]
        return order, scores[order]

    def recommend(self, user_id: str, n: int = 10) -> pd.DataFrame:
        uidx = self.data.user_code(user_id)
        codes, scores = self._recommend_codes(uidx, n)
        return self.data.format_recs(codes, scores)

    def evaluate(self, n_users: int = 300, seed: int = 7) -> float:
        test = self.data.test
        nnz = np.diff(test.indptr)
        eligible = self.data.test_user_codes[nnz[self.data.test_user_codes] > 0]
        rng = np.random.default_rng(seed)
        sample = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
        precisions = []
        for uidx in sample:
            codes, _ = self._recommend_codes(int(uidx), 10)
            if len(codes) == 0:
                precisions.append(0.0)
                continue
            truth = set(test[uidx].indices.tolist())
            hits = len(set(codes.tolist()) & truth)
            precisions.append(hits / 10)
        return float(np.mean(precisions))


class ItemBasedCF:
    def __init__(self, data: CollaborativeData):
        self.data = data.load()
        self._norm = None
        self._norm_csc = None

    @property
    def norm(self):
        if self._norm is None:
            self._norm = normalize(self.data.train, norm="l2", axis=0)
        return self._norm

    @property
    def norm_csc(self):
        if self._norm_csc is None:
            self._norm_csc = self.norm.tocsc()
        return self._norm_csc

    def _similar_codes(self, sidx: int, n: int = 10):
        column = self.norm_csc[:, sidx].toarray().ravel()
        sims = np.asarray(column @ self.norm).ravel()
        sims[sidx] = 0.0
        positive = np.flatnonzero(sims > 0)
        if len(positive) == 0:
            return np.array([], dtype=int), np.array([])
        order = positive[np.argsort(-sims[positive], kind="stable")][:n]
        return order, sims[order]

    def recommend(self, song_id: str, n: int = 10) -> pd.DataFrame:
        sidx = self.data.song_code(song_id)
        codes, scores = self._similar_codes(sidx, n)
        return self.data.format_recs(codes, scores)

    def evaluate(self, n_cases: int = 300, seed: int = 7) -> float:
        test = self.data.test.tocoo()
        user_test: dict[int, list[int]] = {}
        for u, s in zip(test.row, test.col):
            user_test.setdefault(int(u), []).append(int(s))
        cases = [
            (u, s) for u, songs in user_test.items() if len(songs) >= 2 for s in songs
        ]
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(cases), size=min(n_cases, len(cases)), replace=False)
        precisions = []
        for i in picks:
            u, s = cases[int(i)]
            truth = set(user_test[u]) - {s}
            codes, _ = self._similar_codes(s, 10)
            if len(codes) == 0:
                precisions.append(0.0)
                continue
            hits = len(set(codes.tolist()) & truth)
            precisions.append(hits / 10)
        return float(np.mean(precisions))
