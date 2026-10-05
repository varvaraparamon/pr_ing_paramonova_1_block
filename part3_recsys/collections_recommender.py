import difflib

import numpy as np
import pandas as pd
from data_manager import CACHE_DIR, DataManager

KEYWORDS = ["love", "war", "happiness", "loneliness", "money"]
METHODS = ["baseline", "word2vec", "classification"]
W2V_MODEL_PATH = CACHE_DIR / "word2vec.model"
COUNT_THRESHOLD = 2
MAX_REPEAT = 20


class CollectionsRecommender:
    def __init__(self, data_manager: DataManager | None = None):
        self.dm = data_manager or DataManager()
        self._vocab = None
        self._matrix = None
        self._track_ids = None
        self._catalog = None

    def _load(self):
        if self._vocab is None:
            self._vocab, self._matrix, self._track_ids = self.dm.lyrics()

    def _catalog_df(self) -> pd.DataFrame:
        self._load()
        if self._catalog is None:
            tracks = self.dm.tracks()[["track_id", "song_id"]]
            plays = self.dm.song_plays()
            df = pd.DataFrame({"track_id": self._track_ids})
            df = df.merge(tracks, on="track_id", how="left")
            df = df.merge(plays, on="song_id", how="left")
            df["play_count"] = df["play_count"].fillna(0).astype(np.int64)
            self._catalog = df
        return self._catalog

    def _resolve_keyword(self, keyword: str) -> list[int]:
        self._load()
        keyword = str(keyword).lower()
        if keyword not in KEYWORDS:
            raise ValueError(
                f"Unknown keyword '{keyword}'. Valid keywords: {', '.join(KEYWORDS)}"
            )
        vocab = self._vocab
        if keyword in vocab:
            return [vocab.index(keyword)]
        prefixed = [w for w in vocab if w.startswith(keyword)]
        if prefixed:
            return [vocab.index(w) for w in prefixed]
        substr = [w for w in vocab if keyword in w]
        if substr:
            return [vocab.index(w) for w in substr]
        close = difflib.get_close_matches(keyword, vocab, n=3, cutoff=0.6)
        if close:
            return [vocab.index(w) for w in close]
        raise ValueError(
            f"Could not resolve keyword '{keyword}' to any vocabulary word"
        )

    def _counts(self, word_indices: list[int]) -> np.ndarray:
        self._load()
        sub = self._matrix[:, word_indices]
        return np.asarray(sub.sum(axis=1)).ravel()

    def _format(self, scores: np.ndarray, n: int = 50) -> pd.DataFrame:
        catalog = self._catalog_df()
        order = np.argsort(-scores, kind="stable")
        picked = order[scores[order] > 0][:n]
        df = catalog.iloc[picked][["artist", "title", "play_count"]].copy()
        df = df.sort_values("play_count", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "index"
        return df

    def _baseline(self, keyword: str, n: int) -> pd.DataFrame:
        counts = self._counts(self._resolve_keyword(keyword))
        catalog = self._catalog_df()
        scores = np.where(
            counts >= COUNT_THRESHOLD, catalog["play_count"].to_numpy(), 0
        )
        return self._format(scores, n)

    def _build_documents(self) -> list[list[str]]:
        self._load()
        coo = self._matrix.tocoo()
        docs: list[list[str]] = [[] for _ in range(self._matrix.shape[0])]
        for r, c, v in zip(coo.row, coo.col, coo.data):
            docs[r].extend([self._vocab[c]] * min(int(v), MAX_REPEAT))
        return docs

    def _word2vec_model(self):
        from gensim.models import Word2Vec

        if W2V_MODEL_PATH.exists():
            return Word2Vec.load(str(W2V_MODEL_PATH))
        docs = self._build_documents()
        model = Word2Vec(
            sentences=docs,
            vector_size=100,
            window=5,
            min_count=5,
            workers=4,
        )
        model.save(str(W2V_MODEL_PATH))
        return model

    def _word2vec(self, keyword: str, n: int) -> pd.DataFrame:
        indices = self._resolve_keyword(keyword)
        words = [self._vocab[i] for i in indices]
        model = self._word2vec_model()
        similar: list[str] = []
        for word in words:
            if word in model.wv:
                similar.extend(w for w, _ in model.wv.most_similar(word, topn=5))
        all_words = set(words) | set(similar)
        extra_idx = [self._vocab.index(w) for w in all_words if w in self._vocab]
        counts = self._counts(sorted(set(indices) | set(extra_idx)))
        catalog = self._catalog_df()
        scores = np.where(
            counts >= COUNT_THRESHOLD, catalog["play_count"].to_numpy(), 0
        )
        return self._format(scores, n)

    def _classification(self, keyword: str, n: int) -> pd.DataFrame:
        from sklearn.linear_model import LogisticRegression

        self._load()
        counts = self._counts(self._resolve_keyword(keyword))
        labels = (counts >= COUNT_THRESHOLD).astype(np.int8)
        pos = np.flatnonzero(labels)
        neg = np.flatnonzero(labels == 0)
        if len(pos) == 0:
            raise ValueError(f"No positive tracks found for '{keyword}'")
        rng = np.random.default_rng(42)
        pos_sample = rng.choice(pos, size=min(len(pos), 20000), replace=False)
        neg_sample = rng.choice(neg, size=len(pos_sample), replace=False)
        train_idx = np.concatenate([pos_sample, neg_sample])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(self._matrix[train_idx], labels[train_idx])
        proba = clf.predict_proba(self._matrix)[:, 1]
        return self._format(proba, n)

    def top_by_keyword(
        self, keyword: str, method: str = "baseline", n: int = 50
    ) -> pd.DataFrame:
        if method not in METHODS:
            raise ValueError(
                f"Unknown method '{method}'. Valid methods: {', '.join(METHODS)}"
            )
        if method == "baseline":
            return self._baseline(keyword, n)
        if method == "word2vec":
            return self._word2vec(keyword, n)
        return self._classification(keyword, n)
