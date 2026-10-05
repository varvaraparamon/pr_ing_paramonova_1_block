import numpy as np
import pandas as pd
from collaborative import CollaborativeData, UserBasedCF

DEFAULT_W_CF = 0.7


class HybridRecommender:
    def __init__(self, data: CollaborativeData, w_cf: float = DEFAULT_W_CF):
        self.data = data.load()
        self.cf = UserBasedCF(self.data)
        self.w_cf = w_cf
        self._popularity: np.ndarray | None = None

    @property
    def popularity(self) -> np.ndarray:
        if self._popularity is None:
            plays = self.data.dm.song_plays()
            code_by_sid = {sid: i for i, sid in enumerate(self.data.song_ids)}
            known = plays["song_id"].isin(code_by_sid)
            pop: np.ndarray = np.zeros(len(self.data.song_ids), dtype=np.float64)
            idx = [code_by_sid[s] for s in plays.loc[known, "song_id"]]
            pop[idx] = np.log1p(plays.loc[known, "play_count"].to_numpy())
            if pop.max() > 0:
                pop /= pop.max()
            self._popularity = pop
        assert self._popularity is not None
        return self._popularity

    def _popular_codes(self, n: int):
        pop = self.popularity
        order = np.argsort(-pop, kind="stable")[:n]
        return order, pop[order]

    def _recommend_codes(self, uidx: int, n: int):
        cf_codes, cf_scores = self.cf._recommend_codes(uidx, n * 3)
        if len(cf_codes) == 0:
            return np.array([], dtype=int), np.array([])
        cf_norm = cf_scores / cf_scores.max()
        blended = self.w_cf * cf_norm + (1 - self.w_cf) * self.popularity[cf_codes]
        order = np.argsort(-blended, kind="stable")[:n]
        return cf_codes[order], blended[order]

    def recommend(self, user_id: str, n: int = 10) -> pd.DataFrame:
        try:
            uidx = self.data.user_code(user_id)
        except ValueError:
            codes, scores = self._popular_codes(n)
            return self.data.format_recs(codes, scores)

        codes, scores = self._recommend_codes(uidx, n)
        if len(codes) == 0:
            codes, scores = self._popular_codes(n)
        return self.data.format_recs(codes, scores)

    def evaluate(self, n_users: int = 300, seed: int = 7) -> float:
        test = self.data.test
        nnz = np.diff(test.indptr)
        eligible = self.data.test_user_codes[nnz[self.data.test_user_codes] > 0]
        rng = np.random.default_rng(seed)
        sample = rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)
        precisions = []
        for uidx in sample:
            uidx = int(uidx)
            codes, _ = self._recommend_codes(uidx, 10)
            if len(codes) == 0:
                precisions.append(0.0)
                continue
            truth = set(test[uidx].indices.tolist())
            precisions.append(len(set(codes.tolist()) & truth) / 10)
        return float(np.mean(precisions))
