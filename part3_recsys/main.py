import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr] 

from data_manager import DataManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myspotify", description="MySpotify recommender systems"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_top = sub.add_parser("top", help="Top-n most played tracks")
    p_top.add_argument("--n", type=int, default=250)

    p_genre = sub.add_parser("genre", help="Top-n tracks of a genre")
    p_genre.add_argument("genre")
    p_genre.add_argument("--n", type=int, default=100)

    p_col = sub.add_parser("collection", help="Top-50 tracks for a keyword")
    p_col.add_argument("keyword")
    p_col.add_argument("--method", default="baseline")

    p_user = sub.add_parser("user", help="User-based CF recommendations")
    p_user.add_argument("user_id")

    p_track = sub.add_parser("track", help="Item-based similar tracks")
    p_track.add_argument("song_id")

    p_hybrid = sub.add_parser("hybrid", help="Hybrid: user-CF + popularity")
    p_hybrid.add_argument("user_id")
    p_hybrid.add_argument("--w", type=float, default=0.7, help="вес CF (0..1)")

    p_eval = sub.add_parser("evaluate", help="Evaluate recommenders (p@10)")
    p_eval.add_argument("--users", type=int, default=300)
    p_eval.add_argument("--tracks", type=int, default=300)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dm = DataManager()
        if args.command == "top":
            from top_tracks import TopTracksRecommender

            print(TopTracksRecommender(dm).top(args.n).to_string())
        elif args.command == "genre":
            from genre_recommender import GenreRecommender

            print(GenreRecommender(dm).top_by_genre(args.genre, args.n).to_string())
        elif args.command == "collection":
            from collections_recommender import CollectionsRecommender

            rec = CollectionsRecommender(dm)
            print(rec.top_by_keyword(args.keyword, method=args.method).to_string())
        elif args.command == "user":
            from collaborative import CollaborativeData, UserBasedCF

            cf = UserBasedCF(CollaborativeData(dm))
            print(cf.recommend(args.user_id).to_string())
        elif args.command == "track":
            from collaborative import CollaborativeData, ItemBasedCF

            item = ItemBasedCF(CollaborativeData(dm))
            print(item.recommend(args.song_id).to_string())
        elif args.command == "hybrid":
            from collaborative import CollaborativeData
            from hybrid import HybridRecommender

            hyb = HybridRecommender(CollaborativeData(dm), w_cf=args.w)
            print(hyb.recommend(args.user_id).to_string())
        elif args.command == "evaluate":
            from collaborative import CollaborativeData, ItemBasedCF, UserBasedCF
            from hybrid import HybridRecommender

            data = CollaborativeData(dm)
            user_cf = UserBasedCF(data)
            p_user = user_cf.evaluate(n_users=args.users)
            print(f"User-based CF  average p@10 ({args.users} users):  {p_user:.4f}")
            item_cf = ItemBasedCF(data)
            p_item = item_cf.evaluate(n_cases=args.tracks)
            print(f"Item-based CF  average p@10 ({args.tracks} cases): {p_item:.4f}")
            hybrid = HybridRecommender(data)
            p_hybrid = hybrid.evaluate(n_users=args.users)
            print(f"Hybrid         average p@10 ({args.users} users):  {p_hybrid:.4f}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
