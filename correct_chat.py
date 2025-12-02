#!/usr/bin/env python3
"""Simple command-line chat interface to test the recommender."""
import sys
import time
from correct_recommender import CorrectAnimeRecommender


def main():
    print("Loading recommender...")
    r = CorrectAnimeRecommender()
    print("Loaded. Type queries (type 'quit' to exit, 'stats' for stats)")

    while True:
        try:
            q = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break

        if not q:
            continue

        if q.lower() in ("quit", "exit"):
            break

        if q.lower() in ("stats",):
            print(r.get_stats())
            continue

        start = time.time()
        recs = r.recommend(q, k=5)
        elapsed = time.time() - start

        print(f"Results (took {elapsed:.2f}s):")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. {rec['title']} ({rec['year']}) — score: {rec['final_score']:.3f} — Rating: {rec['rating']}/5.0 — Rank: #{rec['rank']}")
            print(f"   Type: {rec['type']} | Studio: {rec['studio']} | Genres: {', '.join(rec['genres'][:3])}")

        if not recs:
            print("No results — check artifacts or try different query")


if __name__ == "__main__":
    main()
