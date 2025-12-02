#!/usr/bin/env python3
"""Core recommender module (CorrectAnimeRecommender).
Depends on artifacts saved by CorrectAnimeTrainer in `model_dir`."""
import os
import ast
import time
import logging
import re
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrectAnimeRecommender:
    """Semantically-correct hybrid recommender.
    Expects these files in `model_dir`:
    - anime_processed.csv
    - embeddings.npy
    - faiss_index.bin
    - encoders.pkl  (contains 'tfidf_vectorizer')
    - tfidf_matrix.npy"""

    def __init__(self, model_dir: str = "correct_model"):
        self.model_dir = model_dir
        self.sentence_model: Optional[SentenceTransformer] = None
        self.tfidf_vectorizer = None
        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.tfidf_matrix: Optional[np.ndarray] = None

        # stats + cache
        self.stats = {"total_queries": 0, "avg_response_time": 0.0, "cache_hits": 0}
        self.query_cache = OrderedDict()
        self.MAX_CACHE = 1000

        # simple in-memory user history for personalization
        self.user_history: Dict[str, Dict] = {}

        if os.path.exists(os.path.join(model_dir, "embeddings.npy")):
            self._load_artifacts()

    def _safe_parse_list(self, x):
        # robust parsing for lists from csv
        if x is None:
            return []
        try:
            if pd.isna(x):
                return []
        except Exception:
            pass
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return []
            try:
                if s.startswith("["):
                    return ast.literal_eval(s)
            except Exception:
                pass
            # fallback: comma-separated
            return [p.strip() for p in re.split(r",|;", s) if p.strip()]
        return []

    def _load_artifacts(self):
        logger.info("Loading model artifacts from %s", self.model_dir)
        # sentence model
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2")

        # dataframe
        df_path = os.path.join(self.model_dir, "anime_processed.csv")
        if not os.path.exists(df_path):
            raise FileNotFoundError(df_path)
        self.df = pd.read_csv(df_path)

        # parse list-like columns
        for col in ["genre_list", "theme_list", "mood_tags"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._safe_parse_list)

        # embeddings
        emb_path = os.path.join(self.model_dir, "embeddings.npy")
        self.embeddings = np.load(emb_path).astype(np.float32)
        if self.df.shape[0] != self.embeddings.shape[0]:
            raise RuntimeError("Data / embeddings length mismatch - rebuild artifacts")

        # normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        self.embeddings = self.embeddings / norms

        # faiss
        try:
            self.faiss_index = faiss.read_index(os.path.join(self.model_dir, "faiss_index.bin"))
            if self.faiss_index.metric_type != faiss.METRIC_INNER_PRODUCT:
                logger.warning("FAISS metric not IP; building fallback flat index")
                raise ValueError("wrong metric")
        except Exception:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)

        # tfidf
        enc_path = os.path.join(self.model_dir, "encoders.pkl")
        if not os.path.exists(enc_path):
            raise FileNotFoundError(enc_path)
        enc = joblib.load(enc_path)
        self.tfidf_vectorizer = enc.get("tfidf_vectorizer")
        self.tfidf_matrix = np.load(os.path.join(self.model_dir, "tfidf_matrix.npy"))

        # normalize tfidf rows
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.tfidf_matrix = self.tfidf_matrix / norms

        logger.info("Loaded %d anime records", len(self.df))

    # ---------------- Query building and expansion ----------------
    def build_semantic_query(self, query: str) -> str:
        q = query.lower().strip()
        parts = [f"anime about {query}"]
        if "dark" in q:
            parts.append("mood dark psychological mature serious")
        if "sad" in q:
            parts.append("mood emotional tragic touching tearjerker")
        if "romance" in q or "love" in q:
            parts.append("genre romance love relationship romantic")
        if "action" in q:
            parts.append("genre action battle fight intense")
        return " | ".join(parts)

    def enhance_query(self, query_text: str) -> Tuple[str, List[str], List[str], Optional[int]]:
        query_lower = query_text.lower()
        detected_genres = []
        detected_moods = []

        # small keyword maps
        if any(w in query_lower for w in ["action", "fight", "battle"]):
            detected_genres.append("action")
        if any(w in query_lower for w in ["romance", "love", "relationship"]):
            detected_genres.append("romance")
        if any(w in query_lower for w in ["dark", "psychological", "mature"]):
            detected_moods.append("dark")

        year_filter = None
        if any(w in query_lower for w in ["old", "classic"]):
            year_filter = 2005
        elif any(w in query_lower for w in ["new", "latest"]):
            year_filter = 2019

        enhanced = query_text
        # small expansion
        if detected_genres:
            enhanced += " " + " ".join(detected_genres)
        if detected_moods:
            enhanced += " " + " ".join(detected_moods)

        return enhanced, detected_genres, detected_moods, year_filter

    # ---------------- Search & re-rank ----------------
    def _search_faiss(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.faiss_index is None or getattr(self.faiss_index, "ntotal", 0) == 0:
            return np.array([]), np.array([])
        q = query_vector.reshape(1, -1).astype(np.float32)
        k = min(k, self.faiss_index.ntotal)
        sims, idxs = self.faiss_index.search(q, k)
        sims = sims[0]
        idxs = idxs[0]
        valid = idxs != -1
        return sims[valid], idxs[valid]

    def adaptive_tfidf_weight(self, query_text: str) -> float:
        t = len(query_text.split())
        if t < 3:
            return 0.0
        if t >= 4:
            return 0.15
        return 0.10

    def rerank(self, query_text: str, candidates: List[int], similarities: np.ndarray,
               detected_genres: List[str], detected_moods: List[str], year_filter: Optional[int]) -> List[Tuple[int, float]]:
        tfidf_w = self.adaptive_tfidf_weight(query_text)
        query_tfidf = self.tfidf_vectorizer.transform([query_text]).toarray()[0]
        qn = np.linalg.norm(query_tfidf)
        if qn > 0:
            query_tfidf = query_tfidf / qn

        out = []
        for idx, sem_sim in zip(candidates, similarities):
            if idx < 0 or idx >= len(self.df):
                continue
            row = self.df.iloc[idx]

            if tfidf_w == 0:
                tfidf_sim = 0.0
            else:
                tfidf_sim = float(np.dot(query_tfidf, self.tfidf_matrix[idx]))

            quality = float(row.get("quality_score", 0.5))
            sem_w = 0.8 if tfidf_w == 0 else 0.78
            quality_w = 0.2
            final = sem_w * float(sem_sim) + tfidf_w * float(tfidf_sim) + quality_w * quality

            # soft boosts
            if detected_genres:
                anime_genres = [g.lower() for g in row.get("genre_list", [])]
                if any(g in anime_genres for g in detected_genres):
                    final += 0.05
            if detected_moods:
                anime_moods = [m.lower() for m in row.get("mood_tags", [])]
                if any(m in anime_moods for m in detected_moods):
                    final += 0.05
            if year_filter is not None:
                yr = int(row.get("Release_year", 0))
                if year_filter == 2005 and yr <= 2005:
                    final += 0.05
                if year_filter == 2019 and yr >= 2019:
                    final += 0.05

            final = float(np.clip(final, 0.0, 1.0))
            out.append((idx, final))

        out.sort(key=lambda x: x[1], reverse=True)
        return out

    # ------------- Diversity & Personalization -------------
    def add_diversity(self, reranked: List[Tuple[int, float]], k: int, detected_genres: Optional[List[str]] = None):
        seen_titles = set()
        seen_genres = set()
        diverse = []

        for idx, score in reranked:
            title = str(self.df.iloc[idx].get("title", "")).lower()
            root = re.split(r":|-", title)[0].strip()
            main_genre = (self.df.iloc[idx].get("genre_list") or ["Unknown"])[0]

            if root in seen_titles:
                continue
            if detected_genres:
                # allow same genre if user requested
                pass
            else:
                if main_genre in seen_genres:
                    continue

            diverse.append((idx, score))
            seen_titles.add(root)
            seen_genres.add(main_genre)
            if len(diverse) >= k:
                break

        return diverse

    def record_user_feedback(self, user_id: str, anime_id: int):
        if user_id not in self.user_history:
            self.user_history[user_id] = {"anime": set(), "genres": set()}
        if 0 <= anime_id < len(self.df):
            self.user_history[user_id]["anime"].add(anime_id)
            for g in self.df.iloc[anime_id].get("genre_list", []):
                self.user_history[user_id]["genres"].add(g.lower())

    def personalize(self, reranked: List[Tuple[int, float]], user_id: Optional[str]):
        if not user_id or user_id not in self.user_history:
            return reranked

        user_g = self.user_history[user_id].get("genres", set())
        user_a = self.user_history[user_id].get("anime", set())
        personalized = []

        # compute mean embedding for clicked items if available
        emb_vec = None
        if len(user_a) >= 2:
            vecs = [self.embeddings[i] for i in user_a if i < len(self.embeddings)]
            if vecs:
                v = np.mean(vecs, axis=0)
                n = np.linalg.norm(v)
                if n > 0:
                    emb_vec = v / n

        for idx, score in reranked:
            add = 0.0
            row_genres = [g.lower() for g in self.df.iloc[idx].get("genre_list", [])]
            if any(g in user_g for g in row_genres):
                add += 0.05
            if emb_vec is not None and idx < len(self.embeddings):
                add += 0.03 * float(np.dot(emb_vec, self.embeddings[idx]))

            new_score = float(np.clip(score + add, 0.0, 1.0))
            personalized.append((idx, new_score))

        personalized.sort(key=lambda x: x[1], reverse=True)
        return personalized

    # ---------------- Public recommend API ----------------
    def recommend(self, query: str, k: int = 10, user_id: Optional[str] = None) -> List[Dict]:
        start = time.time()
        if self.sentence_model is None:
            raise RuntimeError("Sentence model not loaded")

        semantic_query = self.build_semantic_query(query)
        enhanced_query, detected_genres, detected_moods, year_f = self.enhance_query(query)

        # cache key canonicalization
        cache_key = " ".join(semantic_query.lower().split())
        if cache_key in self.query_cache:
            qvec = self.query_cache[cache_key]
            self.query_cache.move_to_end(cache_key)
            self.stats["cache_hits"] += 1
        else:
            qvec = self.sentence_model.encode([semantic_query], convert_to_numpy=True, normalize_embeddings=True)[0]
            self.query_cache[cache_key] = qvec
            if len(self.query_cache) > self.MAX_CACHE:
                self.query_cache.popitem(last=False)

        candidates_k = max(200, k * 10)
        sims, idxs = self._search_faiss(qvec, candidates_k)
        if sims.size == 0:
            return []

        reranked = self.rerank(enhanced_query, idxs.tolist(), sims, detected_genres, detected_moods, year_f)
        if user_id:
            reranked = self.personalize(reranked, user_id)
        reranked = self.add_diversity(reranked, k * 2, detected_genres)

        recs = []
        for idx, score in reranked[:k]:
            row = self.df.iloc[idx]
            recs.append({
                "title": str(row.get("Name", "")),
                "rating": round(float(row.get("Rating", 0.0)), 1),
                "genres": list(row.get("genre_list") or [])[:5],
                "final_score": round(float(score), 3),
                "rank": int(row.get("Rank", 0)),
                "year": int(row.get("Release_year", 0)),
                "type": str(row.get("Type", "")),
                "studio": str(row.get("Studio", "")),
                "mood_tags": list(row.get("mood_tags") or []),
                "description": str(row.get("Description", ""))[:200]
            })

        # stats
        elapsed = time.time() - start
        self.stats["total_queries"] += 1
        self.stats["avg_response_time"] = (
            (self.stats["avg_response_time"] * (self.stats["total_queries"] - 1)) + elapsed
        ) / self.stats["total_queries"]

        return recs

    def get_stats(self):
        hit_rate = 0.0
        if self.stats["total_queries"] > 0:
            hit_rate = self.stats["cache_hits"] / self.stats["total_queries"]
        return {
            "total_anime": len(self.df) if self.df is not None else 0,
            "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "total_queries": self.stats["total_queries"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": round(hit_rate, 3),
            "avg_response_time": round(self.stats["avg_response_time"], 3),
            "model_status": "loaded" if self.faiss_index is not None else "not_loaded"
        }
