#!/usr/bin/env python3
"""Core recommender module (CorrectAnimeRecommender).
Uses Pinecone for cloud-based vector search."""
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
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrectAnimeRecommender:
    """Semantically-correct hybrid recommender using Pinecone.
    Expects these files in `model_dir`:
    - anime_processed.csv
    - embeddings.npy (for local fallback)
    - encoders.pkl  (contains 'tfidf_vectorizer')
    - tfidf_matrix.npy"""

    def __init__(self, model_dir: str = "correct_model"):
        self.model_dir = model_dir
        self.sentence_model: Optional[SentenceTransformer] = None
        self.tfidf_vectorizer = None
        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.pc = None
        self.index = None

        # stats + cache
        self.stats = {"total_queries": 0, "avg_response_time": 0.0, "cache_hits": 0}
        self.query_cache = OrderedDict()
        self.MAX_CACHE = 1000

        # simple in-memory user history for personalization
        self.user_history: Dict[str, Dict] = {}

        self._load_artifacts()
        self._init_pinecone()

    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            logger.warning("PINECONE_API_KEY not found. Using local embeddings only.")
            return
        
        try:
            self.pc = Pinecone(api_key=api_key)
            index_name = os.getenv("PINECONE_INDEX_NAME", "anime-recommender")
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            self.index = None

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

        # parse list-like columns (theme_list doesn't exist in dataset)
        for col in ["genre_list", "mood_tags"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._safe_parse_list)

        # embeddings (for local fallback and personalization)
        emb_path = os.path.join(self.model_dir, "embeddings.npy")
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path).astype(np.float32)
            if self.df.shape[0] != self.embeddings.shape[0]:
                raise RuntimeError("Data / embeddings length mismatch - rebuild artifacts")

            # normalize
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            self.embeddings = self.embeddings / norms
        else:
            logger.warning("Local embeddings not found. Pinecone-only mode.")

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
        """Build query matching dataset embedding structure for better alignment"""
        q = query.lower().strip()
        
        # Extract genre hints from query
        genre_hints = []
        mood_hints = []
        
        if "demon" in q or "demons" in q:
            genre_hints.extend(["demons", "supernatural", "dark fantasy", "battle", "shounen"])
            mood_hints.append("dark")
        if "dark" in q or "psychological" in q:
            genre_hints.extend(["psychological", "thriller"])
            mood_hints.extend(["dark", "mature", "serious"])
        if "sad" in q or "emotional" in q:
            genre_hints.append("drama")
            mood_hints.extend(["emotional", "tragic", "heartbreaking"])
        if "romance" in q or "love" in q:
            genre_hints.extend(["romance", "shoujo"])
            mood_hints.extend(["romantic", "emotional"])
        if "action" in q or "fight" in q:
            genre_hints.extend(["action", "martial arts", "swordplay", "shounen"])
            mood_hints.append("intense")
        if "comedy" in q or "funny" in q:
            genre_hints.extend(["comedy", "slice of life"])
            mood_hints.extend(["funny", "lighthearted"])
        if "school" in q:
            genre_hints.extend(["school", "slice of life"])
        if "fantasy" in q:
            genre_hints.extend(["fantasy", "magic", "adventure"])
        if "historical" in q:
            genre_hints.extend(["historical", "period drama", "samurai"])
        
        # Match dataset embedding structure
        parts = [
            f"Title: anime about {query}",
            f"Synopsis: {query}",
        ]
        
        if genre_hints:
            parts.append(f"Genres: {' '.join(genre_hints)}")
        
        if mood_hints:
            parts.append(f"Mood: {' '.join(mood_hints)}")
        
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
    def _search_pinecone(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using Pinecone with quality filtering"""
        if self.index is None:
            logger.warning("Pinecone not available, using local fallback")
            return self._search_local(query_vector, k)
        
        try:
            # Use metadata filtering to remove low-quality anime
            # Safe cast to Python floats for precision
            results = self.index.query(
                vector=[float(x) for x in query_vector],
                top_k=k,
                include_metadata=True,
                filter={"rating": {"$gte": 3.5}}
            )
            
            if not results.matches:
                return np.array([]), np.array([])
            
            # Safe ID parsing - filter out invalid IDs
            valid_matches = [
                (match.score, int(match.id))
                for match in results.matches
                if match.id.isdigit() and int(match.id) < len(self.df)
            ]
            
            if not valid_matches:
                return np.array([]), np.array([])
            
            scores = np.array([score for score, _ in valid_matches])
            ids = np.array([idx for _, idx in valid_matches])
            
            return scores, ids
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return self._search_local(query_vector, k)
    
    def _search_local(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback local search using numpy"""
        if self.embeddings is None:
            return np.array([]), np.array([])
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_vector)
        top_k_idx = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_idx]
        
        return top_k_scores, top_k_idx

    def adaptive_tfidf_weight(self, query_text: str) -> float:
        t = len(query_text.split())
        if t <= 2:
            return 0.05
        elif t <= 4:
            return 0.10
        else:
            return 0.20

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
            
            # Strict weight normalization (guarantees sum = 1.0)
            base_sem_w = 0.85
            quality_w = 0.08
            scale = base_sem_w + tfidf_w + quality_w
            sem_w = base_sem_w / scale
            tfidf_w_norm = tfidf_w / scale
            quality_w_norm = quality_w / scale
            
            final = sem_w * float(sem_sim) + tfidf_w_norm * float(tfidf_sim) + quality_w_norm * quality

            # Multiplicative boosts (safe, maintains probability space)
            if detected_genres:
                anime_genres = [g.lower() for g in row.get("genre_list", [])]
                if any(g in anime_genres for g in detected_genres):
                    final *= 1.05
            if detected_moods:
                anime_moods = [m.lower() for m in row.get("mood_tags", [])]
                if any(m in anime_moods for m in detected_moods):
                    final *= 1.05
            if year_filter is not None:
                yr = int(pd.to_numeric(row.get("Release_year", 0), errors="coerce") or 0)
                if year_filter == 2005 and yr <= 2008:
                    final *= 1.05
                if year_filter == 2019 and yr >= 2018:
                    final *= 1.05

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
            title = str(self.df.iloc[idx].get("Name", "")).lower()
            root = re.split(r":|-", title)[0].strip()
            main_genre = (self.df.iloc[idx].get("genre_list") or ["Unknown"])[0]

            # Only remove low-quality duplicates
            if root in seen_titles and score < 0.75:
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
        if len(user_a) >= 2 and self.embeddings is not None:
            vecs = [self.embeddings[i] for i in user_a if i < len(self.embeddings)]
            if vecs:
                v = np.mean(vecs, axis=0)
                n = np.linalg.norm(v)
                if n > 0:
                    emb_vec = v / n

        for idx, score in reranked:
            boost = 1.0
            row_genres = [g.lower() for g in self.df.iloc[idx].get("genre_list", [])]
            if any(g in user_g for g in row_genres):
                boost *= 1.05
            if emb_vec is not None and idx < len(self.embeddings):
                sim = float(np.dot(emb_vec, self.embeddings[idx]))
                boost *= (1.0 + 0.03 * sim)

            new_score = float(np.clip(score * boost, 0.0, 1.0))
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
        
        # Hyper-personalization: adaptive blend based on user history size
        if user_id and user_id in self.user_history and self.embeddings is not None:
            liked = self.user_history[user_id]["anime"]
            if len(liked) >= 2:
                vecs = [self.embeddings[i] for i in liked if i < len(self.embeddings)]
                if vecs:
                    user_vec = np.mean(vecs, axis=0)
                    user_vec_norm = np.linalg.norm(user_vec)
                    if user_vec_norm > 0:
                        user_vec = user_vec / user_vec_norm
                        # Adaptive blending: stronger with more history
                        alpha = min(0.45, 0.15 + 0.05 * len(liked))
                        qvec = (1 - alpha) * qvec + alpha * user_vec
                        # Re-normalize
                        qvec = qvec / np.linalg.norm(qvec)

        candidates_k = max(250, k * 12)
        sims, idxs = self._search_pinecone(qvec, candidates_k)
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
                "description": str(row.get("Description", ""))[:350]
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
            "pinecone_status": "connected" if self.index is not None else "not_connected",
            "model_status": "loaded"
        }
