#!/usr/bin/env python3
"""Trainer: builds semantic_text, embeddings, TF-IDF, FAISS index and saves artifacts."""
import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrectAnimeTrainer:
    def __init__(self, model_dir: str = "correct_model"):
        self.model_dir = model_dir
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2")
        self.df = None

    def parse_categories(self, s):
        if pd.isna(s) or not s:
            return []
        return [p.strip() for p in str(s).split(",") if p.strip()]

    def extract_mood(self, text: str):
        if not isinstance(text, str) or not text:
            return []
        t = text.lower()
        moods = []
        if any(w in t for w in ["dark", "psychological", "bleak", "sinister"]):
            moods.append("dark")
        if any(w in t for w in ["romance", "love", "heart"]):
            moods.append("romantic")
        if any(w in t for w in ["funny", "hilarious", "comedy"]):
            moods.append("comedic")
        return moods

    def semantic_text_for_row(self, row) -> str:
        parts = []
        title = str(row.get("Name", "")).strip()
        desc = str(row.get("Description", "")).strip()

        parts.append(f"Title: {title}")
        if desc:
            parts.append(f"Synopsis: {desc[:300]}")

        genres = self.parse_categories(row.get("Tags", ""))
        if genres:
            parts.append("Genres: " + " ".join(genres))

        anime_type = str(row.get("Type", "")).strip()
        if anime_type:
            parts.append(f"Type: {anime_type}")

        studio = str(row.get("Studio", "")).strip()
        if studio:
            parts.append(f"Studio: {studio}")

        moods = self.extract_mood(desc)
        if moods:
            parts.append("Mood: " + " ".join(moods))

        rating = pd.to_numeric(row.get("Rating", None), errors="coerce")
        if pd.notna(rating) and rating >= 4.0:
            parts.append("Quality: highly rated excellent")
        elif pd.notna(rating) and rating >= 3.5:
            parts.append("Quality: good")

        return " | ".join(parts)

    def preprocess(self, data_path: str = "Anime.csv") -> pd.DataFrame:
        logger.info("Reading dataset: %s", data_path)
        df = pd.read_csv(data_path)

        # Map to expected column names
        df["Rating"] = pd.to_numeric(df.get("Rating"), errors="coerce")
        df["Rank"] = pd.to_numeric(df.get("Rank"), errors="coerce")
        df["Release_year"] = pd.to_numeric(df.get("Release_year"), errors="coerce")

        # Filter valid records
        mask = (
            df["Rating"].notna() &
            df.get("Description").apply(lambda x: isinstance(x, str) and len(x) > 30) &
            df.get("Tags").notna() &
            df.get("Name").notna()
        )
        df = df[mask].reset_index(drop=True)

        df["semantic_text"] = df.apply(self.semantic_text_for_row, axis=1)
        df["genre_list"] = df["Tags"].apply(self.parse_categories)
        df["mood_tags"] = df["Description"].apply(self.extract_mood)

        # quality_score engineering (using Rank and Rating)
        df["log_rank"] = np.log1p(df["Rank"].fillna(1000))
        # Invert rank (lower rank = better)
        max_log_rank = df["log_rank"].max()
        df["quality_score"] = (
            0.6 * (df["Rating"].fillna(3.5) / 5.0) +
            0.4 * (1 - (df["log_rank"] / max_log_rank))
        )

        self.df = df
        return df

    def create_embeddings(self) -> np.ndarray:
        texts = self.df["semantic_text"].tolist()
        emb = self.sentence_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
        )
        return emb.astype(np.float32)

    def build_faiss(self, embeddings: np.ndarray):
        d = embeddings.shape[1]
        try:
            index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.add(embeddings)
        except Exception:
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
        return index

    def create_tfidf(self) -> Tuple[object, np.ndarray]:
        tf = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2), min_df=2
        )
        mat = tf.fit_transform(self.df["semantic_text"]).toarray()

        # normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms

        return tf, mat

    def save_artifacts(self, embeddings, faiss_index, tfidf_vectorizer, tfidf_matrix):
        os.makedirs(self.model_dir, exist_ok=True)

        self.df.to_csv(os.path.join(self.model_dir, "anime_processed.csv"), index=False)
        np.save(os.path.join(self.model_dir, "embeddings.npy"), embeddings)
        faiss.write_index(faiss_index, os.path.join(self.model_dir, "faiss_index.bin"))
        np.save(os.path.join(self.model_dir, "tfidf_matrix.npy"), tfidf_matrix)
        joblib.dump(
            {"tfidf_vectorizer": tfidf_vectorizer},
            os.path.join(self.model_dir, "encoders.pkl")
        )

        logger.info("Saved artifacts to %s", self.model_dir)

    def train(self, data_path: str = "Anime.csv"):
        self.preprocess(data_path)
        embeddings = self.create_embeddings()
        index = self.build_faiss(embeddings)
        tfidf_vectorizer, tfidf_matrix = self.create_tfidf()
        self.save_artifacts(embeddings, index, tfidf_vectorizer, tfidf_matrix)
        logger.info("Training completed")


if __name__ == "__main__":
    trainer = CorrectAnimeTrainer()
    trainer.train()
