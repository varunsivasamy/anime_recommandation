#!/usr/bin/env python3
"""Trainer: builds semantic_text, embeddings, TF-IDF, and uploads to Pinecone."""
import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrectAnimeTrainer:
    def __init__(self, model_dir: str = "correct_model"):
        self.model_dir = model_dir
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2")
        self.df = None
        self.pc = None
        self.index = None
        self._init_pinecone()

    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            logger.warning("PINECONE_API_KEY not found. Pinecone features disabled.")
            return
        
        self.pc = Pinecone(api_key=api_key)
        index_name = os.getenv("PINECONE_INDEX_NAME", "anime-recommender")
        
        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=768,  # all-mpnet-base-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            # Wait for index to be ready
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        
        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")

    def parse_categories(self, s):
        if pd.isna(s) or not s:
            return []
        import re
        return [p.strip().lower() for p in re.split(r",|;", str(s)) if p.strip()]

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
        
        # Clamp quality_score to [0.0, 1.0]
        df["quality_score"] = df["quality_score"].clip(0.0, 1.0)

        self.df = df
        return df

    def create_embeddings(self) -> np.ndarray:
        texts = self.df["semantic_text"].tolist()
        emb = self.sentence_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
        )
        return emb.astype(np.float32)

    def upload_to_pinecone(self, embeddings: np.ndarray):
        """Upload embeddings to Pinecone in batches"""
        if self.index is None:
            logger.warning("Pinecone not initialized. Skipping upload.")
            return
        
        logger.info("Uploading embeddings to Pinecone...")
        batch_size = 200
        vectors = []
        
        for idx in range(len(embeddings)):
            # Create metadata for each anime
            metadata = {
                "name": str(self.df.iloc[idx]["Name"]),
                "rank": int(self.df.iloc[idx].get("Rank", 0)),
                "rating": float(self.df.iloc[idx].get("Rating", 0.0)),
                "year": int(self.df.iloc[idx].get("Release_year", 0)),
                "type": str(self.df.iloc[idx].get("Type", "")),
                "studio": str(self.df.iloc[idx].get("Studio", ""))[:100],
                "genres": str(self.df.iloc[idx].get("Tags", ""))[:200],
            }
            
            vectors.append({
                "id": str(idx),
                "values": embeddings[idx].tolist(),
                "metadata": metadata
            })
            
            # Upload in batches
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors)
                vectors = []
                logger.info(f"Uploaded {idx + 1}/{len(embeddings)} vectors")
        
        # Upload remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors)
        
        logger.info(f"Successfully uploaded {len(embeddings)} vectors to Pinecone")

    def create_tfidf(self) -> Tuple[object, np.ndarray]:
        tf = TfidfVectorizer(
            max_features=3000, stop_words="english", ngram_range=(1, 2), min_df=2
        )
        mat = tf.fit_transform(self.df["semantic_text"]).toarray()

        # normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms

        return tf, mat

    def save_artifacts(self, embeddings, tfidf_vectorizer, tfidf_matrix):
        os.makedirs(self.model_dir, exist_ok=True)

        self.df.to_csv(os.path.join(self.model_dir, "anime_processed.csv"), index=False)
        np.save(os.path.join(self.model_dir, "embeddings.npy"), embeddings)
        np.save(os.path.join(self.model_dir, "tfidf_matrix.npy"), tfidf_matrix)
        joblib.dump(
            {"tfidf_vectorizer": tfidf_vectorizer},
            os.path.join(self.model_dir, "encoders.pkl")
        )

        logger.info("Saved artifacts to %s", self.model_dir)

    def train(self, data_path: str = "Anime.csv"):
        self.preprocess(data_path)
        embeddings = self.create_embeddings()
        tfidf_vectorizer, tfidf_matrix = self.create_tfidf()
        
        # Upload to Pinecone
        self.upload_to_pinecone(embeddings)
        
        # Save local artifacts
        self.save_artifacts(embeddings, tfidf_vectorizer, tfidf_matrix)
        logger.info("Training completed")


if __name__ == "__main__":
    trainer = CorrectAnimeTrainer()
    trainer.train()
