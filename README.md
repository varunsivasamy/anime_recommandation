# Correct Anime Recommender

A hybrid semantic anime recommendation system using sentence transformers, FAISS, and TF-IDF.

## Project Structure

- `correct_recommender.py` - Core recommender class with semantic search and re-ranking
- `correct_trainer.py` - Training pipeline that builds embeddings and artifacts
- `correct_chat.py` - CLI chat interface for testing recommendations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Place your anime dataset as `Anime.csv` in the project root
   - Required columns: Name, Description, Tags, Rating, Rank, Release_year, Type, Studio

3. Train the model:
```bash
python correct_trainer.py
```

This will create a `correct_model/` directory with:
- anime_processed.csv
- embeddings.npy
- faiss_index.bin
- encoders.pkl
- tfidf_matrix.npy

4. Run the chat interface:
```bash
python correct_chat.py
```

## Usage

Query examples:
- "dark psychological anime"
- "romance with sad ending"
- "action anime from 2020"
- Type "stats" to see performance metrics
- Type "quit" to exit

## Features

- Semantic search using sentence transformers
- Hybrid scoring (semantic + TF-IDF + quality)
- Query expansion and mood detection
- Diversity filtering
- User personalization support
- Query caching for performance
