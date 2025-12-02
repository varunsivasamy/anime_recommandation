# Correct Anime Recommender

A hybrid semantic anime recommendation system using sentence transformers, Pinecone cloud vector database, and TF-IDF.

## Project Structure

- `correct_recommender.py` - Core recommender class with semantic search and re-ranking
- `correct_trainer.py` - Training pipeline that builds embeddings and uploads to Pinecone
- `correct_chat.py` - CLI chat interface for testing recommendations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Pinecone:
   - Sign up for a free account at https://www.pinecone.io/
   - Get your API key from the Pinecone console
   - Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
   - Edit `.env` with your Pinecone API key:
   ```
   PINECONE_API_KEY=your_api_key_here
   PINECONE_INDEX_NAME=anime-recommender
   ```

3. Prepare your data:
   - Place your anime dataset as `Anime.csv` in the project root
   - Required columns: Name, Description, Tags, Rating, Rank, Release_year, Type, Studio

4. Train the model:
```bash
python correct_trainer.py
```

This will:
- Upload embeddings to Pinecone cloud
- Create a `correct_model/` directory with:
  - anime_processed.csv
  - embeddings.npy (local backup)
  - encoders.pkl
  - tfidf_matrix.npy

5. Run the chat interface:
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

- **Cloud-based vector search** using Pinecone for scalability
- Semantic search using sentence transformers (all-mpnet-base-v2)
- Hybrid scoring (semantic + TF-IDF + quality)
- Query expansion and mood detection
- Diversity filtering
- User personalization support
- Query caching for performance
- Automatic fallback to local search if Pinecone is unavailable

## Architecture

- **Pinecone**: Cloud vector database for fast similarity search
- **Sentence Transformers**: Generate 768-dim embeddings
- **TF-IDF**: Keyword-based relevance scoring
- **Hybrid Re-ranking**: Combines semantic, keyword, and quality signals

## Environment Variables

Create a `.env` file with:
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=anime-recommender
```
