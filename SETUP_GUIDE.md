# Pinecone Setup Guide

## Step 1: Create Pinecone Account

1. Go to https://www.pinecone.io/
2. Click "Sign Up" and create a free account
3. Verify your email

## Step 2: Get API Key

1. Log into Pinecone console: https://app.pinecone.io/
2. Go to "API Keys" in the left sidebar
3. Copy your API key (starts with `pcsk_` or similar)

## Step 3: Configure Environment

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API key:
```
PINECONE_API_KEY=your_actual_api_key_here
PINECONE_INDEX_NAME=anime-recommender
```

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Train and Upload

```bash
python correct_trainer.py
```

This will:
- Process the Anime.csv dataset
- Generate embeddings using sentence transformers
- Create a Pinecone index (if it doesn't exist)
- Upload all vectors to Pinecone cloud
- Save local artifacts for TF-IDF and fallback

## Step 6: Test the Recommender

```bash
python correct_chat.py
```

Try queries like:
- "dark psychological anime"
- "romance with happy ending"
- "action anime from 2020"

## Troubleshooting

### "PINECONE_API_KEY not found"
- Make sure `.env` file exists in the project root
- Check that the API key is correctly copied (no extra spaces)

### "Failed to connect to Pinecone"
- Verify your API key is valid
- Check your internet connection
- The system will automatically fall back to local search

### Index creation fails
- Free tier allows 1 index
- Delete existing indexes in Pinecone console if needed
- Or change `PINECONE_INDEX_NAME` in `.env`

## Pinecone Free Tier Limits

- 1 index
- 100K vectors
- 768 dimensions
- Serverless (pay-as-you-go after free tier)

Perfect for this anime recommender project!
