#!/bin/bash
set -e

echo "Starting Full Data Pipeline"

# 1. Ingestion: Fetches data and places it into data/raw or data/processed
echo "Step 1: Ingestion (Scraping & Raw Processing)..."
python -m src.ingestion.run_ingestion_pipeline

# 2. Indexing: Processes raw/processed data and creates data/merged/unified.jsonl
echo "Step 2: Indexing (Chunking & Refining)..."
python -m src.indexing.chunk_pipeline

# 3. Embedding: Reads unified.jsonl and upserts to ChromaDB (Server)
echo "Step 3: Embedding (Vector DB Update)..."
python -m src.database.run_manager

# 4. Cleanup: Remove intermediate files to save space
echo "Cleaning up intermediate data..."
# We remove processed and merged data. 
# We do NOT remove data/raw to protect manually uploaded files (PDFs/IPYNBs).
if [ -d "data/processed" ]; then
    echo "Removing data/processed contents..."
    rm -rf data/processed/*
fi

if [ -d "data/merged" ]; then
    echo "Removing data/merged contents..."
    rm -rf data/merged/*
fi

echo "Pipeline Completed Successfully! Intermediate data cleaned."
