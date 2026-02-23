"""
Analyze router — Milestone 1
Accepts keywords, searches arXiv corpus, returns NLP analysis results.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Form, UploadFile, File

from app.services.nlp_pipeline import run_pipeline, load_dataset

router = APIRouter()


@router.on_event("startup")
async def preload_dataset():
    """Load arXiv dataset + build TF-IDF at server startup."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_dataset)


@router.post("/analyze")
async def analyze(
    keywords: List[str] = Form(default=[]),
    num_topics: int = Form(default=5),
    summary_sentences: int = Form(default=3),
    use_bow: bool = Form(default=False),
    files: Optional[List[UploadFile]] = File(default=None),
):
    """
    Analyze research papers from the arXiv corpus matching the given keywords.
    Keywords are used to search through 20k pre-loaded arXiv paper abstracts.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: run_pipeline(
            keywords=keywords,
            num_topics=num_topics,
            summary_sentences=summary_sentences,
        ),
    )
    return result
