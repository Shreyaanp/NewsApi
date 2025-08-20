import asyncio
from functools import lru_cache
from collections import Counter
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import hashlib

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

from .models import Article, SentimentSummary, SentimentLabel
from .cache import default_cache

logger = logging.getLogger(__name__)

# Global model cache
_sentiment_pipeline = None
_model_load_time = None

class SentimentAnalyzer:
    """Enhanced sentiment analysis with caching and error handling"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"):
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
        
    async def load_model(self) -> None:
        """Load sentiment analysis model with proper error handling"""
        global _sentiment_pipeline, _model_load_time
        
        if _sentiment_pipeline is not None and _model_load_time is not None:
            # Check if model is still fresh (reload every 24 hours)
            if datetime.utcnow() - _model_load_time < timedelta(hours=24):
                self.pipeline = _sentiment_pipeline
                logger.info("Using cached sentiment model")
                return
        
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            start_time = datetime.utcnow()
            
            # Load model and tokenizer in thread to avoid blocking
            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, self.model_name
            )
            
            self.model = await asyncio.to_thread(
                AutoModelForSequenceClassification.from_pretrained, self.model_name
            )
            
            # Create pipeline
            self.pipeline = await asyncio.to_thread(
                pipeline,
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_all_scores=False
            )
            
            # Cache globally
            _sentiment_pipeline = self.pipeline
            _model_load_time = datetime.utcnow()
            
            load_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Sentiment model loaded successfully in {load_time:.2f}s on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise RuntimeError(f"Could not load sentiment analysis model: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Clean and truncate text
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate to avoid model limits
        if len(text) > self.max_length * 4:  # Rough character estimate
            text = text[:self.max_length * 4]
        
        return text
    
    async def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of texts"""
        if not self.pipeline:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not texts:
            return []
        
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Filter out empty texts
            valid_texts = [(i, text) for i, text in enumerate(processed_texts) if text]
            
            if not valid_texts:
                return [{"label": "neutral", "score": 0.5} for _ in texts]
            
            # Analyze in thread to avoid blocking
            logger.debug(f"Analyzing sentiment for {len(valid_texts)} texts")
            start_time = datetime.utcnow()
            
            results = await asyncio.to_thread(
                self.pipeline, 
                [text for _, text in valid_texts]
            )
            
            analysis_time = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"Sentiment analysis completed in {analysis_time:.2f}s")
            
            # Map results back to original order
            full_results = [{"label": "neutral", "score": 0.5} for _ in texts]
            for (original_index, _), result in zip(valid_texts, results):
                full_results[original_index] = result
            
            return full_results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Return neutral sentiment as fallback
            return [{"label": "neutral", "score": 0.5} for _ in texts]
    
    def normalize_label(self, label: str) -> SentimentLabel:
        """Normalize sentiment labels to standard format"""
        label_lower = label.lower().strip()
        
        # Map various label formats to standard ones
        if label_lower in ["positive", "pos", "1", "good", "happy"]:
            return SentimentLabel.POSITIVE
        elif label_lower in ["negative", "neg", "-1", "bad", "sad"]:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL

# Global instance
_sentiment_analyzer = None

async def get_or_load_sentiment_model() -> SentimentAnalyzer:
    """Get or load the global sentiment analyzer"""
    global _sentiment_analyzer
    
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
        await _sentiment_analyzer.load_model()
    
    return _sentiment_analyzer

async def sentiment_counts(articles: List[Article], analyzer: Optional[SentimentAnalyzer] = None) -> SentimentSummary:
    """
    Analyze sentiment for all articles and return summary with caching
    """
    if not articles:
        return SentimentSummary(
            positive=0,
            negative=0,
            neutral=0,
            total=0,
            positive_ratio=0.0
        )
    
    # Use provided analyzer or get global one
    if analyzer is None:
        analyzer = await get_or_load_sentiment_model()
    
    # Create cache key based on article content
    article_hash = hashlib.md5(
        "|".join(f"{a.url}:{a.title}" for a in articles).encode()
    ).hexdigest()
    cache_key = f"sentiment_summary:{article_hash}"
    
    # Check cache first
    cached_result = await default_cache.get(cache_key)
    if cached_result:
        logger.info("Using cached sentiment analysis")
        return SentimentSummary(**cached_result)
    
    logger.info(f"Analyzing sentiment for {len(articles)} articles")
    start_time = datetime.utcnow()
    
    try:
        # Extract texts for analysis
        texts = []
        for article in articles:
            # Use title and description for sentiment analysis
            text = article.title
            if article.description:
                text += " " + article.description
            texts.append(text)
        
        # Analyze sentiments in batches
        batch_size = 32  # Process in smaller batches for memory efficiency
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = await analyzer.analyze_batch(batch_texts)
            all_results.extend(batch_results)
        
        # Count sentiments
        sentiment_counts_dict = Counter()
        
        for result in all_results:
            normalized_label = analyzer.normalize_label(result["label"])
            sentiment_counts_dict[normalized_label.value] += 1
        
        # Create summary
        summary = SentimentSummary(
            positive=sentiment_counts_dict.get("positive", 0),
            negative=sentiment_counts_dict.get("negative", 0),
            neutral=sentiment_counts_dict.get("neutral", 0)
        )
        
        # Cache the result
        await default_cache.set(cache_key, summary.model_dump(), ttl=3600)  # Cache for 1 hour
        
        analysis_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Sentiment analysis completed in {analysis_time:.2f}s")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        # Return neutral sentiment as fallback
        return SentimentSummary(
            positive=0,
            negative=0,
            neutral=len(articles),
            total=len(articles),
            positive_ratio=0.0
        )

async def analyze_article_sentiments(articles: List[Article], analyzer: Optional[SentimentAnalyzer] = None) -> List[Article]:
    """
    Analyze sentiment for individual articles and attach to article objects
    """
    if not articles:
        return articles
    
    # Use provided analyzer or get global one
    if analyzer is None:
        analyzer = await get_or_load_sentiment_model()
    
    logger.info(f"Analyzing individual sentiments for {len(articles)} articles")
    
    try:
        # Extract texts
        texts = []
        for article in articles:
            text = article.title
            if article.description:
                text += " " + article.description
            texts.append(text)
        
        # Analyze sentiments
        results = await analyzer.analyze_batch(texts)
        
        # Attach sentiments to articles
        enhanced_articles = []
        for article, result in zip(articles, results):
            # Create a copy of the article with sentiment
            article_dict = article.model_dump()
            article_dict["sentiment"] = analyzer.normalize_label(result["label"])
            enhanced_articles.append(Article(**article_dict))
        
        return enhanced_articles
        
    except Exception as e:
        logger.error(f"Error analyzing individual sentiments: {e}")
        return articles  # Return original articles on error

async def get_sentiment_trends(articles: List[Article], time_window_hours: int = 24) -> Dict[str, Any]:
    """
    Get sentiment trends over time
    """
    if not articles:
        return {"error": "No articles available"}
    
    try:
        # Analyze sentiments
        analyzer = await get_or_load_sentiment_model()
        enhanced_articles = await analyze_article_sentiments(articles, analyzer)
        
        # Filter articles by time window
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_articles = [
            a for a in enhanced_articles 
            if a.published_at >= cutoff_time
        ]
        
        if not recent_articles:
            return {"error": f"No articles in the last {time_window_hours} hours"}
        
        # Group by hour
        hourly_sentiments = {}
        for article in recent_articles:
            hour_key = article.published_at.strftime("%Y-%m-%d %H:00")
            if hour_key not in hourly_sentiments:
                hourly_sentiments[hour_key] = {"positive": 0, "negative": 0, "neutral": 0}
            
            if article.sentiment:
                hourly_sentiments[hour_key][article.sentiment.value] += 1
        
        # Sort by hour
        sorted_hours = sorted(hourly_sentiments.keys())
        
        trends = {
            "time_window_hours": time_window_hours,
            "total_articles": len(recent_articles),
            "hourly_data": [
                {
                    "hour": hour,
                    "sentiments": hourly_sentiments[hour],
                    "total": sum(hourly_sentiments[hour].values())
                }
                for hour in sorted_hours
            ],
            "overall_trend": "positive" if sum(
                h["sentiments"]["positive"] for h in hourly_sentiments.values()
            ) > sum(
                h["sentiments"]["negative"] for h in hourly_sentiments.values()
            ) else "negative"
        }
        
        return trends
        
    except Exception as e:
        logger.error(f"Error calculating sentiment trends: {e}")
        return {"error": f"Failed to calculate trends: {str(e)}"}

# Utility functions for backward compatibility
async def label_articles(articles: List[Article]) -> None:
    """
    Add sentiment labels to articles in-place (for backward compatibility)
    """
    try:
        enhanced_articles = await analyze_article_sentiments(articles)
        
        for original, enhanced in zip(articles, enhanced_articles):
            if enhanced.sentiment:
                setattr(original, 'sentiment', enhanced.sentiment.value)
                
    except Exception as e:
        logger.error(f"Error labeling articles: {e}")

# Health check for sentiment analysis
async def sentiment_health_check() -> Dict[str, Any]:
    """Check if sentiment analysis is working properly"""
    try:
        analyzer = await get_or_load_sentiment_model()
        
        # Test with sample text
        test_texts = [
            "This is great news!",
            "This is terrible news.",
            "This is neutral news."
        ]
        
        start_time = datetime.utcnow()
        results = await analyzer.analyze_batch(test_texts)
        analysis_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "status": "healthy",
            "model": analyzer.model_name,
            "device": analyzer.device,
            "response_time_seconds": analysis_time,
            "test_results": results
        }
        
    except Exception as e:
        logger.error(f"Sentiment health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }