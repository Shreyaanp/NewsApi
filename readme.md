Installation and Setup Guide
Overview
This improved News Analytics API provides robust RSS feed aggregation, sentiment analysis, and content clustering with enhanced error handling, performance optimizations, and production-ready features.

Key Improvements
1. Enhanced Models (models.py)
Flexible URL validation: Uses AnyHttpUrl instead of strict HttpUrl

Comprehensive field validation: Custom validators for text cleaning and data sanitization

Error handling: Graceful handling of parsing errors and missing data

Enum support: Proper sentiment label enumeration

Model composition: Response wrappers and error handling models

2. Async RSS Fetching (news_fetcher.py)
Full async implementation: Non-blocking RSS feed fetching

Connection management: Proper connection pooling and timeouts

Retry logic: Exponential backoff for failed requests

Error recovery: Graceful degradation when feeds fail

Content cleaning: Advanced text preprocessing and sanitization

Deduplication: Automatic removal of duplicate articles

3. Improved Main Application (main.py)
Lifespan management: Proper startup/shutdown with model preloading

Enhanced middleware: Request logging and performance monitoring

Comprehensive error handling: Global exception handlers

Health checks: Detailed health monitoring endpoints

Background tasks: Automatic feed updates

4. Richer Clustering (reach.py)
Description-aware embeddings for clustering and similarity scoring for each cluster

API improvements: Better pagination, filtering, and response models

4. Advanced Sentiment Analysis (sentiment.py)
Model optimization: Efficient loading and caching of ML models

Batch processing: Optimized batch analysis for better performance

GPU support: Automatic GPU detection and utilization

Caching: Results caching to avoid repeated analysis

Error resilience: Fallback to neutral sentiment on errors

Trend analysis: Time-based sentiment tracking

5. Enhanced Caching (cache.py)
Redis support: Production-ready Redis caching with fallback

Cache management: Statistics, health checks, and monitoring

Error handling: Graceful cache failures

Context managers: Clean cache operations

Decorators: Easy function result caching

Installation
Prerequisites
bash
# Python 3.8+
python --version

# Install dependencies
pip install fastapi uvicorn aiohttp aiocache redis
pip install torch transformers sentence-transformers
pip install scikit-learn pandas numpy
pip install pydantic[email] python-multipart
Optional Redis Setup
bash
# Install Redis (Ubuntu/Debian)
sudo apt update
sudo apt install redis-server

# Or using Docker
docker run -d -p 6379:6379 redis:alpine

# Set environment variables
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
Configuration
Environment Variables
bash
# Cache configuration
CACHE_TTL_SECONDS=3600
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password  # if required

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Model configuration
SENTIMENT_MODEL=cardiffnlp/twitter-xlm-roberta-base-sentiment
SIMILARITY_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RSS Feeds Configuration (config.py)
python
RSS_FEEDS = [
    'https://feeds.bbci.co.uk/news/rss.xml',
    'https://rss.cnn.com/rss/edition.rss',
    'https://www.theguardian.com/world/rss',
    # Add your feeds here
]
Running the Application
Development
bash
# Start the application
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or using the script
python -m main
Production
bash
# Using Gunicorn with Uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
docker build -t news-analytics .
docker run -p 8000:8000 -e REDIS_URL=redis://host.docker.internal:6379 news-analytics
API Endpoints
Core Endpoints
GET / - API information

GET /health - Comprehensive health check

GET /stats - Usage statistics

Content Endpoints
GET /articles - Get articles with pagination and filtering

?limit=100 - Number of articles

?offset=0 - Skip articles

?source=cnn - Filter by source

Analytics Endpoints
GET /analytics/sentiment - Overall sentiment summary

GET /analytics/sentiment/hourly - Hourly sentiment counts for recent articles

GET /analytics/reach - Content clustering analysis with similarity scores

GET /analytics/trend - Trend analysis with ratios

Admin Endpoints
POST /admin/refresh - Force refresh articles

Monitoring and Maintenance
Health Monitoring
bash
# Check API health
curl http://localhost:8000/health

# Check cache statistics
curl http://localhost:8000/stats
Logs
The application provides structured logging with different levels:

Request/response logging

Error tracking

Performance monitoring

Cache statistics

Performance Tuning
Redis Configuration: Use Redis for production caching

Worker Processes: Scale with multiple Gunicorn workers

Connection Pooling: Configure aiohttp connection limits

Model Caching: ML models are cached globally

Batch Processing: Sentiment analysis uses batching

Troubleshooting
Common Issues
RSS Feed Timeouts

Check network connectivity

Increase timeout values in news_fetcher.py

Monitor failed feeds in logs

Memory Issues

Reduce ML model batch sizes

Use CPU instead of GPU for smaller deployments

Monitor memory usage with health checks

Cache Issues

Verify Redis connection

Check cache statistics endpoint

Clear cache if corrupted: POST /admin/refresh

Model Loading Errors

Ensure sufficient disk space for model downloads

Check internet connectivity for initial download

Verify GPU/CUDA setup if using GPU

Error Recovery
The application includes comprehensive error handling:

Automatic retries for temporary failures

Graceful degradation when services fail

Fallback responses for ML model errors

Cache failure handling with in-memory fallback

Security Considerations
Production Deployment
Environment Variables: Use secure secret management

CORS Configuration: Restrict allowed origins

Rate Limiting: Implement request rate limiting

SSL/TLS: Use HTTPS in production

Input Validation: All inputs are validated and sanitized

Monitoring
Set up application monitoring (e.g., Datadog, New Relic)

Configure log aggregation (e.g., ELK Stack)

Monitor resource usage and performance metrics

Set up alerts for health check failures

Scaling
Horizontal Scaling
Deploy multiple application instances

Use load balancer (Nginx, HAProxy)

Shared Redis instance for caching

Database for persistent storage (optional)

Vertical Scaling
Increase worker processes

Allocate more memory for ML models

Use faster storage for model caching

Upgrade to GPU instances for better ML performance