from datetime import datetime
from typing import List, Optional, Union
from enum import Enum
from pydantic.networks import AnyHttpUrl
import logging, re   # ← add re

from pydantic import BaseModel, Field, field_validator, model_validator, FieldValidationInfo
logger = logging.getLogger(__name__)
MAX_DESC_LEN = 5000 

class SentimentLabel(str, Enum):
    """Enumeration for sentiment labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"

class Article(BaseModel):
    # ── enrichment fields (filled later) ──────────────────────────
    id:            str | None = Field(default=None, description="MD5(url) – filled in later")
    cron_timestamp: datetime  = Field(default_factory=datetime.utcnow)
    cluster_id:    Optional[str] = None
    district:      Optional[str] = None
    # ──────────────────────────────────────────────────────────────

    title:        str            = Field(..., min_length=1, max_length=500)
    url:          AnyHttpUrl     = Field(...)
    source:       str            = Field(..., min_length=1, max_length=200)
    published_at: datetime       = Field(default_factory=datetime.utcnow)
    # **removed max_length here – we truncate ourselves**
    description:  str            = Field(default="", description="Cleaned & truncated desc")
    image_url:    Optional[AnyHttpUrl] = None
    author:       str            = Field(default="Unknown", max_length=200)
    categories:   List[str]      = Field(default_factory=list)
    content:      Optional[str]  = Field(None, max_length=10_000)
    sentiment:    Optional[SentimentLabel] = None

    # ---------- validators ----------
    @field_validator("title", "description", mode="before")
    @classmethod
    def clean_and_truncate(cls, v: str, info: FieldValidationInfo):
        if not isinstance(v, str):
            return ""
        # strip HTML + collapse whitespace
        v = re.sub(r"<[^>]+>", "", v)
        v = re.sub(r"\s+", " ", v).strip()
        if info.field_name == "description" and len(v) > MAX_DESC_LEN:
            v = v[: MAX_DESC_LEN - 1] + "…"
        return v

    @field_validator("categories", mode="before")
    @classmethod
    def dedupe_categories(cls, v, info: FieldValidationInfo):
        return list({cat.strip() for cat in (v or []) if cat.strip()})

    @model_validator(mode="before")
    @classmethod
    def fix_common_feed_issues(cls, data):
        if not isinstance(data, dict):
            return data
        # blank title → placeholder
        if not data.get("title", "").strip():
            data["title"] = "Untitled Article"
        # _try_ to normalise bad URLs
        url = str(data.get("url", ""))
        if url and not url.startswith(("http://", "https://")):
            data["url"] = f"https://{url}"
        if not data.get("published_at"):
            data["published_at"] = datetime.utcnow()
        return data

    class Config:
        validate_assignment = True
        extra = "ignore"
        str_strip_whitespace = True
        
class SentimentSummary(BaseModel):
    """Summary of sentiment analysis results"""
    positive: int = Field(ge=0, description="Number of positive articles")
    negative: int = Field(ge=0, description="Number of negative articles")
    neutral: int = Field(ge=0, description="Number of neutral articles")
    total: int = Field(ge=0, description="Total number of articles")
    positive_ratio: float = Field(ge=0.0, le=1.0, description="Positive articles ratio")
    
    @model_validator(mode='before')
    @classmethod
    def calculate_totals(cls, data):
        """Calculate total and ratios"""
        if isinstance(data, dict):
            total = data.get('positive', 0) + data.get('negative', 0) + data.get('neutral', 0)
            data['total'] = total
            if total > 0:
                data['positive_ratio'] = data.get('positive', 0) / total
            else:
                data['positive_ratio'] = 0.0
        return data

class ReachCluster(BaseModel):
    """Cluster of similar articles based on content similarity"""
    centroid_title: str = Field(..., min_length=1, description="Representative title")
    size: int = Field(ge=1, description="Number of articles in cluster")
    titles: List[str] = Field(..., min_items=1, description="All titles in cluster")
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Cluster similarity score")
    
    @field_validator('titles')
    @classmethod
    def validate_titles(cls, v: List[str]) -> List[str]:
        """Ensure all titles are valid"""
        return [title.strip() for title in v if title.strip()]
    
    @model_validator(mode='before')
    @classmethod
    def validate_consistency(cls, data):
        """Ensure size matches titles length"""
        if isinstance(data, dict) and 'titles' in data:
            data['size'] = len(data['titles'])
        return data

class APIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(default="Success", description="Response message")
    data: Optional[Union[List[Article], SentimentSummary, List[ReachCluster]]] = Field(None)
    error: Optional[str] = Field(None, description="Error details if any")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0", description="API version")
    checks: dict = Field(default_factory=dict, description="Individual service checks")

# Exception handling models
class ValidationErrorDetail(BaseModel):
    """Detailed validation error information"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Error message")
    invalid_value: Optional[str] = Field(None, description="The invalid value")

class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = Field(default=False)
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[ValidationErrorDetail]] = Field(None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)