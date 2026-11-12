from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator

# ============================================================================
# USER & AUTH SCHEMAS
# ============================================================================

class UserRegister(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    name: str = Field(..., min_length=2, max_length=100)


class UserLogin(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str = "user"
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ============================================================================
# ITEM SCHEMAS & MODELS
# ============================================================================

class ItemType(str, Enum):
    lost = "lost"
    found = "found"


class ItemCategory(str, Enum):
    electronics = "electronics"
    clothing = "clothing"
    accessories = "accessories"
    documents = "documents"
    keys = "keys"
    pets = "pets"
    sports = "sports"
    books = "books"
    other = "other"


class ItemCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=100)
    description: str = Field(..., min_length=10, max_length=1000)
    category: ItemCategory
    color: Optional[str] = Field(default=None, max_length=50)
    brand: Optional[str] = Field(default=None, max_length=100)
    location: str = Field(..., min_length=3, max_length=200)
    latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    date_lost_or_found: str
    contact_name: Optional[str] = Field(default=None, max_length=100)
    contact_email: Optional[str] = Field(default=None, max_length=200)
    verification_questions: Optional[List[str]] = None
    private_details: Optional[str] = None

    @field_validator("date_lost_or_found")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date_lost_or_found must be in YYYY-MM-DD format, e.g. '2024-11-01'")
        return v

    @field_validator("title", "location")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()

    @field_validator("contact_email")
    @classmethod
    def validate_email_basic(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and "@" not in v:
            raise ValueError("contact_email must be a valid email address")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Blue iPhone 14",
                "description": "Lost near the cafeteria. Has a cracked screen protector.",
                "category": "electronics",
                "color": "Blue",
                "brand": "Apple",
                "location": "Cafeteria, Building A",
                "date_lost_or_found": "2024-11-01",
                "contact_name": "Jiya Sharma",
                "contact_email": "jiya@example.com",
            }
        }
    }


class ItemResponse(BaseModel):
    id: str
    item_type: str
    title: str
    description: str
    category: str
    color: Optional[str] = None
    brand: Optional[str] = None
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    date_lost_or_found: str
    image_url: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    owner_id: Optional[str] = None
    status: str = "active"
    verification_questions: Optional[List[str]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ItemListResponse(BaseModel):
    count: int
    items: List[ItemResponse]


class ItemInDB(BaseModel):
    id: str
    item_type: str
    created_at: datetime
    title: str
    description: str
    category: str
    color: Optional[str] = None
    brand: Optional[str] = None
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    date_lost_or_found: str
    image_url: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    owner_id: Optional[str] = None
    status: str = "active"
    verification_questions: Optional[List[str]] = None
    private_details: Optional[str] = None


# ============================================================================
# MATCHING ENGINE SCHEMAS
# ============================================================================

class MatchItem(BaseModel):
    id: str
    item_type: str
    title: str
    description: str
    category: str
    color: Optional[str] = None
    brand: Optional[str] = None
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    date_lost_or_found: str
    image_url: Optional[str] = None
    similarity_score: float


class MatchResponse(BaseModel):
    target_item_id: str
    target_item_type: str
    target_title: str
    threshold: float
    total_candidates: int
    matched_count: int
    matches: List[MatchItem]


class ImageMatchItem(BaseModel):
    id: str
    title: str
    image_url: str
    similarity_score: float


class ImageMatchResponse(BaseModel):
    target_item_id: str
    target_item_type: str
    target_title: str
    target_image_url: str
    threshold: float
    total_candidates: int
    valid_image_candidates: int
    matched_count: int
    matches: List[ImageMatchItem]


class AttributeMatchItem(BaseModel):
    id: str
    item_type: str
    title: str
    category: str
    color: Optional[str] = None
    brand: Optional[str] = None
    attribute_score: float
    breakdown: dict
    explanation: str


class AttributeMatchResponse(BaseModel):
    target_item_id: str
    target_item_type: str
    target_title: str
    threshold: float
    total_candidates: int
    matched_count: int
    matches: List[AttributeMatchItem]


class LocationTimeMatchItem(BaseModel):
    id: str
    item_type: str
    title: str
    location: str
    date_lost_or_found: str
    location_score: float
    time_score: float
    combined_score: float
    distance_km: Optional[float] = None
    time_difference_days: Optional[float] = None
    explanation: str


class LocationTimeMatchResponse(BaseModel):
    target_item_id: str
    target_item_type: str
    target_title: str
    target_location: str
    target_date: str
    threshold: float
    max_days: int
    total_candidates: int
    matched_count: int
    matches: List[LocationTimeMatchItem]


class HybridMatchItem(BaseModel):
    id: str
    item_type: str
    title: str
    image_url: Optional[str] = None
    hybrid_score: float
    text_score: Optional[float] = None
    image_score: Optional[float] = None
    attribute_score: Optional[float] = None
    location_score: Optional[float] = None
    time_score: Optional[float] = None
    signals_used: List[str]
    signals_unavailable: List[str]
    explanation: str


class HybridMatchResponse(BaseModel):
    target_item_id: str
    target_item_type: str
    target_title: str
    threshold: float
    weights: dict
    total_candidates: int
    matched_count: int
    matches: List[HybridMatchItem]


# ============================================================================
# PERSISTED MATCH LIFECYCLE SCHEMAS
# ============================================================================

class MatchStatus(str, Enum):
    suggested = "suggested"
    viewed = "viewed"
    interested = "interested"
    claim_pending = "claim_pending"
    verified = "verified"
    rejected = "rejected"
    recovered = "recovered"


class MatchStatusUpdate(BaseModel):
    status: MatchStatus


class MatchRecordResponse(BaseModel):
    id: str
    lost_item_id: str
    found_item_id: str
    hybrid_score: float
    scores: Dict[str, Any] = {}
    explanation: str = ""
    status: str
    created_at: datetime
    updated_at: datetime


class MatchListResponse(BaseModel):
    total: int
    items: List[MatchRecordResponse]


# ============================================================================
# CLAIMS & VERIFICATION SCHEMAS
# ============================================================================

class ClaimStatus(str, Enum):
    submitted = "submitted"
    under_review = "under_review"
    additional_information_required = "additional_information_required"
    approved = "approved"
    rejected = "rejected"
    cancelled = "cancelled"


class ClaimCreate(BaseModel):
    match_id: Optional[str] = None
    found_item_id: str
    lost_item_id: Optional[str] = None
    claim_description: str = Field(..., min_length=5, max_length=1000)
    answers: Optional[Dict[str, str]] = {}


class ClaimDecision(BaseModel):
    decision: ClaimStatus
    note: Optional[str] = None


class ClaimCancel(BaseModel):
    reason: Optional[str] = None


class ClaimAuditEntry(BaseModel):
    action: str
    by_user_id: str
    from_status: Optional[str] = None
    to_status: str
    note: Optional[str] = None
    timestamp: datetime


class ClaimResponse(BaseModel):
    id: str
    match_id: Optional[str] = None
    claimant_id: str
    found_item_id: str
    lost_item_id: Optional[str] = None
    claim_description: str
    status: str
    claim_status: Optional[str] = None
    answers: Dict[str, str] = {}
    audit_trail: List[Dict[str, Any]] = []
    created_at: datetime
    updated_at: datetime


class ClaimListResponse(BaseModel):
    total: int
    items: List[ClaimResponse]


# ============================================================================
# NOTIFICATIONS SCHEMAS
# ============================================================================

class NotificationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    message: str
    type: str
    reference_id: str
    is_read: bool = False
    created_at: datetime


class NotificationListResponse(BaseModel):
    total: int
    unread_count: int
    items: List[NotificationResponse]
