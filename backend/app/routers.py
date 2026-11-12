import os
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, Query, UploadFile, File, Depends

from app.schemas import (
    UserRegister,
    UserLogin,
    UserResponse,
    TokenResponse,
    ItemCreate,
    ItemResponse,
    ItemListResponse,
    MatchResponse,
    MatchItem,
    ImageMatchResponse,
    ImageMatchItem,
    AttributeMatchResponse,
    AttributeMatchItem,
    LocationTimeMatchResponse,
    LocationTimeMatchItem,
    HybridMatchResponse,
    HybridMatchItem,
    MatchStatus,
    MatchStatusUpdate,
    MatchRecordResponse,
    MatchListResponse,
    ClaimCreate,
    ClaimDecision,
    ClaimCancel,
    ClaimResponse,
    ClaimListResponse,
    ClaimStatus,
    NotificationResponse,
    NotificationListResponse,
)
from app.core import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    get_optional_current_user,
)
from app import database as crud
from app.matching import (
    find_matches,
    process_image_matching,
    find_attribute_matches,
    find_location_time_matches,
    find_hybrid_matches,
    DEFAULT_WEIGHTS,
)
from app.services import (
    upload_image_bytes,
    match_service,
    claim_service,
)


# ============================================================================
# 1. AUTHENTICATION ROUTER
# ============================================================================

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(data: UserRegister) -> TokenResponse:
    existing = await crud.get_user_by_email(data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    user_doc = {
        "email": data.email.lower().strip(),
        "hashed_password": hash_password(data.password),
        "name": data.name.strip(),
        "role": "user",
        "created_at": datetime.now(timezone.utc),
    }

    created = await crud.save_user(user_doc)
    token = create_access_token({"sub": created["id"], "role": created["role"]})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse(**created),
    )


@auth_router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin) -> TokenResponse:
    user = await crud.get_user_by_email(data.email)
    if not user or not verify_password(data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token({"sub": user["id"], "role": user.get("role", "user")})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse(**user),
    )


@auth_router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)) -> UserResponse:
    return UserResponse(**current_user)


# ============================================================================
# 2. ITEMS ROUTER
# ============================================================================

items_router = APIRouter(prefix="/items", tags=["Items"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def _build_item_document(data: ItemCreate, item_type: str, owner_id: Optional[str] = None) -> dict:
    return {
        "item_type": item_type,
        "created_at": datetime.now(timezone.utc),
        "title": data.title,
        "description": data.description,
        "category": data.category.value,
        "color": data.color,
        "brand": data.brand,
        "location": data.location,
        "latitude": data.latitude,
        "longitude": data.longitude,
        "date_lost_or_found": data.date_lost_or_found,
        "contact_name": data.contact_name,
        "contact_email": data.contact_email,
        "owner_id": owner_id,
        "status": "active",
        "verification_questions": data.verification_questions or [],
        "private_details": data.private_details,
        "image_url": None,
    }


@items_router.post("/lost", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
async def report_lost_item(
    data: ItemCreate,
    current_user: Optional[dict] = Depends(get_optional_current_user),
) -> ItemResponse:
    owner_id = current_user["id"] if current_user else None
    saved = await crud.save_item(_build_item_document(data, "lost", owner_id=owner_id))
    return ItemResponse(**saved)


@items_router.post("/found", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
async def report_found_item(
    data: ItemCreate,
    current_user: Optional[dict] = Depends(get_optional_current_user),
) -> ItemResponse:
    owner_id = current_user["id"] if current_user else None
    saved = await crud.save_item(_build_item_document(data, "found", owner_id=owner_id))
    return ItemResponse(**saved)


@items_router.get("", response_model=ItemListResponse)
async def get_all_items(type: Optional[str] = Query(default=None)) -> ItemListResponse:
    if type is not None and type not in ("lost", "found"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameter 'type' must be 'lost' or 'found'.",
        )
    items = await crud.get_all_items(item_type=type)
    return ItemListResponse(count=len(items), items=[ItemResponse(**i) for i in items])


@items_router.get("/{item_id}", response_model=ItemResponse)
async def get_item_by_id(item_id: str) -> ItemResponse:
    item = await crud.get_item_by_id(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item found with id '{item_id}'.",
        )
    return ItemResponse(**item)


@items_router.post("/{item_id}/image", response_model=ItemResponse)
async def upload_item_image(
    item_id: str,
    file: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_optional_current_user),
) -> ItemResponse:
    item = await crud.get_item_by_id(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item found with id '{item_id}'.",
        )

    if item.get("owner_id"):
        if not current_user or (current_user["id"] != item["owner_id"] and current_user.get("role") != "admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not authorized to modify this item report.",
            )

    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.content_type not in ALLOWED_CONTENT_TYPES or (ext and ext not in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only JPEG, PNG, and WEBP images are supported.",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds the 5 MB limit.",
        )

    try:
        secure_url = await upload_image_bytes(file_bytes, file.filename or "image")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload image to cloud storage.",
        )

    updated_item = await crud.update_item_image(item_id, secure_url)
    if updated_item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item found with id '{item_id}'.",
        )

    return ItemResponse(**updated_item)


@items_router.get("/{item_id}/matches", response_model=MatchResponse)
async def get_item_matches(
    item_id: str,
    threshold: float = Query(default=0.3, ge=0.0, le=1.0),
    limit: int = Query(default=5, ge=1, le=20),
) -> MatchResponse:
    item = await crud.get_item_by_id(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item found with id '{item_id}'.",
        )

    opposite_type = "found" if item["item_type"] == "lost" else "lost"
    candidates = await crud.get_all_items(item_type=opposite_type)

    matches = await find_matches(
        target_item=item,
        candidates=candidates,
        threshold=threshold,
        top_k=limit,
    )

    return MatchResponse(
        target_item_id=item["id"],
        target_item_type=item["item_type"],
        target_title=item["title"],
        threshold=threshold,
        total_candidates=len(candidates),
        matched_count=len(matches),
        matches=[MatchItem(**m) for m in matches],
    )


@items_router.get("/{item_id}/image-matches", response_model=ImageMatchResponse)
async def get_item_image_matches(
    item_id: str,
    threshold: float = Query(default=0.2, ge=0.0, le=1.0),
    limit: int = Query(default=5, ge=1, le=20),
) -> ImageMatchResponse:
    item = await crud.get_item_by_id(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item found with id '{item_id}'.",
        )

    target_image_url = item.get("image_url")
    if not target_image_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Item '{item_id}' does not have an image uploaded. Image matching requires an image.",
        )

    opposite_type = "found" if item["item_type"] == "lost" else "lost"
    candidates = await crud.get_all_items(item_type=opposite_type)

    try:
        valid_count, matches = await process_image_matching(
            target_image_url=target_image_url,
            candidates=candidates,
            threshold=threshold,
            top_k=limit,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing image matching.",
        )

    return ImageMatchResponse(
        target_item_id=item["id"],
        target_item_type=item["item_type"],
        target_title=item["title"],
        target_image_url=target_image_url,
        threshold=threshold,
        total_candidates=len(candidates),
        valid_image_candidates=valid_count,
        matched_count=len(matches),
        matches=[ImageMatchItem(**m) for m in matches],
    )


@items_router.get("/{item_id}/attribute-matches", response_model=AttributeMatchResponse)
async def get_item_attribute_matches(
    item_id: str,
    threshold: float = Query(default=0.3, ge=0.0, le=1.0),
    limit: int = Query(default=5, ge=1, le=20),
) -> AttributeMatchResponse:
    item = await crud.get_item_by_id(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item found with id '{item_id}'.",
        )

    opposite_type = "found" if item["item_type"] == "lost" else "lost"
    candidates = await crud.get_all_items(item_type=opposite_type)

    matches = await find_attribute_matches(
        target_item=item,
        candidates=candidates,
        threshold=threshold,
        top_k=limit,
    )

    return AttributeMatchResponse(
        target_item_id=item["id"],
        target_item_type=item["item_type"],
        target_title=item["title"],
        threshold=threshold,
        total_candidates=len(candidates),
        matched_count=len(matches),
        matches=[AttributeMatchItem(**m) for m in matches],
    )


@items_router.get("/{item_id}/location-time-matches", response_model=LocationTimeMatchResponse)
async def get_item_location_time_matches(
    item_id: str,
    threshold: float = Query(default=0.2, ge=0.0, le=1.0),
    max_days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=5, ge=1, le=20),
) -> LocationTimeMatchResponse:
    item = await crud.get_item_by_id(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item found with id '{item_id}'.",
        )

    opposite_type = "found" if item["item_type"] == "lost" else "lost"
    candidates = await crud.get_all_items(item_type=opposite_type)

    matches = await find_location_time_matches(
        target_item=item,
        candidates=candidates,
        threshold=threshold,
        max_days=max_days,
        top_k=limit,
    )

    return LocationTimeMatchResponse(
        target_item_id=item["id"],
        target_item_type=item["item_type"],
        target_title=item["title"],
        target_location=item.get("location", ""),
        target_date=item.get("date_lost_or_found", ""),
        threshold=threshold,
        max_days=max_days,
        total_candidates=len(candidates),
        matched_count=len(matches),
        matches=[LocationTimeMatchItem(**m) for m in matches],
    )


@items_router.get("/{item_id}/hybrid-matches", response_model=HybridMatchResponse)
async def get_item_hybrid_matches(
    item_id: str,
    threshold: float = Query(default=0.35, ge=0.0, le=1.0),
    limit: int = Query(default=5, ge=1, le=20),
) -> HybridMatchResponse:
    item = await crud.get_item_by_id(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No item found with id '{item_id}'.",
        )

    opposite_type = "found" if item["item_type"] == "lost" else "lost"
    candidates = await crud.get_all_items(item_type=opposite_type)

    matches = await find_hybrid_matches(
        target_item=item,
        candidates=candidates,
        threshold=threshold,
        top_k=limit,
    )

    return HybridMatchResponse(
        target_item_id=item["id"],
        target_item_type=item["item_type"],
        target_title=item["title"],
        threshold=threshold,
        weights=DEFAULT_WEIGHTS,
        total_candidates=len(candidates),
        matched_count=len(matches),
        matches=[HybridMatchItem(**m) for m in matches],
    )


# ============================================================================
# 3. MATCHES ROUTER
# ============================================================================

matches_router = APIRouter(prefix="/matches", tags=["Matches"])


@matches_router.get("", response_model=MatchListResponse)
async def list_matches(
    match_status: Optional[MatchStatus] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
):
    if current_user.get("role") == "admin":
        total, items = await crud.list_matches(
            item_ids=None,
            match_status=match_status.value if match_status else None,
            skip=skip,
            limit=limit,
        )
    else:
        user_items = await crud.get_items_by_owner(current_user["id"])
        item_ids = [it["id"] for it in user_items]
        total, items = await crud.list_matches(
            item_ids=item_ids,
            match_status=match_status.value if match_status else None,
            skip=skip,
            limit=limit,
        )

    return MatchListResponse(total=total, items=items)


@matches_router.get("/{match_id}", response_model=MatchRecordResponse)
async def get_match_detail(
    match_id: str,
    current_user: dict = Depends(get_current_user),
):
    match_record = await crud.get_match_by_id(match_id)
    if not match_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match record '{match_id}' not found.",
        )

    lost_item = await crud.get_item_by_id(match_record["lost_item_id"])
    found_item = await crud.get_item_by_id(match_record["found_item_id"])

    if not match_service.verify_user_match_access(current_user, match_record, lost_item, found_item):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to view this match record.",
        )

    return match_record


@matches_router.patch("/{match_id}/status", response_model=MatchRecordResponse)
async def update_match_status_endpoint(
    match_id: str,
    payload: MatchStatusUpdate,
    current_user: dict = Depends(get_current_user),
):
    updated = await match_service.update_status_with_lifecycle(
        match_id=match_id,
        new_status=payload.status.value,
        user=current_user,
    )
    return updated


@matches_router.post("/run/{item_id}")
async def run_and_persist_matches(
    item_id: str,
    threshold: float = Query(0.35, ge=0.0, le=1.0),
    limit: int = Query(10, ge=1, le=50),
    current_user: dict = Depends(get_current_user),
):
    target = await crud.get_item_by_id(item_id)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item '{item_id}' not found.",
        )

    if target.get("owner_id") and target.get("owner_id") != current_user["id"] and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to run matches for this item.",
        )

    count, matches = await match_service.run_and_persist_matches(
        item_id=item_id,
        threshold=threshold,
        limit=limit,
    )
    return {
        "item_id": item_id,
        "matches_persisted": count,
        "matches": matches,
    }


# ============================================================================
# 4. CLAIMS ROUTER
# ============================================================================

claims_router = APIRouter(prefix="/claims", tags=["Claims & Ownership Verification"])


@claims_router.post("", response_model=ClaimResponse, status_code=status.HTTP_201_CREATED)
async def submit_new_claim(
    payload: ClaimCreate,
    current_user: dict = Depends(get_current_user),
):
    created = await claim_service.submit_claim(user=current_user, claim_data=payload)
    return created


@claims_router.get("", response_model=ClaimListResponse)
async def list_user_claims(
    claim_status: Optional[ClaimStatus] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
):
    status_filter = claim_status.value if claim_status else None

    if current_user.get("role") == "admin":
        total, items = await crud.list_claims(
            claimant_id=None,
            item_ids=None,
            claim_status=status_filter,
            skip=skip,
            limit=limit,
        )
    else:
        user_items = await crud.get_items_by_owner(current_user["id"])
        item_ids = [it["id"] for it in user_items]
        total, items = await crud.list_claims(
            claimant_id=current_user["id"],
            item_ids=item_ids,
            claim_status=status_filter,
            skip=skip,
            limit=limit,
        )

    return ClaimListResponse(total=total, items=items)


@claims_router.get("/{claim_id}", response_model=ClaimResponse)
async def get_claim_detail(
    claim_id: str,
    current_user: dict = Depends(get_current_user),
):
    claim = await crud.get_claim_by_id(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )

    found_item = await crud.get_item_by_id(claim["found_item_id"])
    is_admin = current_user.get("role") == "admin"
    is_claimant = claim["claimant_id"] == current_user["id"]
    is_finder = found_item and found_item.get("owner_id") == current_user["id"]

    if not (is_admin or is_claimant or is_finder):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to view this claim.",
        )

    return claim


@claims_router.patch("/{claim_id}/decision", response_model=ClaimResponse)
async def decide_claim(
    claim_id: str,
    payload: ClaimDecision,
    current_user: dict = Depends(get_current_user),
):
    updated = await claim_service.review_claim(
        user=current_user,
        claim_id=claim_id,
        decision=payload.decision.value,
        note=payload.note,
    )
    return updated


@claims_router.patch("/{claim_id}/cancel", response_model=ClaimResponse)
async def cancel_claim_request(
    claim_id: str,
    payload: Optional[ClaimCancel] = None,
    current_user: dict = Depends(get_current_user),
):
    reason = payload.reason if payload else None
    updated = await claim_service.cancel_claim(
        user=current_user,
        claim_id=claim_id,
        reason=reason,
    )
    return updated


@claims_router.post("/{claim_id}/recover")
async def finalize_recovery(
    claim_id: str,
    note: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    result = await claim_service.complete_recovery(
        user=current_user,
        claim_id=claim_id,
        note=note,
    )
    return result


# ============================================================================
# 5. NOTIFICATIONS ROUTER
# ============================================================================

notifications_router = APIRouter(prefix="/notifications", tags=["Notifications"])


@notifications_router.get("", response_model=NotificationListResponse)
async def list_notifications(
    unread_only: bool = False,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
):
    total, items = await crud.list_notifications(
        user_id=current_user["id"],
        unread_only=unread_only,
        skip=skip,
        limit=limit,
    )
    unread_count = await crud.count_unread_notifications(current_user["id"])
    return NotificationListResponse(
        total=total,
        unread_count=unread_count,
        items=items,
    )


@notifications_router.patch("/{notification_id}/read", response_model=NotificationResponse)
async def mark_single_read(
    notification_id: str,
    current_user: dict = Depends(get_current_user),
):
    updated = await crud.mark_notification_read(
        notif_id=notification_id,
        user_id=current_user["id"],
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notification '{notification_id}' not found.",
        )
    return updated


@notifications_router.patch("/read-all")
async def mark_all_read(
    current_user: dict = Depends(get_current_user),
):
    count = await crud.mark_all_notifications_read(current_user["id"])
    return {
        "message": f"Marked {count} notifications as read.",
        "updated_count": count,
    }


# Backwards compatibility wrappers
class _RouterWrapper:
    def __init__(self, router):
        self.router = router

auth = _RouterWrapper(auth_router)
items = _RouterWrapper(items_router)
matches = _RouterWrapper(matches_router)
notifications = _RouterWrapper(notifications_router)
claims = _RouterWrapper(claims_router)
