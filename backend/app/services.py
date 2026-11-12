import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List, Dict, Any, Tuple
import cloudinary
import cloudinary.uploader
from fastapi import HTTPException, status

from app.core import settings
from app.database import (
    save_notification,
    has_notification,
    get_match_by_id,
    get_item_by_id,
    update_match_status,
    upsert_match_record,
    get_all_items,
    get_approved_claim_for_item,
    get_active_claim_by_claimant_and_item,
    save_claim,
    get_claim_by_id,
    update_claim_status,
    update_item_status,
)
from app.schemas import ClaimCreate
from app.matching import find_hybrid_matches


# ============================================================================
# 1. CLOUDINARY IMAGE STORAGE SERVICE
# ============================================================================

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True,
)


async def upload_image_bytes(file_bytes: bytes, filename: str) -> str:
    res = await asyncio.to_thread(
        cloudinary.uploader.upload,
        file_bytes,
        folder="findit_ai/items",
        resource_type="image",
    )
    return res.get("secure_url")


# ============================================================================
# 2. SMART NOTIFICATION & EMAIL SERVICE
# ============================================================================

def _send_email_sync(to_email: str, subject: str, body: str) -> bool:
    if not settings.SMTP_ENABLED or not to_email:
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.SMTP_FROM_EMAIL
        msg["To"] = to_email
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=5) as server:
            server.starttls()
            if settings.SMTP_USERNAME and settings.SMTP_PASSWORD:
                server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM_EMAIL, [to_email], msg.as_string())
        return True
    except Exception:
        return False


async def send_email_notification(to_email: str, subject: str, body: str) -> bool:
    if not settings.SMTP_ENABLED or not to_email:
        return False
    return await asyncio.to_thread(_send_email_sync, to_email, subject, body)


async def create_smart_notification(
    user_id: str,
    title: str,
    message: str,
    notif_type: str,
    reference_id: str,
    recipient_email: Optional[str] = None,
) -> Optional[dict]:
    if await has_notification(user_id, notif_type, reference_id):
        return None

    notif_data = {
        "user_id": user_id,
        "title": title,
        "message": message,
        "type": notif_type,
        "reference_id": reference_id,
    }
    saved = await save_notification(notif_data)

    if settings.SMTP_ENABLED and recipient_email:
        asyncio.create_task(
            send_email_notification(recipient_email, title, message)
        )

    return saved


# ============================================================================
# 3. MATCH LIFECYCLE MANAGEMENT SERVICE
# ============================================================================

VALID_TRANSITIONS = {
    "suggested": {"viewed", "interested", "rejected"},
    "viewed": {"interested", "rejected"},
    "interested": {"claim_pending", "rejected"},
    "claim_pending": {"verified", "rejected", "interested"},
    "verified": {"recovered", "rejected"},
    "rejected": {"viewed", "interested"},
    "recovered": set(),
}


def can_transition(current_status: str, target_status: str) -> bool:
    allowed = VALID_TRANSITIONS.get(current_status, set())
    return target_status in allowed


def verify_user_match_access(user: dict, match: dict, lost_item: Optional[dict], found_item: Optional[dict]) -> bool:
    if user.get("role") == "admin":
        return True
    user_id = user["id"]
    if lost_item and lost_item.get("owner_id") == user_id:
        return True
    if found_item and found_item.get("owner_id") == user_id:
        return True
    return False


async def update_status_with_lifecycle(match_id: str, new_status: str, user: dict) -> dict:
    match = await get_match_by_id(match_id)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Match '{match_id}' not found.",
        )

    lost_item = await get_item_by_id(match["lost_item_id"])
    found_item = await get_item_by_id(match["found_item_id"])

    if not verify_user_match_access(user, match, lost_item, found_item):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to update this match.",
        )

    current_status = match.get("status", "suggested")
    if not can_transition(current_status, new_status):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid match status transition from '{current_status}' to '{new_status}'.",
        )

    updated = await update_match_status(match_id, new_status)

    if new_status == "interested":
        user_id = user["id"]
        other_owner_id = None
        if lost_item and lost_item.get("owner_id") == user_id and found_item:
            other_owner_id = found_item.get("owner_id")
        elif found_item and found_item.get("owner_id") == user_id and lost_item:
            other_owner_id = lost_item.get("owner_id")

        if other_owner_id:
            await create_smart_notification(
                user_id=other_owner_id,
                title="Match Interest Expressed",
                message="A user expressed interest in potential match for your item.",
                notif_type="match_interest",
                reference_id=match_id,
            )

    return updated


async def run_and_persist_matches(
    item_id: str,
    threshold: float = 0.35,
    limit: int = 10,
) -> Tuple[int, List[dict]]:
    target = await get_item_by_id(item_id)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item '{item_id}' not found.",
        )

    opposite_type = "found" if target["item_type"] == "lost" else "lost"
    candidates = await get_all_items(item_type=opposite_type)

    if not candidates:
        return 0, []

    raw_matches = await find_hybrid_matches(
        target_item=target,
        candidates=candidates,
        threshold=threshold,
        top_k=limit,
    )

    persisted_matches = []
    for m in raw_matches:
        if target["item_type"] == "lost":
            lost_id, found_id = target["id"], m["id"]
            cand_item = next((c for c in candidates if c["id"] == m["id"]), None)
            found_owner = cand_item.get("owner_id") if cand_item else None
            lost_owner = target.get("owner_id")
        else:
            lost_id, found_id = m["id"], target["id"]
            cand_item = next((c for c in candidates if c["id"] == m["id"]), None)
            lost_owner = cand_item.get("owner_id") if cand_item else None
            found_owner = target.get("owner_id")

        match_doc = {
            "lost_item_id": lost_id,
            "found_item_id": found_id,
            "hybrid_score": m["hybrid_score"],
            "scores": {
                "text_score": m.get("text_score"),
                "image_score": m.get("image_score"),
                "attribute_score": m.get("attribute_score"),
                "location_score": m.get("location_score"),
                "time_score": m.get("time_score"),
            },
            "explanation": m.get("explanation", ""),
        }

        persisted, is_new = await upsert_match_record(match_doc)
        persisted_matches.append(persisted)

        if is_new and m["hybrid_score"] >= settings.MATCH_NOTIFICATION_THRESHOLD:
            msg = f"Potential match discovered with {int(m['hybrid_score'] * 100)}% match score."
            if lost_owner:
                await create_smart_notification(
                    user_id=lost_owner,
                    title="Promising Match Found",
                    message=msg,
                    notif_type="potential_match",
                    reference_id=persisted["id"],
                )
            if found_owner and found_owner != lost_owner:
                await create_smart_notification(
                    user_id=found_owner,
                    title="Promising Match Found",
                    message=msg,
                    notif_type="potential_match",
                    reference_id=persisted["id"],
                )

    return len(persisted_matches), persisted_matches


# ============================================================================
# 4. CLAIMS & OWNERSHIP VERIFICATION WORKFLOW SERVICE
# ============================================================================

VALID_CLAIM_TRANSITIONS = {
    "submitted": {"under_review", "additional_information_required", "approved", "rejected", "cancelled"},
    "under_review": {"additional_information_required", "approved", "rejected", "cancelled"},
    "additional_information_required": {"under_review", "approved", "rejected", "cancelled"},
    "approved": {"rejected", "cancelled"},
    "rejected": set(),
    "cancelled": set(),
}


async def submit_claim(user: dict, claim_data: ClaimCreate) -> dict:
    found_item = await get_item_by_id(claim_data.found_item_id)
    if not found_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Found item '{claim_data.found_item_id}' not found.",
        )

    if found_item.get("item_type") != "found":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Claims can only be filed against items reported as 'found'.",
        )

    if found_item.get("status") == "recovered":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This item has already been marked as recovered.",
        )

    if found_item.get("owner_id") and found_item.get("owner_id") == user["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot submit a claim on an item that you reported as found.",
        )

    approved_claim = await get_approved_claim_for_item(claim_data.found_item_id)
    if approved_claim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This item already has an approved claim.",
        )

    active_claim = await get_active_claim_by_claimant_and_item(
        claimant_id=user["id"],
        found_item_id=claim_data.found_item_id,
    )
    if active_claim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already have an active claim pending or approved for this item.",
        )

    if claim_data.lost_item_id:
        lost_item = await get_item_by_id(claim_data.lost_item_id)
        if not lost_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lost item '{claim_data.lost_item_id}' not found.",
            )
        if lost_item.get("item_type") != "lost":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Referenced lost item must be of type 'lost'.",
            )
        if lost_item.get("owner_id") and lost_item.get("owner_id") != user["id"] and user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not own the referenced lost item report.",
            )

    if claim_data.match_id:
        match_record = await get_match_by_id(claim_data.match_id)
        if not match_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Match record '{claim_data.match_id}' not found.",
            )
        if match_record.get("found_item_id") != claim_data.found_item_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Match record found_item_id does not match the claim found_item_id.",
            )
        try:
            await update_match_status(claim_data.match_id, "claim_pending")
        except Exception:
            pass

    claim_doc = {
        "claimant_id": user["id"],
        "found_item_id": claim_data.found_item_id,
        "lost_item_id": claim_data.lost_item_id,
        "match_id": claim_data.match_id,
        "claim_description": claim_data.claim_description,
        "answers": claim_data.answers or {},
        "status": "submitted",
        "claim_status": "submitted",
    }

    created = await save_claim(claim_doc)

    finder_id = found_item.get("owner_id")
    if finder_id:
        await create_smart_notification(
            user_id=finder_id,
            title="New Ownership Claim Received",
            message=f"A user has submitted an ownership claim for your found item '{found_item.get('title')}'.",
            notif_type="claim_submitted",
            reference_id=created["id"],
        )

    return created


async def review_claim(
    user: dict,
    claim_id: str,
    decision: str,
    note: Optional[str] = None,
) -> dict:
    claim = await get_claim_by_id(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )

    found_item = await get_item_by_id(claim["found_item_id"])
    if not found_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated found item not found.",
        )

    is_admin = user.get("role") == "admin"
    is_finder = found_item.get("owner_id") == user["id"]

    if claim.get("claimant_id") == user["id"] and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Claimants cannot review or decide their own claims.",
        )

    if not is_admin and not is_finder:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the finder who reported this item (or an administrator) can review this claim.",
        )

    current_status = claim.get("status") or claim.get("claim_status", "submitted")
    allowed_transitions = VALID_CLAIM_TRANSITIONS.get(current_status, set())
    if decision not in allowed_transitions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot transition claim from '{current_status}' to '{decision}'.",
        )

    if decision == "approved":
        existing_approved = await get_approved_claim_for_item(claim["found_item_id"])
        if existing_approved and existing_approved["id"] != claim_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Another claim for this item has already been approved. Conflicting approvals are blocked.",
            )

    updated = await update_claim_status(
        claim_id=claim_id,
        new_status=decision,
        action=f"decision_{decision}",
        note=note,
        actor_id=user["id"],
    )

    if decision == "approved" and claim.get("match_id"):
        try:
            await update_match_status(claim["match_id"], "verified")
        except Exception:
            pass

    claimant_id = claim.get("claimant_id")
    if claimant_id:
        title = f"Claim Status Updated: {decision.replace('_', ' ').title()}"
        msg = f"Your claim for '{found_item.get('title')}' is now '{decision}'."
        if note:
            msg += f" Note: {note}"
        await create_smart_notification(
            user_id=claimant_id,
            title=title,
            message=msg,
            notif_type="claim_decision",
            reference_id=claim_id,
        )

    return updated


async def cancel_claim(user: dict, claim_id: str, reason: Optional[str] = None) -> dict:
    claim = await get_claim_by_id(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )

    is_admin = user.get("role") == "admin"
    is_claimant = claim.get("claimant_id") == user["id"]

    if not is_admin and not is_claimant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the claimant or an administrator can cancel this claim.",
        )

    current_status = claim.get("status") or claim.get("claim_status")
    if current_status in ["cancelled", "recovered"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Claim is already {current_status}.",
        )

    updated = await update_claim_status(
        claim_id=claim_id,
        new_status="cancelled",
        action="cancelled_by_user",
        note=reason,
        actor_id=user["id"],
    )
    return updated


async def complete_recovery(user: dict, claim_id: str, note: Optional[str] = None) -> dict:
    claim = await get_claim_by_id(claim_id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )

    current_status = claim.get("status") or claim.get("claim_status")
    if current_status != "approved":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only an approved claim can be marked as recovered (current status is '{current_status}').",
        )

    found_item = await get_item_by_id(claim["found_item_id"])
    if not found_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Associated found item not found.",
        )

    is_admin = user.get("role") == "admin"
    is_finder = found_item.get("owner_id") == user["id"]

    if not is_admin and not is_finder:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the item finder or an administrator can mark recovery as complete.",
        )

    await update_item_status(claim["found_item_id"], "recovered")

    if claim.get("lost_item_id"):
        await update_item_status(claim["lost_item_id"], "recovered")

    if claim.get("match_id"):
        try:
            await update_match_status(claim["match_id"], "recovered")
        except Exception:
            pass

    updated_claim = await update_claim_status(
        claim_id=claim_id,
        new_status="approved",
        action="recovery_completed",
        note=note or "Item successfully returned to owner and marked recovered.",
        actor_id=user["id"],
    )

    claimant_id = claim.get("claimant_id")
    if claimant_id:
        await create_smart_notification(
            user_id=claimant_id,
            title="Item Officially Recovered",
            message=f"The recovery of '{found_item.get('title')}' is officially complete! Thank you for using FindIt AI.",
            notif_type="item_recovered",
            reference_id=claim_id,
        )

    return {
        "message": "Recovery successfully completed and items marked recovered.",
        "claim": updated_claim,
        "found_item_id": claim["found_item_id"],
        "lost_item_id": claim.get("lost_item_id"),
        "match_id": claim.get("match_id"),
    }


class _MatchService:
    can_transition = staticmethod(can_transition)
    verify_user_match_access = staticmethod(verify_user_match_access)
    update_status_with_lifecycle = staticmethod(update_status_with_lifecycle)
    run_and_persist_matches = staticmethod(run_and_persist_matches)

match_service = _MatchService()

class _ClaimService:
    submit_claim = staticmethod(submit_claim)
    review_claim = staticmethod(review_claim)
    cancel_claim = staticmethod(cancel_claim)
    complete_recovery = staticmethod(complete_recovery)

claim_service = _ClaimService()

