from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
from bson import ObjectId
from pymongo import AsyncMongoClient, ReturnDocument
from app.core import settings
from app.schemas import ItemInDB

# ============================================================================
# MONGODB CLIENT & CONNECTION MANAGEMENT
# ============================================================================

client: Optional[AsyncMongoClient] = None


def get_database():
    return client[settings.DB_NAME]


async def connect_db():
    global client
    client = AsyncMongoClient(
        settings.MONGODB_URI,
        tls=True,
        tlsAllowInvalidCertificates=False,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=5000,
        retryWrites=True,
    )
    try:
        await client.admin.command("ping")
        print(f"Connected to MongoDB — database: '{settings.DB_NAME}'")
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to MongoDB Atlas on startup ({e}). If your IP changed, ensure your current IP is whitelisted in MongoDB Atlas Network Access.")


async def close_db():
    global client
    if client:
        await client.close()
        print("MongoDB connection closed.")


# ============================================================================
# IN-MEMORY FALLBACK STORAGE (FOR UNIT TESTS / STANDALONE)
# ============================================================================

_lost_items: list[ItemInDB] = []
_found_items: list[ItemInDB] = []


def _get_store(item_type: str) -> list[ItemInDB]:
    if item_type == "lost":
        return _lost_items
    return _found_items


def save_item_in_memory(item: ItemInDB) -> ItemInDB:
    store = _get_store(item.item_type)
    store.append(item)
    return item


def get_all_items_in_memory(item_type: Optional[str] = None) -> list[ItemInDB]:
    if item_type == "lost":
        return list(_lost_items)
    if item_type == "found":
        return list(_found_items)
    all_items = _lost_items + _found_items
    all_items.sort(key=lambda x: x.created_at, reverse=True)
    return all_items


def get_item_by_id_in_memory(item_id: str) -> Optional[ItemInDB]:
    for item in _lost_items + _found_items:
        if item.id == item_id:
            return item
    return None


def get_counts_in_memory() -> dict:
    return {
        "lost": len(_lost_items),
        "found": len(_found_items),
        "total": len(_lost_items) + len(_found_items),
    }


# ============================================================================
# MONGODB CRUD OPERATIONS
# ============================================================================

ITEMS_COL = "items"
USERS_COL = "users"
MATCHES_COL = "matches"
NOTIFS_COL = "notifications"
CLAIMS_COL = "claims"


def _serialize(doc: Optional[dict]) -> Optional[dict]:
    if not doc:
        return None
    doc = dict(doc)
    if "_id" in doc:
        doc["id"] = str(doc.pop("_id"))
    return doc


def _to_oid(val: str) -> Optional[ObjectId]:
    try:
        return ObjectId(val)
    except Exception:
        return None


# USERS
async def save_user(user_data: dict) -> dict:
    db = get_database()
    result = await db[USERS_COL].insert_one(user_data)
    created = await db[USERS_COL].find_one({"_id": result.inserted_id})
    return _serialize(created)


async def get_user_by_email(email: str) -> Optional[dict]:
    db = get_database()
    user = await db[USERS_COL].find_one({"email": email.lower().strip()})
    return _serialize(user)


async def get_user_by_id(user_id: str) -> Optional[dict]:
    oid = _to_oid(user_id)
    if not oid:
        return None
    db = get_database()
    user = await db[USERS_COL].find_one({"_id": oid})
    return _serialize(user)


# ITEMS
async def save_item(item_data: dict) -> dict:
    db = get_database()
    result = await db[ITEMS_COL].insert_one(item_data)
    created = await db[ITEMS_COL].find_one({"_id": result.inserted_id})
    return _serialize(created)


async def get_all_items(item_type: Optional[str] = None) -> list[dict]:
    db = get_database()
    query = {"item_type": item_type} if item_type else {}
    cursor = db[ITEMS_COL].find(query).sort("created_at", -1)
    return [_serialize(doc) async for doc in cursor]


async def get_item_by_id(item_id: str) -> Optional[dict]:
    oid = _to_oid(item_id)
    if not oid:
        return None
    db = get_database()
    doc = await db[ITEMS_COL].find_one({"_id": oid})
    return _serialize(doc)


async def update_item_image(item_id: str, image_url: str) -> Optional[dict]:
    oid = _to_oid(item_id)
    if not oid:
        return None
    db = get_database()
    doc = await db[ITEMS_COL].find_one_and_update(
        {"_id": oid},
        {"$set": {"image_url": image_url}},
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc)


async def update_item_status(item_id: str, status: str) -> Optional[dict]:
    oid = _to_oid(item_id)
    if not oid:
        return None
    db = get_database()
    doc = await db[ITEMS_COL].find_one_and_update(
        {"_id": oid},
        {"$set": {"status": status, "updated_at": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc)


async def get_items_by_owner(user_id: str) -> list[dict]:
    db = get_database()
    cursor = db[ITEMS_COL].find({"owner_id": user_id}).sort("created_at", -1)
    return [_serialize(doc) async for doc in cursor]


async def get_counts() -> dict:
    db = get_database()
    lost = await db[ITEMS_COL].count_documents({"item_type": "lost"})
    found = await db[ITEMS_COL].count_documents({"item_type": "found"})
    return {"lost": lost, "found": found, "total": lost + found}


# MATCHES
async def upsert_match_record(match_data: dict) -> Tuple[dict, bool]:
    db = get_database()
    lost_id = match_data["lost_item_id"]
    found_id = match_data["found_item_id"]

    existing = await db[MATCHES_COL].find_one({
        "lost_item_id": lost_id,
        "found_item_id": found_id,
    })

    now = datetime.now(timezone.utc)
    if existing:
        update_fields = {
            "hybrid_score": match_data["hybrid_score"],
            "scores": match_data.get("scores", {}),
            "explanation": match_data.get("explanation", ""),
            "updated_at": now,
        }
        updated = await db[MATCHES_COL].find_one_and_update(
            {"_id": existing["_id"]},
            {"$set": update_fields},
            return_document=ReturnDocument.AFTER,
        )
        return _serialize(updated), False
    else:
        doc = dict(match_data)
        doc["status"] = doc.get("status", "suggested")
        doc["created_at"] = now
        doc["updated_at"] = now
        res = await db[MATCHES_COL].insert_one(doc)
        created = await db[MATCHES_COL].find_one({"_id": res.inserted_id})
        return _serialize(created), True


async def get_match_by_id(match_id: str) -> Optional[dict]:
    oid = _to_oid(match_id)
    if not oid:
        return None
    db = get_database()
    doc = await db[MATCHES_COL].find_one({"_id": oid})
    return _serialize(doc)


async def get_match_by_pair(lost_id: str, found_id: str) -> Optional[dict]:
    db = get_database()
    doc = await db[MATCHES_COL].find_one({
        "lost_item_id": lost_id,
        "found_item_id": found_id,
    })
    return _serialize(doc)


async def update_match_status(match_id: str, new_status: str) -> Optional[dict]:
    oid = _to_oid(match_id)
    if not oid:
        return None
    db = get_database()
    doc = await db[MATCHES_COL].find_one_and_update(
        {"_id": oid},
        {"$set": {"status": new_status, "updated_at": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc)


async def list_matches(
    item_ids: Optional[List[str]] = None,
    match_status: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
) -> Tuple[int, List[dict]]:
    db = get_database()
    query = {}
    if item_ids is not None:
        query["$or"] = [
            {"lost_item_id": {"$in": item_ids}},
            {"found_item_id": {"$in": item_ids}},
        ]
    if match_status:
        query["status"] = match_status

    total = await db[MATCHES_COL].count_documents(query)
    cursor = db[MATCHES_COL].find(query).sort("hybrid_score", -1).skip(skip).limit(limit)
    items = [_serialize(doc) async for doc in cursor]
    return total, items


# NOTIFICATIONS
async def save_notification(notif_data: dict) -> dict:
    db = get_database()
    doc = dict(notif_data)
    doc["is_read"] = False
    doc["created_at"] = datetime.now(timezone.utc)
    res = await db[NOTIFS_COL].insert_one(doc)
    created = await db[NOTIFS_COL].find_one({"_id": res.inserted_id})
    return _serialize(created)


async def has_notification(user_id: str, notif_type: str, reference_id: str) -> bool:
    db = get_database()
    count = await db[NOTIFS_COL].count_documents({
        "user_id": user_id,
        "type": notif_type,
        "reference_id": reference_id,
    })
    return count > 0


async def list_notifications(
    user_id: str,
    unread_only: bool = False,
    skip: int = 0,
    limit: int = 50,
) -> Tuple[int, List[dict]]:
    db = get_database()
    query: Dict[str, Any] = {"user_id": user_id}
    if unread_only:
        query["is_read"] = False

    total = await db[NOTIFS_COL].count_documents(query)
    cursor = db[NOTIFS_COL].find(query).sort("created_at", -1).skip(skip).limit(limit)
    items = [_serialize(doc) async for doc in cursor]
    return total, items


async def count_unread_notifications(user_id: str) -> int:
    db = get_database()
    return await db[NOTIFS_COL].count_documents({"user_id": user_id, "is_read": False})


async def mark_notification_read(notif_id: str, user_id: str) -> Optional[dict]:
    oid = _to_oid(notif_id)
    if not oid:
        return None
    db = get_database()
    doc = await db[NOTIFS_COL].find_one_and_update(
        {"_id": oid, "user_id": user_id},
        {"$set": {"is_read": True}},
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc)


async def mark_all_notifications_read(user_id: str) -> int:
    db = get_database()
    res = await db[NOTIFS_COL].update_many(
        {"user_id": user_id, "is_read": False},
        {"$set": {"is_read": True}},
    )
    return res.modified_count


# CLAIMS
async def save_claim(claim_data: dict) -> dict:
    db = get_database()
    doc = dict(claim_data)
    now = datetime.now(timezone.utc)
    doc["created_at"] = now
    doc["updated_at"] = now
    st = doc.get("status") or doc.get("claim_status") or "submitted"
    doc["status"] = st
    doc["claim_status"] = st
    doc["audit_trail"] = [
        {"action": "submitted", "timestamp": now, "note": "Claim submitted by claimant"}
    ]
    res = await db[CLAIMS_COL].insert_one(doc)
    created = await db[CLAIMS_COL].find_one({"_id": res.inserted_id})
    return _serialize(created)


async def get_claim_by_id(claim_id: str) -> Optional[dict]:
    oid = _to_oid(claim_id)
    if not oid:
        return None
    db = get_database()
    doc = await db[CLAIMS_COL].find_one({"_id": oid})
    return _serialize(doc)


async def get_active_claim_by_claimant_and_item(claimant_id: str, found_item_id: str) -> Optional[dict]:
    db = get_database()
    doc = await db[CLAIMS_COL].find_one({
        "claimant_id": claimant_id,
        "found_item_id": found_item_id,
        "claim_status": {"$in": ["submitted", "under_review", "additional_information_required", "approved"]},
    })
    return _serialize(doc)


async def get_approved_claim_for_item(found_item_id: str) -> Optional[dict]:
    db = get_database()
    doc = await db[CLAIMS_COL].find_one({
        "found_item_id": found_item_id,
        "claim_status": "approved",
    })
    return _serialize(doc)


async def list_claims(
    claimant_id: Optional[str] = None,
    item_ids: Optional[List[str]] = None,
    claim_status: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
) -> Tuple[int, List[dict]]:
    db = get_database()
    query: Dict[str, Any] = {}

    if claimant_id and item_ids:
        query["$or"] = [
            {"claimant_id": claimant_id},
            {"found_item_id": {"$in": item_ids}},
            {"lost_item_id": {"$in": item_ids}},
        ]
    elif claimant_id:
        query["claimant_id"] = claimant_id
    elif item_ids:
        query["$or"] = [
            {"found_item_id": {"$in": item_ids}},
            {"lost_item_id": {"$in": item_ids}},
        ]

    if claim_status:
        query["claim_status"] = claim_status

    total = await db[CLAIMS_COL].count_documents(query)
    cursor = db[CLAIMS_COL].find(query).sort("created_at", -1).skip(skip).limit(limit)
    items = [_serialize(doc) async for doc in cursor]
    return total, items


async def update_claim_status(
    claim_id: str,
    new_status: str,
    action: str,
    note: Optional[str] = None,
    actor_id: Optional[str] = None,
) -> Optional[dict]:
    oid = _to_oid(claim_id)
    if not oid:
        return None
    db = get_database()
    now = datetime.now(timezone.utc)
    audit_entry = {
        "action": action,
        "new_status": new_status,
        "timestamp": now,
        "note": note or "",
        "actor_id": actor_id,
    }
    doc = await db[CLAIMS_COL].find_one_and_update(
        {"_id": oid},
        {
            "$set": {
                "claim_status": new_status,
                "status": new_status,
                "updated_at": now,
            },
            "$push": {"audit_trail": audit_entry},
        },
        return_document=ReturnDocument.AFTER,
    )
    return _serialize(doc)
