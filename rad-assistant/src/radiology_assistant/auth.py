"""
Authentication & Authorization for the Radiology Assistant API.

Provides JWT token creation/verification and role-based access control
using FastAPI's OAuth2 security utilities.
"""

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, List

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import Config

logger = logging.getLogger(__name__)

# Password hashing context (pbkdf2_sha256)
# bcrypt had issues in this environment, pbkdf2_sha256 is a secure alternative.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# OAuth2 token URL — used by Swagger UI
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token", auto_error=False)


# ---------------------------------------------------------------------------
# Role Model
# ---------------------------------------------------------------------------

class UserRole(str, Enum):
    """Application-level roles for RBAC."""
    RADIOLOGIST = "radiologist"
    ADMIN = "admin"
    PATIENT_VIEWER = "patient_viewer"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class TokenData(BaseModel):
    """Payload stored inside the JWT."""
    sub: str                        # subject — typically username or user_id
    role: UserRole
    exp: Optional[datetime] = None


class TokenResponse(BaseModel):
    """Response body returned from the login endpoint."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int                 # seconds


class LoginRequest(BaseModel):
    """Body for the /v1/auth/token endpoint."""
    username: str
    password: str


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Return bcrypt hash of a plain-text password."""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the bcrypt hash."""
    return pwd_context.verify(plain, hashed)


def create_access_token(
    sub: str,
    role: UserRole,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        sub: Subject (username or user_id).
        role: The user's role to embed in the token.
        expires_delta: Override default expiry from Config.

    Returns:
        Signed JWT string.
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=Config.JWT_EXPIRE_MINUTES)
    )
    payload = {
        "sub": sub,
        "role": role.value,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)
    logger.debug("Created JWT for sub=%s role=%s exp=%s", sub, role, expire)
    return token


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT. Raises HTTPException 401 on any error.

    Args:
        token: Raw JWT string.

    Returns:
        Decoded TokenData.

    Raises:
        HTTPException 401: If token is missing, expired, or malformed.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            Config.JWT_SECRET_KEY,
            algorithms=[Config.JWT_ALGORITHM],
        )
        sub: str = payload.get("sub")
        role_str: str = payload.get("role")
        if sub is None or role_str is None:
            raise credentials_exception
        role = UserRole(role_str)
        return TokenData(sub=sub, role=role)
    except JWTError as e:
        logger.warning("JWT decode error: %s", e)
        raise credentials_exception


# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------

def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> TokenData:
    """
    FastAPI dependency: validates the Bearer token and returns TokenData.
    If no token is provided, raises 401.
    """
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(token)


def require_role(*allowed_roles: UserRole):
    """
    Dependency factory for role-based access control.

    Usage:
        @app.get("/admin", dependencies=[Depends(require_role(UserRole.ADMIN))])

    Raises:
        HTTPException 403: If the current user's role is not in allowed_roles.
    """
    def _check(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}",
            )
        return current_user
    return _check


# ---------------------------------------------------------------------------
# In-Memory User Store (replace with DB in production)
# ---------------------------------------------------------------------------
# This is seeded with 3 demo accounts so the API can be tested immediately
# without a running database. Replace with SQLite/Postgres user lookup.

_DEMO_USERS: dict = {
    "admin": {
        "hashed_password": hash_password("admin123"),
        "role": UserRole.ADMIN,
    },
    "dr_smith": {
        "hashed_password": hash_password("radiologist123"),
        "role": UserRole.RADIOLOGIST,
    },
    "patient_viewer": {
        "hashed_password": hash_password("viewer123"),
        "role": UserRole.PATIENT_VIEWER,
    },
}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Verify username + password against the user store.
    Returns user dict on success, None on failure.
    """
    user = _DEMO_USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return {"username": username, **user}
