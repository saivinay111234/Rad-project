"""
Middleware for the Radiology Assistant.

Includes:
1. AuditLoggingMiddleware: Logs all clinical API requests to the database.
"""

import time
import json
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .database import SessionLocal
from .db_models import AuditLogDB
from .auth import decode_token

logger = logging.getLogger(__name__)


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log clinical API requests to the AuditLogDB table.
    Captures user ID, endpoint, action, and timestamp.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # We only audit clinical endpoints (v1/...)
        path = request.url.path
        if not path.startswith("/v1/"):
            return await call_next(request)

        # Skip health, auth/token, and metrics to avoid noise
        if any(skip in path for skip in ["/health", "/auth/token", "/metrics"]):
            return await call_next(request)

        start_time = time.time()
        
        # Get user from JWT if present
        user_id = "anonymous"
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = decode_token(token)
                user_id = payload.get("sub", "unknown")
            except Exception:
                pass

        # Proceed with request
        response = await call_next(request)
        
        process_time = time.time() - start_time
        status_code = response.status_code

        # Save to DB in background (using a fresh session)
        try:
            db = SessionLocal()
            log_entry = AuditLogDB(
                user_id=user_id,
                action=request.method,
                resource=path,
                status_code=status_code,
                ip_address=request.client.host if request.client else "unknown"
            )
            db.add(log_entry)
            db.commit()
            db.close()
        except Exception:
            logger.exception("Failed to write audit log entry")
            # We don't fail the request if logging fails

        return response
