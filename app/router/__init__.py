from fastapi import APIRouter
from .research import research_router
from .user import user_router

__all__ = ["research_router", "user_router"]