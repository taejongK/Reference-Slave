from fastapi import FastAPI
from router import (
    research_router,
    user_router
)

app = FastAPI()

app.include_router(user_router, prefix="/user", tags=["user"])
app.include_router(research_router, prefix="/research", tags=["research"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
