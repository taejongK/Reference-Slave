from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

user_router = APIRouter()

@user_router.get("/")
def get_root():
    return {"message": "Hello, World!"}


class CreateUser(BaseModel):
    email: str
    
class UpdateUser(BaseModel):
    current_email: str
    new_email: str
    
class DeleteUser(BaseModel):
    email: str
    
@user_router.post("/create")
def create_user(user: CreateUser):
    # TODO: Check if user already exists
    # TODO: Implement user creation
    if user.email == "":
        raise HTTPException(status_code=400, detail="Email is required")
    if "@" not in user.email or "." not in user.email:
        raise HTTPException(status_code=400, detail="Invalid email format")
    return user

@user_router.put("/update")
def update_user(user: UpdateUser):
    # TODO: Check if user exists
    # TODO: Implement user update
    if user.current_email == "":
        raise HTTPException(status_code=400, detail="Current email is required")
    if user.new_email == "":
        raise HTTPException(status_code=400, detail="New email is required")
    if "@" not in user.new_email or "." not in user.new_email:
        raise HTTPException(status_code=400, detail="Invalid email format")
    return user

@user_router.delete("/delete")
def delete_user(user: DeleteUser):
    # TODO: Check if user exists
    if user.email == "":
        raise HTTPException(status_code=400, detail="Email is required")
    return user

if __name__ == "__main__":
    # Test code
    test_user = CreateUser(email="test@example.com")
    print(f"Creating test user: {test_user}")
    
    created_user = create_user(test_user)
    print(f"Created user: {created_user}")
    
    update_data = UpdateUser(current_email="test@example.com", new_email="updated@example.com")
    print(f"Updating user: {update_data}")
    
    updated_user = update_user(update_data)
    print(f"Updated user: {updated_user}")
    
    delete_data = DeleteUser(email="updated@example.com")
    print(f"Deleting user: {delete_data}")
    
    deleted_user = delete_user(delete_data)
    print(f"Deleted user: {deleted_user}")