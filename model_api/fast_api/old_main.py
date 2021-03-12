from enum import Enum
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


# Order matters matches the first path seen
# if this didn't come before the below one, it would always match the below
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}


# typing input
@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}


# How you can create specific values that must be sent
@app.get("/model/{model_name}")
async def get_model(model_name: ModelName):
    if model_name == ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

# tells the API this is a path
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}

# defaults set here, are the defaults when just hitting this path
# words used here are to be used in the path after ?
@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]

# how to set optional params, doesn't have to be set
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}

# Create a model in the form of the body that will be posted to the endpoint
# you can add path and query paramaters to this endpoint, and FastAPI will figure it out.
# Basically the variable that is a subclass of BaseModel will be assumed as the 
# object in the request body.
# 
@app.post("/items/")
async def create_item(item: Item):
    return item

# Basically a lot you can do to manage the query paramater:
# https://fastapi.tiangolo.com/tutorial/query-params-str-validations/

# Functionality also available to validate path params:
# https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/

# You can describe descriptions of fields inside models via the Field object from pydantic

# You can do some pretty cool nesting and data validation along the way
# automatically the correct error will surface to the sender of the request

# You basically do the same thing for response outputs
# response_model=
# response_model_exclude_unset -> useful param to not return something if it doesn't exist

# response_model can be union of objects, response_model=Union[PlaneItem, CarItem]
# meaning either or

# you can set the status_codes of response
#status_code=status.HTTP_201_CREATED

# Nice Methods to handle FileUploads
# might need this for sending model files back and forth

# how to split up code between mulitple files
# APIRouter -> mini applications that allow you to bundle together
# functions from same 

# really easy to write tests
# https://fastapi.tiangolo.com/tutorial/testing/









