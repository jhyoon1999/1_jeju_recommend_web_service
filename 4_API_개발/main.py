from fastapi import FastAPI
from mangum import Mangum
import uvicorn
from starlette.middleware.cors import CORSMiddleware

from src.domain.recommend import recommend_router

app = FastAPI()
handler = Mangum(app)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def home() :
    return {"message" : "제주도"}

app.include_router(recommend_router.router)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0.', port=9000)