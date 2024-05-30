import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from application.initializer import IncludeAPIRouter


def get_application():
    _app = FastAPI(title="Fast API")
    _app.include_router(IncludeAPIRouter())
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,    
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return _app


app = get_application()
app.mount("/static", StaticFiles(directory="./application/main/static"), name="static")
if __name__ == "__main__":
    uvicorn.run("manage:app", host="0.0.0.0", port=8000, use_colors=True, reload=True)
    