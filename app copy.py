from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks
from fastapi.templating import Jinja2Templates
import uvicorn
import shutil
import os
import uuid
import json
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")
templates.env.autoescape = False

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
if __name__ == "__main__":
    uvicorn.run(app, debug=True)