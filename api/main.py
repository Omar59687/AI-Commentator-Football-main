import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from services.pipeline_service import PipelineService

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from uuid import uuid4


load_dotenv()

app = FastAPI(
    title="AI Football Commentator"
)

UPLOAD_DIR = Path("media/input")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


app.mount(
    "/static",
    StaticFiles(directory="frontend/static"),
    name="static"
)

templates = Jinja2Templates(
    directory="frontend/templates"
)



@app.post("/generate")
async def generate_video(
    file: UploadFile = File(...)
):

    video_path = UPLOAD_DIR / f"{uuid4()}.mp4"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    service = PipelineService(

        api_key=os.getenv("GEMINI_API_KEY"),

        model=os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash"
        )

    )

    result = service.run(
        str(video_path),
    )

    return {
        "status": "success",
        "summary": result["summary"],
        "english": result["english"],
        "arabic": result["arabic"],
        "english_video_url": "/output/english",
        "arabic_video_url": "/output/arabic",
    }


@app.get("/output/{language}")
async def get_commentary_video(language: str):

    outputs = {
        "english": Path("media/output/english_commentary.mp4"),
        "arabic": Path("media/output/arabic_commentary.mp4"),
    }
    output = outputs.get(language)

    if output is None or not output.is_file():
        return {"status": "not_found"}

    return FileResponse(
        output,
        media_type="video/mp4",
        filename=f"{language}_commentary.mp4",
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )
