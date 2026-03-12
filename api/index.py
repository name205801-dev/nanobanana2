import base64
import io
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "AIzaSyAs7wdnho1KsufQlUmfysbKi-NbaB9Mfco"
MODEL = "gemini-3.1-flash-image-preview"


class GenerateRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "3:4"
    image_size: str = "1K"
    use_grounding: bool = False
    reference_images: Optional[List[Dict[str, str]]] = None  # list of {data: base64str, mime_type: str}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    try:
        client = genai.Client(api_key=API_KEY)

        # Build contents list
        contents = [req.prompt]

        if req.reference_images:
            for img_info in req.reference_images[:14]:
                # Decode base64 image and open as PIL Image
                from PIL import Image as PILImage
                img_data = base64.b64decode(img_info["data"])
                pil_img = PILImage.open(io.BytesIO(img_data))
                contents.append(pil_img)

        # Build config
        tools = []
        if req.use_grounding:
            tools.append({"google_search": {}})

        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=req.aspect_ratio,
                image_size=req.image_size,
            ),
            thinking_config=types.ThinkingConfig(
                thinking_level="High",
                include_thoughts=True,
            ),
        )
        if tools:
            config.tools = tools

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )

        result_image_b64 = None
        thought_text = ""
        response_text = ""

        for part in response.parts:
            if part.thought:
                if part.text:
                    thought_text += part.text
            elif part.text is not None:
                response_text += part.text
            elif part.inline_data is not None:
                img_bytes = part.inline_data.data
                result_image_b64 = base64.b64encode(img_bytes).decode("utf-8")
                mime = part.inline_data.mime_type or "image/png"

        if result_image_b64 is None:
            raise HTTPException(status_code=500, detail="이미지 생성에 실패했습니다. 프롬프트를 수정하거나 다시 시도하세요.")

        return JSONResponse({
            "image": result_image_b64,
            "mime_type": mime,
            "thought": thought_text,
            "text": response_text,
        })

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "Quota exceeded" in error_msg:
            detail = "무료 API 일일 사용량을 초과했습니다. 잠시 후 안내된 시간(예: 47초) 뒤에 다시 시도하시거나 내일 다시 이용해주세요."
            raise HTTPException(status_code=429, detail=detail)
        raise HTTPException(status_code=500, detail=error_msg)


# Mount static files (for local dev only; Vercel handles this differently)
try:
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public")
    if os.path.exists(static_dir):
        app.mount("/public", StaticFiles(directory=static_dir), name="public")
except Exception:
    pass
