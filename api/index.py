import base64
import io
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.1-flash-image-preview"


class GenerateRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "1:1"
    image_size: str = "512"
    use_grounding: bool = False
    use_no_person: bool = False
    reference_images: Optional[List[Dict[str, str]]] = None  # list of {data: base64str, mime_type: str}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    try:
        if not API_KEY:
            raise HTTPException(status_code=500, detail="서버에 API 키가 설정되지 않았습니다.")
        
        client = genai.Client(api_key=API_KEY)

        # Build contents list
        # 1. 사용자의 원래 프롬프트를 바탕으로 부가적인 지시사항을 텍스트로 합쳐서 전달합니다.
        final_prompt = req.prompt
        if req.use_no_person:
            # 모델이 사람을 생성하지 않도록 매우 강력한 네거티브 지시를 프롬프트의 시작과 끝에 모두 배치합니다.
            strong_negative = "CRITICAL INSTRUCTION: You MUST remove all humans, people, characters, and human-like figures from the reference images. DO NOT generate any people. If there are people in the reference image, completely erase them and reconstruct the background or surrounding objects instead. The final result MUST be 100% free of any human presence."
            final_prompt = f"{strong_negative}\n\nUser Prompt: {final_prompt}\n\n(Reminder: Erase ALL people from the reference. No humans allowed in the final image.)"
        
        # 2. 사고 과정을 한국어로 출력하도록 시스템 지시사항을 슬쩍 덧붙입니다.
        #    (Gemini 3.1 flash image 모델의 thought 언어 제어를 위해 프롬프트에 명시)
        final_prompt += "\n\n(Important rule for your thinking process: Output your thought process entirely in Korean.)"

        contents = [final_prompt]

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

        # 사고 과정(thought)은 part.text에 누적되며, part.thought가 True인 파트로 들어옵니다.
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
