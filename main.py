import os
import asyncio
import time
import logging
from typing import Optional, List, Union, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import replicate
from dotenv import load_dotenv

# Configuración Inicial
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="NanoSeedream Microservice", version="1.1.0")

# Replicate Settings
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_TIMEOUT = int(os.getenv("REPLICATE_TIMEOUT", "300"))
LLM_MODEL = "openai/gpt-4o-mini"
SEEDREAM_MODEL = "bytedance/seedream-5-lite"

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapeo de Aspect Ratio para Seedream
RATIO_MAP = {
    "4:5": "match_input_image",
    "5:4": "match_input_image",
    "1:1": "1:1",
    "16:9": "16:9",
    "4:3": "4:3",
    "3:4": "3:4",
    "9:16": "9:16"
}

class SeedreamRequest(BaseModel):
    image_url: str
    prompt: str
    reference_image_urls: Optional[Union[str, List[str]]] = None
    image_aspect_ratio: Optional[str] = "1:1"

class SeedreamResponse(BaseModel):
    status: str
    model_used: str
    output_url: Optional[str] = None
    translated_prompt: Optional[str] = None
    error: Optional[str] = None
    execution_time: float
    metadata: Optional[Dict[str, Any]] = None

async def agentic_translate(user_prompt: str) -> str:
    """
    Usa Llama-3-8B-Instruct para convertir un prompt descriptivo de Nanobanana 
    en una instrucción narrativa optimizada para SeeDream-5-Lite.
    """
    system_prompt = (
        "You are a prompt translator for SeeDream-5-Lite. Transform the NanoBanana 'prompt' field into narrative instructions for SeeDream.\n\n"
        "CRITICAL - IGNORE ANALYSIS SECTIONS:\n"
        "The input may contain JSON with fields like 'scene_analysis', 'subject_anatomy', 'marks_detected', 'lighting_design'. "
        "These are ANALYSIS for internal use. DO NOT include them in the output.\n"
        "ONLY translate the content from the 'prompt' field.\n\n"
        "CRITICAL - IMAGE INPUT STRUCTURE:\n"
        "- If prompt says 'Image 1 Left is the Target' + 'Reference (Right)':\n"
        "  START WITH: 'Image 1 Left is the target photo to edit. The Right image is the reference for style/elements to apply.'\n"
        "- If prompt says 'Image 1 is the original photo' + 'additional images are face reference crops':\n"
        "  START WITH: 'Image 1 is the target photo to edit. The additional images are face reference crops. Use each face reference to preserve that person's exact facial identity.'\n"
        "- If NO reference images mentioned:\n"
        "  START WITH: 'Keeping the exact facial features, bone structure, pose, body positioning, and framing completely unchanged...'\n\n"
        "CRITICAL - IDENTITY PRESERVATION:\n"
        "When prompt mentions 'Identity Lock', 'Zero head-swaps', 'Zero skeletal deformation':\n"
        "- Include these EXACT phrases in the opening\n"
        "- Describe EACH subject with their specific characteristics as listed in the prompt\n\n"
        "CRITICAL - BANNER AND ANNOTATIONS:\n"
        "When prompt mentions 'bottom-banner text', 'visual annotations', 'arrows':\n"
        "- Include: 'Translate all visual annotations and the bottom-banner text into photorealistic edits'\n"
        "- Include: 'Transfer elements from the Reference to the Target as indicated'\n\n"
        "SECTIONS TO TRANSLATE (in order):\n"
        "1. OPENING (based on image structure)\n"
        "2. IDENTITY LOCK (if present)\n"
        "3. TECHNIQUE - name and description from the 'prompt' field only\n"
        "4. LIGHTING/COMPOSITION - technical specs from 'prompt' field\n"
        "5. CAMERA/AESTHETIC - from 'prompt' field\n"
        "6. CLOSING: 'extremely sharp focus, high-resolution 8k, detailed skin texture, raw photo style, professional color grading.'\n\n"
        "RULE: Only use content from the 'prompt' field. IGNORE scene_analysis, lighting_design, and other analysis sections."
    )
    
    try:
        start_t = time.time()
        # Llamada a Llama-3 para 'agenciar' el prompt
        output = await asyncio.to_thread(
            replicate.run,
            LLM_MODEL,
            input={
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "max_new_tokens": 2000,
                "temperature": 0.3
            }
        )
        # Combinar la salida si es una lista (Replicate suele devolver generadores/listas para LLMs)
        full_translation = "".join(output).strip() if isinstance(output, list) else str(output).strip()
        
        # Limpiar prefijos no deseados del LLM
        unwanted_prefixes = [
            "here is the translated prompt:",
            "here's the translated prompt:",
            "the translated prompt is:",
            "translated prompt:",
            "here is the translation:",
            "here's the translation:",
            "translation:",
        ]
        lower_translation = full_translation.lower()
        for prefix in unwanted_prefixes:
            if lower_translation.startswith(prefix):
                full_translation = full_translation[len(prefix):].strip()
                lower_translation = full_translation.lower()
        
        # Asegurar que siempre sea string
        if not full_translation:
            full_translation = f"Keeping the subject intact, modify: {user_prompt}. Detailed 8k raw photo quality."
        
        logger.info(f"Traducción Agéntica (LLM) completada en {time.time() - start_t:.2f}s -> {full_translation[:50]}...")
        return full_translation
    except Exception as e:
        logger.error(f"Error en traducción LLM: {str(e)}")
        fallback = f"Keeping the subject intact, modify the image to: {user_prompt}. Detailed 8k raw photo quality."
        logger.info(f"USING FALLBACK: {fallback[:50]}...")
        return fallback

@app.post("/v1/edit", response_model=SeedreamResponse)
async def edit_image(request: SeedreamRequest):
    start_time = time.time()
    
    logger.info(f"=== ENDPOINT CALLED ===")
    logger.info(f"Request prompt: {request.prompt}")
    logger.info(f"Request image_url: {request.image_url[:50] if request.image_url else 'None'}...")
    
    if not REPLICATE_API_TOKEN:
        logger.error("REPLICATE_API_TOKEN is missing!")
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN is missing")

    try:
        # 1. Traducción Agéntica del Prompt
        logger.info("Starting prompt translation...")
        translated_prompt = await agentic_translate(request.prompt)
        logger.info(f"Translation result: {translated_prompt[:100] if translated_prompt else 'NONE'}")

        # 2. Preparación de Referencias
        image_input = [request.image_url]
        if request.reference_image_urls:
            refs = request.reference_image_urls if isinstance(request.reference_image_urls, list) else [request.reference_image_urls]
            valid_refs = [r for r in refs if r and r.strip()]
            image_input.extend(valid_refs)
        
        # 3. Mapeo de Ratio
        final_ratio = RATIO_MAP.get(request.image_aspect_ratio, request.image_aspect_ratio)
        
        logger.info(f"SEND TO SEEDREAM -> prompt: {translated_prompt[:80] if translated_prompt else 'NULL'}...")
        
        # 4. Ejecución de SeeDream-5-Lite
        output = await asyncio.wait_for(
            asyncio.to_thread(
                replicate.run,
                SEEDREAM_MODEL,
                input={
                    "prompt": translated_prompt,
                    "image_input": image_input,
                    "size": "3K",
                    "aspect_ratio": final_ratio,
                    "output_format": "jpeg"
                }
            ),
            timeout=REPLICATE_TIMEOUT
        )

        output_url = output[0] if isinstance(output, list) and len(output) > 0 else str(output)

        return SeedreamResponse(
            status="success",
            model_used="seedream-5-lite",
            output_url=output_url,
            translated_prompt=translated_prompt,
            execution_time=time.time() - start_time,
            metadata={
                "aspect_ratio": final_ratio,
                "resolution": "3K"
            }
        )

    except Exception as e:
        logger.error(f"Error procesando microservicio: {str(e)}")
        return SeedreamResponse(
            status="error",
            model_used="seedream-5-lite",
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.get("/health")
def health():
    return {"status": "ok", "service": "nanoseedream", "llm": LLM_MODEL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
