import os
import asyncio
import time
import logging
from typing import Optional, List, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import replicate
from dotenv import load_dotenv

# Timeout for replicate calls (seconds)
REPLICATE_TIMEOUT = int(os.getenv("REPLICATE_TIMEOUT", "300"))

# Configuración Inicial
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="NanoSeedream Microservice", version="1.0.0")

# CORS para comunicación externa
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constantes de Compatibilidad
RATIO_MAP = {
    "4:5": "3:4",
    "5:4": "4:3"
}

# Modelos de Datos (Compatibles con el ecosistema existente)
class SeedreamRequest(BaseModel):
    image_url: str
    prompt: str
    reference_image_urls: Optional[Union[str, List[str]]] = None
    image_aspect_ratio: Optional[str] = "1:1"
    resolution: Optional[str] = "3K"

class SeedreamResponse(BaseModel):
    status: str
    model_used: str
    output_url: Optional[str] = None
    error: Optional[str] = None
    execution_time: float

@app.post("/v1/edit", response_model=SeedreamResponse)
async def edit_image(request: SeedreamRequest):
    start_time = time.time()
    
    # 1. Validación de Token
    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN is missing")

    try:
        # 2. Preparación de Referencias
        image_input = [request.image_url]
        if request.reference_image_urls:
            # Filtrar strings vacíos
            refs = request.reference_image_urls if isinstance(request.reference_image_urls, list) else [request.reference_image_urls]
            valid_refs = [r for r in refs if r and r.strip()]
            if valid_refs:
                image_input.extend(valid_refs)
        
        # 3. Lógica de Aspect Ratio (Mapeo Silencioso)
        final_ratio = RATIO_MAP.get(request.image_aspect_ratio, request.image_aspect_ratio)
        
        logger.info(f"Procesando Seedream 5 Lite - Prompt: {request.prompt[:50]}...")
        logger.info(f"Map AR: {request.image_aspect_ratio} -> {final_ratio}")

        # 4. Ejecución en Replicate (Seedream 5 Lite) - Async con timeout
        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(
                    replicate.run,
                    "bytedance/seedream-5-lite",
                    input={
                        "prompt": request.prompt,
                        "image_input": image_input,
                        "size": "3K",  # Resolución fija a 3K (máximo del modelo)
                        "aspect_ratio": final_ratio,
                        "output_format": "jpeg",
                        "max_images": 5
                    }
                ),
                timeout=REPLICATE_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Seedream timeout after {REPLICATE_TIMEOUT}s")
            return SeedreamResponse(
                status="error",
                model_used="seedream-5-lite",
                error=f"Timeout after {REPLICATE_TIMEOUT} seconds",
                execution_time=time.time() - start_time
            )

        # 5. Parsea la salida (Replicate devuelve una lista o un solo URL dependiendo del modelo)
        output_url = output[0] if isinstance(output, list) and len(output) > 0 else output

        return SeedreamResponse(
            status="success",
            model_used="seedream-5-lite",
            output_url=output_url,
            execution_time=time.time() - start_time
        )

    except Exception as e:
        logger.error(f"Error en el proceso SeeDream: {str(e)}")
        return SeedreamResponse(
            status="error",
            model_used="seedream-5-lite",
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.get("/health")
def health():
    return {"status": "ok", "service": "nanoseedream"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
