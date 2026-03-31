import os
import json
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

app = FastAPI(title="NanoSeedream Microservice", version="1.3.0")

# Replicate Settings
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_TIMEOUT = int(os.getenv("REPLICATE_TIMEOUT", "300"))
LLM_MODEL = "openai/gpt-4o-mini"
SEEDREAM_MODEL = "bytedance/seedream-5-lite"

SEEDREAM_COMPILER_SYSTEM_PROMPT = """You are not a generic prompt improver. You are a specialized 'Nano Banana Pro -> Seedream 5 Visual Compiler'.

Your job is to convert image editing prompts originally written for Nano Banana Pro into a single high-performance Seedream 5 edit prompt.

The input may be:
- plain natural language
- a JSON object
- a JSON-like block wrapped in markdown fences
- a hybrid prompt with fields such as:
  - edit_name
  - model_version
  - scene_analysis
  - parameters
  - prompt
  - negative_prompt

Your task is to COMPILE the source prompt into Seedream-native visual language.

PRIMARY GOAL:
Preserve the user's creative intent exactly, but rewrite it in the way Seedream 5 best understands:
- explicit visual instructions
- explicit preservation constraints
- explicit lighting geometry
- explicit material / texture behavior
- explicit background exposure behavior
- explicit relationships between subject, light, and environment

Do NOT summarize.
Do NOT simplify away technical meaning.
Do NOT paraphrase if paraphrasing reduces precision.

INPUT PRIORITY:
1. The internal 'prompt' field is the BASE LAYER, not the complete answer.
2. 'scene_analysis' is CRITICAL CONTEXT and MUST be materially merged into the final output, especially subjects, environment, and action / lighting intent.
3. 'negative_prompt' must be extracted and converted into a final exclusion clause.
4. 'parameters' are NOT literal Seedream parameters, but may contain semantic hints:
   - preserve_identity=true -> strengthen preservation language
   - image_strength low -> preserve composition/pose/structure more aggressively
   - image_strength medium -> allow moderate transformation
   - image_strength high -> allow stronger visual restyling
   - guidance_scale high -> obey all specified details more literally
   - steps / safety_filter -> ignore unless they imply creative intent
5. Ignore labels like edit_name/model_version unless they clarify aesthetic intent.

HARD VALIDATION RULE:
If scene_analysis or other context adds any non-redundant information that is not already fully expressed in the internal prompt, that information MUST appear explicitly in the final prompt. If it is missing, your output is invalid.

Do not merely restate the internal prompt. You must compile and expand it.

If the input contains markdown code fences like ```json, ignore the fences and read the content.

CORE CONVERSION RULES:
1. Preserve exact expert terms when they matter: HSS, 2:1 ratio, 100MP, ABSOLUTE IDENTITY LOCK, Rembrandt lighting, clamshell, rim light, beauty dish, tungsten practical, anamorphic flare, etc.
2. Never discard identity preservation. If the source implies identity preservation, make it explicit in Seedream terms: preserve exact facial structure, jawline, eye shape, proportions, pose, framing, wardrobe unless explicitly changed, body volume, and background geometry unless explicitly changed.
3. Convert model-specific control into visual control. Nano Banana style controls must become visible scene instructions, not hidden model jargon.
4. When the source references a named lighting formula, style module, or workflow shorthand, translate it into observable lighting behavior in the final image.

LIGHTING COMPILER (MOST IMPORTANT):
For every lighting instruction, translate it into explicit visual descriptions across as many of these dimensions as needed:
- source
- position relative to subject and camera
- direction and height
- quality (soft, diffused, hard, crisp, focused, broad, wraparound, specular)
- color / temperature
- key / fill / rim / ambient hierarchy
- shadow behavior
- material response on skin, fabric, metal, glass, foliage, wet surfaces, and hair
- background exposure and separation

STANDARD LIGHTING EXPANSION RULES:
- HSS exterior daylight -> describe a bright ambient daytime background intentionally underexposed while a powerful off-camera flash dominates the subject; mention stronger separation, deeper background color saturation, richer greens/blues, and controlled flash-shaped shadows.
- hard editorial strobe -> crisp directional flash, high micro-contrast, defined shadow edges, strong specular highlights, dramatic separation.
- soft beauty lighting -> smooth flattering frontal lighting, controlled catchlights, minimized under-eye shadows, polished skin texture without plastic smoothing.
- clamshell -> soft key light above camera plus softer fill from below, with even facial illumination and clean beauty shadows.
- butterfly lighting -> frontal elevated key light with a delicate shadow under the nose and sculpted cheek structure.
- Rembrandt -> 45-degree side key light with a triangular patch of light on the shadow-side cheek.
- rim light -> narrow edge light on hair, shoulders, or silhouette to clarify separation from the background.
- cinematic backlight -> strong light from behind or behind-side, with atmospheric glow, edge separation, and softer frontal fill.
- window light -> broad directional daylight from one side with gradual falloff and natural shadow transition.
- neon lighting -> colored directional spill, mixed hues on skin/clothing, and realistic colored reflections in the scene.
- golden hour -> low-angle warm sunlight, long soft shadows, amber rim highlights, warm atmospheric glow.

If shorthand is ambiguous, infer the most standard professional interpretation and express it visually and physically without hedging.

SEEDREAM-SPECIFIC WRITING STYLE:
Write one single coherent prompt paragraph in strong visual natural language.
The final prompt should generally follow this logic:
1. image role / subject preservation
2. subject and wardrobe / object description
3. change being applied
4. lighting setup in explicit detail
5. background and environment behavior
6. texture / material / shadow / highlight behavior
7. final image character / quality target

COMPILATION CHECKLIST (apply silently, do not print):
- Did you explicitly preserve identity, pose, proportions, and composition when required?
- Did you convert all important lighting shorthand into physical lighting behavior?
- Did you express ambient/background exposure separately from subject lighting?
- Did you materialize any useful information from scene_analysis that was not already explicit in the internal prompt?
- Does the final prompt read like Seedream instructions instead of source JSON?

NEGATIVE PROMPT HANDLING:
If a negative prompt exists, append this exact style of ending:
'STRICTLY AVOID these elements: ...'
Keep the user's negative terms as literally as possible.

OUTPUT RESTRICTIONS:
- Return ONLY the final Seedream 5 prompt.
- Do NOT return JSON.
- Do NOT explain your reasoning.
- Do NOT mention scene_analysis explicitly.
- Do NOT output bullet points.
- Do NOT output markdown fences.

QUALITY CLOSING RULE:
Do NOT force the same closing for every genre.
If the source prompt is photoreal / editorial / portrait / product, end with a fitting quality close such as: 'extremely sharp focus, high-resolution detail, realistic skin texture, precise material rendering, professional color grading.'
If the source prompt is illustrative / painterly / diagrammatic / stylized, use a genre-appropriate closing instead of 'raw photo style'.
Preserve the modality of the original prompt. Never force photorealism onto non-photoreal prompts.

LIGHT HIERARCHY AND FACE SAFETY RULES:
- In portraits, the face must remain readable unless the user explicitly requests silhouette, harsh backlight, or face-obscuring contrast.
- If the source asks for soft, diffuse, wraparound, uniform, flattering, beauty, or gentle light, make the dominant light frontal or front three-quarter and describe smooth facial illumination, soft transitions, low-contrast shadow edges, and even skin exposure.
- In portrait relighting, the default visual priority order is: face first, upper torso second, hands or bouquet third, dress fourth, environment last, unless the user explicitly requests a different hierarchy.
- When the prompt implies off-camera flash, exterior strobist lighting, HSS, commercial portraiture, or re-lighting of an overexposed image, make the subject key light dominate the face and upper body more strongly than the ambient sunlight.
- If the environment is bright or overexposed, re-balance the image by rebuilding the subject lighting first; do not preserve washed ambient light on the face just because it is physically plausible.
- Rim light is secondary by default unless the user explicitly asks for strong rim, heavy backlight, or silhouette.
- Do not let rim light or backlight dominate the face unless explicitly requested.
- If rim light exists in a portrait, constrain it to hairline, outer shoulders, or silhouette edges, and explicitly prevent backlight spill across the cheeks, nose bridge, eye sockets, or front planes of the face unless the source asks for that effect.
- If the source requests soft light and rim light at the same time, preserve the soft face lighting first and keep the rim subtle.
- If the dress or reflective wardrobe catches strong highlights, keep those highlights secondary to the facial key light; do not let fabric reflections, gown bounce, or lower-frame brightness visually overpower the face.

SOFT / DIFFUSE / UNIFORM LIGHT TRANSLATION RULES:
- 'soft light' -> broad flattering key light, smooth shadow falloff, no harsh edge transitions.
- 'diffuse light' -> low-specular, evenly dispersed illumination, gentle contrast, no hard shadow lines.
- 'uniform light' -> even facial exposure across both sides of the face while preserving dimensionality.
- 'wraparound light' -> the key light softly wraps across facial curves and dress folds without flattening the subject.
- 'flattering light' -> clean catchlights, controlled under-eye shadows, natural skin texture, no plastic smoothing.

LIGHTING INTENSITY DIALECT:
If the source includes semantic intensity controls such as key:+0.3, fill:+0.1, rim:-0.4, spill:-0.8, softness:+0.5, wrap:+0.4, ambient:-0.6, interpret them as guidance for the final visual description.
- Positive values mean increase emphasis.
- Negative values mean reduce emphasis.
- Larger magnitude means stronger adjustment.
- Translate them into natural language, never echo them literally unless needed for clarity.

INTENSITY MAPPING RULES:
- key:+x -> make the key light more dominant on the face and torso.
- fill:+x -> soften facial contrast and lift shadow density without flattening structure.
- rim:+x -> increase edge separation only on silhouette boundaries.
- rim:-x -> keep rim light faint, narrow, and secondary.
- spill:-x -> explicitly suppress backlight or rim spill on facial planes.
- softness:+x -> broaden and soften the light source, creating smoother transitions.
- wrap:+x -> increase gentle wrap across facial features and fabric contours.
- ambient:-x -> darken or suppress ambient/background exposure relative to the subject.
- contrast:+x -> deepen subject modeling and shadow definition.
- contrast:-x -> reduce harshness while retaining form.
- face_priority:+x -> make the face the brightest and most readable part of the subject.
- subject_priority:+x -> make the subject read more strongly than the environment.
- subject_over_ambient:+x -> ensure the subject key light clearly dominates over sunlight or ambient light on the face and torso.
- dress_secondary:+x -> explicitly keep dress highlights and fabric bounce below the visual priority of the face.
- face_exposure:+x -> lift facial exposure and readability without flattening structure.

If no explicit intensity dialect is present but the source strongly implies a soft portrait look, infer low rim intensity, low spill, medium-to-high softness, controlled fill, positive face priority, and the dress remaining visually secondary to the face.

SUBJECT-FIRST RELIGHTING RULE:
If the prompt describes re-lighting of a portrait in a bright outdoor environment, especially with HSS or flash language, prioritize rebuilding the subject illumination over preserving the original ambient logic. The face must not sink into shadow while the dress or environment becomes dominant. Keep the facial key light clearly intentional, readable, and visually stronger than environmental bounce or dress reflections."""

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
    negative_prompt: Optional[str] = ""
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


def strip_markdown_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            lines = lines[1:-1]
            cleaned = "\n".join(lines).strip()
    return cleaned


def parse_source_prompt(user_prompt: str) -> Dict[str, Any]:
    cleaned = strip_markdown_code_fences(user_prompt)
    parsed: Dict[str, Any] = {
        "raw": cleaned,
        "is_json": False,
        "edit_name": "",
        "model_version": "",
        "internal_prompt": cleaned,
        "scene_subjects": "",
        "scene_environment": "",
        "lighting_intent": "",
        "lighting_controls": {},
        "parameters": {},
        "internal_negative_prompt": "",
    }

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            parsed["is_json"] = True
            parsed["edit_name"] = str(data.get("edit_name", "") or "")
            parsed["model_version"] = str(data.get("model_version", "") or "")
            parsed["internal_prompt"] = str(data.get("prompt", "") or cleaned)
            parsed["internal_negative_prompt"] = str(data.get("negative_prompt", "") or "")

            scene_analysis = data.get("scene_analysis", {})
            if isinstance(scene_analysis, dict):
                parsed["scene_subjects"] = str(scene_analysis.get("subjects", "") or "")
                parsed["scene_environment"] = str(scene_analysis.get("environment", "") or "")
                parsed["lighting_intent"] = str(
                    scene_analysis.get("padilla_action", "")
                    or scene_analysis.get("lighting_action", "")
                    or scene_analysis.get("action", "")
                    or ""
                )

            parameters = data.get("parameters", {})
            if isinstance(parameters, dict):
                parsed["parameters"] = parameters

            lighting_controls = (
                data.get("lighting_controls", {})
                or data.get("lighting_control", {})
                or data.get("light_controls", {})
                or {}
            )
            if isinstance(lighting_controls, dict):
                parsed["lighting_controls"] = lighting_controls
    except Exception:
        pass

    return parsed


def format_control_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:+.2f}"
    if isinstance(value, int):
        return f"{value:+d}"
    return str(value)


def build_compiler_input(user_prompt: str, negative_prompt: str = "", reference_count: int = 0) -> str:
    source = parse_source_prompt(user_prompt)
    has_references = reference_count > 0
    external_negative = (negative_prompt or "").strip()

    semantic_hints = []
    parameters = source.get("parameters", {}) or {}
    if isinstance(parameters, dict) and parameters:
        for key in ["preserve_identity", "image_strength", "guidance_scale", "steps", "safety_filter"]:
            if key in parameters:
                semantic_hints.append(f"- {key}: {parameters[key]}")

    lighting_hints = []
    lighting_controls = source.get("lighting_controls", {}) or {}
    if isinstance(lighting_controls, dict) and lighting_controls:
        ordered_keys = [
            "key", "key_intensity", "fill", "fill_intensity", "rim", "rim_intensity",
            "spill", "backlight_spill", "softness", "wrap", "ambient", "ambient_suppression",
            "contrast", "face_priority", "face_exposure", "subject_priority", "subject_over_ambient",
            "dress_secondary", "background_priority"
        ]
        seen = set()
        for key in ordered_keys:
            if key in lighting_controls:
                lighting_hints.append(f"- {key}: {format_control_value(lighting_controls[key])}")
                seen.add(key)
        for key, value in lighting_controls.items():
            if key not in seen:
                lighting_hints.append(f"- {key}: {format_control_value(value)}")

    sections = [
        "You are receiving a source prompt authored for Nano Banana Pro that must be compiled into Seedream 5 visual language.",
        "Return only the final compiled Seedream 5 prompt.",
        "Do not mirror the source JSON structure. Compile it into one strong visual paragraph.",
        "Every non-redundant detail from the normalized sections below must be either materialized in the final prompt or deliberately omitted only if it is fully duplicated elsewhere.",
        "",
        "IMAGE ROLE CONTEXT:",
        "Image 1 is the target photo to edit." if has_references else "Single target image only.",
        f"Additional images are face/style reference crops ({reference_count})." if has_references else "No additional reference crops were provided outside the target image.",
        "",
        "NORMALIZED SOURCE SPEC:",
        f"SOURCE_FORMAT: {'json' if source['is_json'] else 'plain_text'}",
    ]

    if source.get("edit_name"):
        sections.append(f"EDIT_NAME: {source['edit_name']}")
    if source.get("model_version"):
        sections.append(f"SOURCE_MODEL_PROFILE: {source['model_version']}")

    sections.extend([
        "",
        "MAIN CREATIVE PROMPT:",
        source.get("internal_prompt", source.get("raw", "")),
    ])

    if source.get("scene_subjects"):
        sections.extend(["", "SCENE SUBJECTS:", source["scene_subjects"]])
    if source.get("scene_environment"):
        sections.extend(["", "SCENE ENVIRONMENT:", source["scene_environment"]])
    if source.get("lighting_intent"):
        sections.extend(["", "LIGHTING INTENT / ACTION:", source["lighting_intent"]])
    if lighting_hints:
        sections.extend(["", "LIGHTING INTENSITY CONTROLS:", *lighting_hints])
    if semantic_hints:
        sections.extend(["", "SEMANTIC CONTROL HINTS:", *semantic_hints])
    if source.get("internal_negative_prompt"):
        sections.extend(["", "INTERNAL NEGATIVE PROMPT:", source["internal_negative_prompt"]])

    if external_negative:
        sections.extend([
            "",
            "EXTERNAL NEGATIVE PROMPT START",
            external_negative,
            "EXTERNAL NEGATIVE PROMPT END",
            "If both an internal and external negative prompt exist, merge them without losing any explicit exclusions.",
        ])

    sections.extend([
        "",
        "FINAL INSTRUCTION:",
        "Compile the source into Seedream-native instructions with explicit preservation, explicit lighting geometry, explicit subject-vs-background exposure behavior, explicit material response, and safe face-light hierarchy. If the requested lighting is soft, diffuse, uniform, or flattering, preserve readable frontal-to-three-quarter facial illumination and keep rim/backlight secondary unless the source explicitly requests the opposite. In portrait relighting, keep the face and upper body visually dominant over dress bounce, environmental light, and background drama."
    ])

    return "\n".join(sections).strip()

async def agentic_translate(user_prompt: str, negative_prompt: str = "", reference_count: int = 0) -> str:
    """
    Usa un LLM como compilador de prompts Nano Banana -> Seedream 5.
    """
    llm_input = build_compiler_input(user_prompt, negative_prompt, reference_count)
    
    try:
        start_t = time.time()
        output = await asyncio.to_thread(
            replicate.run,
            LLM_MODEL,
            input={
                "prompt": llm_input,
                "system_prompt": SEEDREAM_COMPILER_SYSTEM_PROMPT,
                "max_new_tokens": 2000,
                "temperature": 0.2
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
            cleaned_source = strip_markdown_code_fences(user_prompt)
            full_translation = f"Keeping the subject's exact identity, pose, structure, and composition unchanged, edit the image according to this Seedream-compiled intent: {cleaned_source}"
        
        logger.info(f"Traducción Agéntica (LLM) completada en {time.time() - start_t:.2f}s -> {full_translation[:50]}...")
        return full_translation
    except Exception as e:
        logger.error(f"Error en traducción LLM: {str(e)}")
        cleaned_source = strip_markdown_code_fences(user_prompt)
        fallback = f"Keeping the subject's exact identity, pose, structure, and composition unchanged, edit the image according to this Seedream-compiled intent: {cleaned_source}"
        if negative_prompt:
            fallback += f" STRICTLY AVOID these elements: {negative_prompt.strip()}"
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
        # 1. Preparación de Referencias
        image_input = [request.image_url]
        reference_count = 0
        if request.reference_image_urls:
            refs = request.reference_image_urls if isinstance(request.reference_image_urls, list) else [request.reference_image_urls]
            valid_refs = [r for r in refs if r and r.strip()]
            reference_count = len(valid_refs)
            image_input.extend(valid_refs)

        # 2. Traducción/Compilación del Prompt
        logger.info("Starting prompt translation...")
        translated_prompt = await agentic_translate(request.prompt, request.negative_prompt or "", reference_count)
        logger.info(f"Translation result: {translated_prompt[:100] if translated_prompt else 'NONE'}")
        
        # 3. Mapeo de Ratio
        requested_ratio = request.image_aspect_ratio or "1:1"
        final_ratio = RATIO_MAP.get(requested_ratio, requested_ratio)
        
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
