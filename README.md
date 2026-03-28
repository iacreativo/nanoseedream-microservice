# NanoSeedream Microservice

Microservicio de edición de imágenes basado en **ByteDance Seedream-5-Lite** (vía Replicate).

## Características

- Edición de imágenes con prompts textuales
- Soporte para múltiples imágenes de referencia (hasta 14)
- Mapeo automático de aspect ratios no soportados por el modelo
- Resolución fija a 3K (máximo del modelo)
- Timeout configurable
- Docker-ready para Coolify

## Endpoints

### `POST /v1/edit`

Edita una imagen usando Seedream-5-Lite.

**Request Body:**
```json
{
  "image_url": "https://example.com/image.jpg",
  "prompt": "Enhance colors and adjust lighting",
  "reference_image_urls": ["https://example.com/ref1.jpg"],
  "image_aspect_ratio": "4:5"
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `image_url` | string | URL de la imagen principal |
| `prompt` | string | Instrucción de edición |
| `reference_image_urls` | string \| array | Imágenes de referencia opcionales |
| `image_aspect_ratio` | string | Ratio: `1:1`, `16:9`, `3:4`, `4:3`, `4:5`, `5:4` |

**Response:**
```json
{
  "status": "success",
  "model_used": "seedream-5-lite",
  "output_url": "https://output.image.url/result.jpg",
  "execution_time": 12.45
}
```

### `GET /health`

Health check del servicio.

```json
{"status": "ok", "service": "nanoseedream"}
```

## Mapeo de Aspect Ratios

Seedream-5-Lite no soporta todos los ratios. El microservicio mapea automáticamente:

| Input | Aplicado al modelo |
|-------|-------------------|
| `4:5` | `3:4` |
| `5:4` | `4:3` |
| `1:1`, `16:9`, `3:4`, `4:3` | Igual |

## Configuración

1. Copiar `.env.example` a `.env`
2. Agregar tu `REPLICATE_API_TOKEN`
3. Opcional: ajustar `REPLICATE_TIMEOUT` (default: 300s)

```bash
cp .env.example .env
```

## Desarrollo Local

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Docker

```bash
# Build
docker build -t nanoseedream .

# Run
docker run -p 8000:8000 --env-file .env nanoseedream
```

## Despliegue en Coolify

1. Crear nueva aplicación Docker
2. GitHub repo: `https://github.com/iacreativo/nanoseedream-microservice`
3. Branch: `main`
4. Dockerfile: detectado automáticamente
5. Puerto: `8000`
6. Health check: `/health`
7. Variables de entorno: configurar `REPLICATE_API_TOKEN`

## Costos

- Seedream-5-Lite: **2 créditos** por ejecución

## Stack

- Python 3.11
- FastAPI
- Replicate SDK
- Uvicorn
