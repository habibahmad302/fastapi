from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from retry import retry
from PIL import Image, ImageEnhance
import os
import uuid
from pathlib import Path
import hashlib
from cachetools import TTLCache
import tempfile
import shopify
import base64
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Dynamic CORS for universal compatibility
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates (for manual testing)
templates = Jinja2Templates(directory="templates")

# Configuration
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BASE_URL = os.getenv("BASE_URL", "https://your-railway-app.up.railway.app")

# Create directories at startup for Railway
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Shopify configuration from environment variables
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET")

# Cache setup (TTL: 1 hour)
cache = TTLCache(maxsize=100, ttl=3600)

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file_path: str) -> bool:
    """Validate if the file exists and is an image."""
    if not os.path.exists(file_path):
        return False
    if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return False
    return True

def get_file_hash(file_content: bytes) -> str:
    """Generate a hash for a file to use as cache key."""
    return hashlib.sha256(file_content).hexdigest()

def compress_image(content: bytes, max_size: int = 1024) -> bytes:
    """Compress image to reduce size while maintaining quality."""
    temp_file = None
    temp_output = None
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file.write(content)
        temp_file.flush()
        img = Image.open(temp_file.name)
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        temp_output = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(temp_output.name, "PNG", optimize=True, quality=85)
        with open(temp_output.name, "rb") as f:
            compressed_content = f.read()
        return compressed_content
    except Exception as e:
        logger.error(f"Compression error: {e}")
        return content
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        if temp_output and os.path.exists(temp_output.name):
            os.unlink(temp_output.name)

def enhance_image(image_path: str) -> None:
    """Enhance image quality using sharpening."""
    try:
        img = Image.open(image_path)
        enhancer = ImageEnhance.Sharpness(img)
        img_enhanced = enhancer.enhance(2.0)  # Increase sharpness
        img_enhanced.save(image_path, "PNG")
    except Exception as e:
        logger.error(f"Enhancement error: {e}")

def save_output_image(result_path: str, output_dir: str, output_name: str) -> str:
    """Save the result image to a specified directory and enhance it."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_name)
        img = Image.open(result_path)
        img = img.convert("RGB")
        img.save(output_path, "PNG")
        enhance_image(output_path)  # Enhance quality
        return output_path
    except Exception as e:
        logger.error(f"Save output error: {e}")
        return ""

@retry(tries=3, delay=2, backoff=2)
async def face_swap(source_image: str, dest_image: str, source_face_idx: int = 1, dest_face_idx: int = 1) -> str:
    """Perform face swap using Gradio Client with retry logic."""
    try:
        if not all([validate_file(source_image), validate_file(dest_image)]):
            return "Invalid input files"

        client = Client("Dentro/face-swap")
        result = client.predict(
            sourceImage=handle_file(source_image),
            sourceFaceIndex=source_face_idx,
            destinationImage=handle_file(dest_image),
            destinationFaceIndex=dest_face_idx,
            api_name="/predict"
        )

        if result and os.path.exists(result):
            unique_filename = f"face_swap_{uuid.uuid4().hex}.png"
            final_path = save_output_image(result, OUTPUT_FOLDER, unique_filename)
            if final_path:
                return final_path
            return "Failed to save output"
        return "Face swap failed"
    except Exception as e:
        logger.error(f"Face swap error: {e}")
        return f"Error: {str(e)}"

@app.get("/")
async def index(request: Request):
    """Render a basic page for manual testing."""
    return templates.TemplateResponse("index.html", {"request": request, "result_image": None})

@app.post("/swap")
async def swap_faces(source_image: UploadFile = File(...), dest_image: UploadFile = File(...)):
    """Perform face swap on uploaded images."""
    try:
        # Validate file uploads
        if not source_image.filename or not dest_image.filename:
            return JSONResponse(status_code=400, content={"error": "No file selected"})

        if not (allowed_file(source_image.filename) and allowed_file(dest_image.filename)):
            return JSONResponse(status_code=400, content={"error": "Invalid file format. Only PNG, JPG, JPEG allowed"})

        # Read and compress file contents
        source_content = await source_image.read()
        dest_content = await dest_image.read()
        source_content = compress_image(source_content)
        dest_content = compress_image(dest_content)

        # Generate cache key
        cache_key = f"{get_file_hash(source_content)}:{get_file_hash(dest_content)}"

        # Check cache
        if cache_key in cache:
            result_url = f"{BASE_URL}/{cache[cache_key]}"
            return {"result_image": result_url}

        # Save files temporarily
        with tempfile.TemporaryDirectory() as temp_dir:
            source_filename = f"source_{uuid.uuid4().hex}.{source_image.filename.rsplit('.', 1)[1]}"
            dest_filename = f"dest_{uuid.uuid4().hex}.{dest_image.filename.rsplit('.', 1)[1]}"
            source_path = os.path.join(temp_dir, source_filename)
            dest_path = os.path.join(temp_dir, dest_filename)

            # Write compressed files
            with open(source_path, "wb") as f:
                f.write(source_content)
            with open(dest_path, "wb") as f:
                f.write(dest_content)

            # Perform face swap
            result = await face_swap(source_path, dest_path)
            if result.startswith("Error") or result == "Invalid input files" or result == "Failed to save output":
                return JSONResponse(status_code=500, content={"error": result})

            # Cache result
            cache[cache_key] = result

            # Return absolute URL
            result_url = f"{BASE_URL}/{result}"
            return {"result_image": result_url}
    except Exception as e:
        logger.error(f"Swap endpoint error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})

@app.post("/create-shopify-product")
async def create_shopify_product(
    shopify_access_token: str = Form(...),
    shop_url: str = Form(...),
    image_url: str = Form(...),
    product_title: str = Form(...),
    price: float = Form(...),
    api_key: Optional[str] = Form(None),
    api_secret: Optional[str] = Form(None)
):
    """Create a Shopify product with the swapped image."""
    try:
        # Use provided API key/secret or fallback to env vars
        api_key = api_key or SHOPIFY_API_KEY
        api_secret = api_secret or SHOPIFY_API_SECRET
        if not all([api_key, api_secret, shopify_access_token, shop_url]):
            return JSONResponse(status_code=400, content={"error": "Missing Shopify credentials"})

        shopify.Session.setup(api_key=api_key, secret=api_secret)
        shop = shopify.ShopifyResource.set_site(f"https://{shop_url}/admin")
        shopify.ShopifyResource.activate_session(shopify.Session(shop_url, "2023-04", shopify_access_token))
        
        # Convert image URL to local path
        local_path = image_url.replace(BASE_URL + '/', '')
        if not os.path.exists(local_path):
            return JSONResponse(status_code=400, content={"error": "Image file not found"})

        # Upload image to Shopify
        with open(local_path, "rb") as f:
            file = shopify.File.create({
                "filename": os.path.basename(image_url),
                "attachment": base64.b64encode(f.read()).decode("utf-8")
            })
        
        # Create product
        product = shopify.Product.create({
            "title": product_title,
            "variants": [{"price": price}],
            "images": [{"src": file.attributes['public_url']}]
        })
        shopify.ShopifyResource.clear_session()
        return {
            "product_id": product.id,
            "product_url": f"https://{shop_url}/products/{product.handle}"
        }
    except Exception as e:
        logger.error(f"Shopify product creation error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Shopify error: {str(e)}"})
