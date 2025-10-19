import uvicorn
from fastapi import FastAPI
from collections import OrderedDict
from typing import Any, Dict
import asyncio
from time import monotonic
from pydantic import BaseModel
from app.core import NatixClassifier  # Assuming this is the classifier implementation
from PIL import Image
import base64
import io
import numpy as np
import cv2

# -----------------------------
# Request model
# -----------------------------
class ImageRequest(BaseModel):
    image: str  # base64 string or unique image identifier

# -----------------------------
# LRU cache with TTL
# -----------------------------
class LRUCache:
    """Simple LRU cache with TTL for storing results."""
    def __init__(self, maxsize: int = 1024, ttl: float = 300.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str):
        async with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            ts, value = item
            if monotonic() - ts > self.ttl:
                # expired
                del self._data[key]
                return None
            # move to end (most recently used)
            self._data.move_to_end(key)
            return value

    async def set(self, key: str, value: Any):
        async with self._lock:
            if key in self._data:
                del self._data[key]
            self._data[key] = (monotonic(), value)
            # evict oldest if over capacity
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    async def clear(self):
        async with self._lock:
            self._data.clear()

# -----------------------------
# API
# -----------------------------
class NatixClassifierAPI:
    def __init__(self, lru_size: int = 1024, lru_ttl: float = 300.0):
        self.app = FastAPI()
        self.cache = LRUCache(maxsize=lru_size, ttl=lru_ttl)
        # Tracks images currently being processed
        self._in_progress: Dict[str, asyncio.Event] = {}
        start_time = monotonic()
        self.classifier = NatixClassifier(backbone_name="swin_large_patch4_window7_224", device="cpu")
        end_time = monotonic()
        print(f"Classifier initialized in {end_time - start_time:.2f} seconds.")
        self.request_count = 0
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/classify")
        async def classify_image(request: ImageRequest):
            image_data = request.image
            print(f"Received image data of length: {len(image_data)}")
            result = None

            # Check cache first
            cached_result = await self.cache.get(image_data)
            if cached_result is not None:
                result = {"result": cached_result, "cached": True}
                print(f"result used previous: {result}")
                return result

            # Check if another request is already processing this image
            if image_data in self._in_progress:
                event = self._in_progress[image_data]
                await event.wait()  # wait for first request to finish
                # Return cached result
                cached_result = await self.cache.get(image_data)
                result = {"result": cached_result, "cached": True}
                print(f"result used simultaneous: {result}")
                return result

            # Mark as in-progress
            event = asyncio.Event()
            self._in_progress[image_data] = event

            try:
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
                cv2.imwrite(f"images/received_image_{self.request_count}.png", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
                self.request_count += 1
                print(f"Processing request #{self.request_count}")
                image_np = np.array(image)
                roadwork_score = self.classifier.predict(image_np)

                classification_result = {"roadwork_confidence": roadwork_score}

                # Store result in cache
                await self.cache.set(image_data, classification_result)
            finally:
                # Notify waiting requests
                event.set()
                del self._in_progress[image_data]

            result = {"result": classification_result, "cached": False}
            print(f"result computed: {result}")
            return result

        @self.app.get("/health")
        def health_check():
            if self.classifier is None:
                return {"status": "error", "message": "Classifier not initialized"}
            if not isinstance(self.classifier, NatixClassifier):
                return {"status": "error", "message": "Classifier instance invalid"}
            if not hasattr(self.classifier, "predict"):
                return {"status": "error", "message": "Classifier predict method missing"}
            if not callable(self.classifier.predict):
                return {"status": "error", "message": "Classifier predict method not callable"}
            return {"status": "ok"}
# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Natix Classifier API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    args = parser.parse_args()

    api = NatixClassifierAPI()
    uvicorn.run(api.app, host=args.host, port=args.port)
