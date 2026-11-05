import base64
import binascii
from typing import Any

import cv2
import numpy as np


MAX_SIDE: int = 640  # fit 640x640, keeps tokens low and signs legible
JPEG_QUALITY: int = 90
MAX_DATA_URL_CHARS: int = 30_000_000   # about 22 MB of image before base64

# Downscale image to save token from my Anthropic Key 
def downscale(bgr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Fit the image inside max_side x max_side, keeping the aspect ratio; never upscale."""
    # Get the height and width of the image
    h, w = bgr.shape[:2]
    # Calculate the scale factor
    scale = max_side / max(h, w)
    # If the scale factor is greater than or equal to 1.0, return the image
    if scale >= 1.0:
        return bgr
    # Calculate the new size of the image
    size = ( max(1, round(w * scale)), max(1, round(h * scale)) )
    # Resize the image to the new size using the area interpolation method
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)


def image_block(bgr: np.ndarray, *, cache: bool = False) -> dict[str, Any]:
    """A base64 JPEG image content block for the Messages API."""
    ok, buf = cv2.imencode('.jpg', downscale(bgr), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError('could not encode the image as JPEG')
    block: dict[str, Any] = {
        'type'   : 'image',
        'source' : {'type': 'base64', 
                    'media_type': 'image/jpeg', # MIME type
                    'data': base64.standard_b64encode(buf.tobytes()).decode('ascii')},
    }
    if cache:
        block['cache_control'] = {'type': 'ephemeral'}
    # Return the block as a dictionary with the image data encoded in base64
    return block


def decode_data_url(data_url: str) -> np.ndarray:
    """Decode a data URL (or bare base64) into a BGR image; ValueError if it is not one."""
    if len(data_url) > MAX_DATA_URL_CHARS:
        raise ValueError('image is too large')
    # Split the data URL into the payload and the MIME type
    payload = data_url.split(',', 1)[1] if data_url.startswith('data:') else data_url
    try:
        # Decode the base64 encoded data into a raw bytes object
        raw = base64.b64decode(payload, validate=False)
    # If the base64 encoded data is not valid, raise an error
    except (binascii.Error, ValueError) as e:
        raise ValueError('image is not valid base64') from e
    # Decode the raw bytes object into a BGR image
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    # If the image could not be decoded, raise an error
    if bgr is None:
        raise ValueError('image could not be decoded')
    # Return the BGR image
    return bgr


