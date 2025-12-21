import base64
import struct
import numpy as np

from typing import List


def cosine_similarity(a: List[float], b: List[float]) -> float:
	a, b = np.array(a), np.array(b)
	return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def encode_vector(vec: List[float]) -> str:
	return base64.b64encode(struct.pack(f"{len(vec)}f", *vec)).decode("utf-8")

def decode_vector(b64: str) -> List[float]:
	raw = base64.b64decode(b64)
	count = len(raw) // 4
	return list(struct.unpack(f"{count}f", raw))
