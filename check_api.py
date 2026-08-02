from dotenv import load_dotenv
load_dotenv()

import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

try:
    resp = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents="test sentence"
    )

    print("✅ SUCCESS — API key and quota are working.")
    print("Vector length:", len(resp.embeddings[0].values))

except Exception as e:
    print("❌ FAILED:", e)