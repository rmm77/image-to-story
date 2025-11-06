import os, boto3
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from datetime import timedelta
from anthropic import Anthropic

app = FastAPI(title="Image → Story (Claude)")

# --- S3 client for pre-signed URLs ---
s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
BUCKET = os.getenv("S3_BUCKET")

# --- Anthropic client ---
anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")  # latest Sonnet
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1200"))  # ~1–2 pages


class UploadReq(BaseModel):
    filename: str
    content_type: str


@app.post("/upload-url")
def upload_url(req: UploadReq):
    if not BUCKET:
        raise HTTPException(500, "S3_BUCKET not configured")
    key = f"uploads/{req.filename}"
    put_url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": req.content_type},
        ExpiresIn=int(timedelta(minutes=10).total_seconds()),
    )
    return {"put_url": put_url, "key": key}


@app.get("/generate")
def generate_story(
    key: str = Query(..., description="S3 object key returned from /upload-url"),
    style: str = Query("literary, lyrical, warm, cohesive", description="Optional style tags"),
    length_hint: str = Query("700-900 words", description="Rough target length"),
    audience: str = Query("general", description="Target audience"),
    temperature: float = Query(0.8, ge=0.0, le=1.0),
):
    if not BUCKET:
        raise HTTPException(500, "S3_BUCKET not configured")

    # Create a short-lived GET URL for Claude to read the image
    get_url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=300
    )

    system_prompt = (
        "You are an award-winning author with an eye for evocative detail. "
        "Write a cohesive short story inspired by the image. Favor scene, character, "
        "and subtle emotional arc over object listing. Show, don't tell."
    )

    user_text = (
        f"Write a {length_hint} story inspired by this image for a(n) {audience} audience.\n"
        f"Tone/style: {style}. Avoid bullet points or captions. Keep the narrative unified."
    )

    try:
        resp = anthropic.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "source": {"type": "url", "url": get_url}},
                ],
            }],
        )
        story = "".join(part.text for part in resp.content if getattr(part, "type", "") == "text")
        return {"story": story}
    except Exception as e:
        raise HTTPException(500, f"Anthropic error: {e}")
