import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EarColor Analyzer", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"ok": True, "service": "earcolor-analyzer"}

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    # Deployment smoke-test endpoint. The trained BTC inference layer is attached next.
    if not audio.filename:
        raise HTTPException(400, "Missing audio file")
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio file")
    return {"status": "backend-online", "filename": audio.filename, "bytes": len(data), "segments": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
