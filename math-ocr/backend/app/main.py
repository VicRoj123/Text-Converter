from fastapi import FastAPI, UploadFile, File
from app.modelsvc import predict_latex  # you implement this

app = FastAPI()

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/recognize")
async def recognize(img: UploadFile = File(...)):
    data = await img.read()
    latex, conf = predict_latex(data)
    return {"latex": latex, "confidence": conf}
