import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cancer_config import CANCER_TYPES
from predict_utils import predict_user_csv, list_available_models
from agent import route_to_model  # Import the mock agent

app = FastAPI(title="Cancer Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001", "https://gene-x.vercel.app"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Cancer Prediction API"}

@app.get("/api/cancers")
def get_cancers():
    return {"cancers": list_available_models()}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv file.")

    tmp_path = os.path.join(tempfile.gettempdir(), f"upload_{file.filename}")
    try:
        with open(tmp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        
        # 1. Agent automatically routes the file
        cancer_type = route_to_model(tmp_path)
        
        # 2. Run prediction on the routed model
        result = predict_user_csv(cancer_type, tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
