import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from agent import extract_text_via_ocr_api, build_agent_graph

load_dotenv()

OCR_API_KEY = os.getenv("OCR_SPACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="NutriAgent API")

# Allow frontend HTML/JS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_graph = build_agent_graph(GROQ_API_KEY)

@app.post("/api/analyze")
async def analyze_report(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # 1. OCR Step
        raw_text = extract_text_via_ocr_api(contents, OCR_API_KEY)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not read text from image.")
            
        # 2. Analysis Step
        output = agent_graph.invoke({"raw_ocr_text": raw_text})
        analysis = output["analysis_result"]
        
        # 3. Direct JSON Response (No Google Sheets needed)
        return {
            "status": "success",
            "patient_name": analysis.patient_name,
            "biomarkers": [item.model_dump() for item in analysis.biomarker_analysis]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
