import os
import io
import requests
from typing import List, TypedDict
from PIL import Image, ImageOps, ImageEnhance
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# --- SCHEMAS ---
class BiomarkerItem(BaseModel):
    parameter_name: str = Field(description="Name of the biomarker, e.g., Fasting Glucose, HbA1c")
    value: str = Field(description="Measured numerical value with units, e.g., 138 mg/dL")
    status: str = Field(description="Parameter status: High, Low, or Normal")
    plain_language_meaning: str = Field(description="Simple explanation of what this result means for health")
    recommended_foods: List[str] = Field(description="Specific recommended foods to manage this biomarker level")
    foods_to_avoid: List[str] = Field(description="Foods to avoid or limit based on this parameter level")

class BloodReportAnalysis(BaseModel):
    patient_name: str = Field(description="Patient identifier or name extracted from raw text, default to 'Patient' if not found")
    biomarker_analysis: List[BiomarkerItem] = Field(description="Complete analysis for every single biomarker found")

class AgentState(TypedDict):
    raw_ocr_text: str
    analysis_result: BloodReportAnalysis

# --- OCR ENGINE ---
def extract_text_via_ocr_api(image_bytes: bytes, api_key: str) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    
    # 1. Fix rotation based on EXIF camera tags
    img = ImageOps.exif_transpose(img)
    
    # 2. Convert to standard RGB
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
        
    # 3. Enhance contrast slightly to sharpen faint numbers
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    # 4. Resize boundary (Lowered to 1000px to speed up API transfer)
    img.thumbnail((1000, 1000))
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    optimized_bytes = buffer.getvalue()

    url = "https://api.ocr.space/parse/image"
    payload = {
        'apikey': api_key,
        'language': 'eng',
        'isOverlayRequired': False,
        'OCREngine': 2,        # Multi-column table engine
        'isTable': True,        # Forces strict row/column table alignment
        'scale': True,
        'detectOrientation': True  # Auto-rotates image if upside down or sideways
    }
    
    files_payload = [('file', ('report.jpg', optimized_bytes, 'image/jpeg'))]
    
    response = requests.post(url, data=payload, files=files_payload, timeout=20)
    result = response.json()
    
    parsed = result.get("ParsedResults", [])
    raw_text = parsed[0].get("ParsedText", "") if parsed else ""
    
    # Debug print in VS Code terminal to inspect exact text returned from OCR
    print("\n--- RAW OCR TEXT extracted ---")
    print(raw_text)
    print("-----------------------------\n")
    
    return raw_text

# --- LANGGRAPH WORKFLOW BUILDER ---
def build_agent_graph(groq_api_key: str):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)
    structured_llm = llm.with_structured_output(BloodReportAnalysis)

    def analyze_text_node(state: AgentState) -> AgentState:
        system_prompt = (
            "You are an expert medical AI pathology analyst.\n"
            "CRITICAL EXTRACTION RULES FOR TABULAR LAB DATA:\n"
            "1. Read the raw text line-by-line. A single line typically contains: [Parameter Name] [Result] [Normal Range] [Units].\n"
            "2. Do NOT confuse numbers in the 'Normal Range' column with the 'Result' column. For example, if ESR is 2 and Range is Up to 15, ESR result is 2 (Normal).\n"
            "3. Cross-check each parameter's extracted value against its listed reference range to accurately assign 'High', 'Low', or 'Normal'.\n"
            "4. Output clean values and units without OCR noise or symbol corruption."
        )
        response = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"LAB REPORT RAW TEXT:\n{state['raw_ocr_text']}")
        ])
        return {"analysis_result": response}

    builder = StateGraph(AgentState)
    builder.add_node("text_analyzer", analyze_text_node)
    builder.add_edge(START, "text_analyzer")
    builder.add_edge("text_analyzer", END)
    return builder.compile()
