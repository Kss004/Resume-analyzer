from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from mongo_utils import get_mongodb_connection
from functions import extract_text_from_pdf, get_gpt_analysis
from rag_utils import search_similar_template
from agent_utils import format_agent_prompt
from openai import OpenAI
from bson import ObjectId
from dotenv import load_dotenv
import os
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client, db, fs = get_mongodb_connection()

@app.post("/upload/")
async def analyze_resume(
    background_tasks: BackgroundTasks,
    resume: UploadFile = File(...),
    jd: UploadFile = File(...)
):
    """
    Analyze a resume against a job description and suggest matching templates
    """
    try:
        if not resume.filename.endswith(".pdf") or not jd.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Read and process files
        resume_bytes = await resume.read()
        jd_bytes = await jd.read()

        resume_text = extract_text_from_pdf(resume_bytes)
        jd_text = extract_text_from_pdf(jd_bytes)
        
        if not resume_text or not jd_text:
            raise HTTPException(status_code=400, detail="Could not extract text from one or more PDF files")

        # Store resume in GridFS with metadata
        resume_fs_id = fs.put(
            resume_bytes,
            filename=resume.filename,
            metadata={
                "source": "user_upload",
                "original_filename": resume.filename,
                "content_type": resume.content_type,
                "file_size": len(resume_bytes)
            }
        )
        logger.info(f"Stored resume in GridFS with ID: {resume_fs_id}")

        # Get GPT analysis in background
        background_tasks.add_task(get_gpt_analysis, resume_text)

        # Search for similar templates with improved scoring
        top_template_matches = search_similar_template(
            jd_text,
            top_k=3,
            score_threshold=float(os.getenv("SCORE_THRESHOLD", 0.7))
        )
        
        if not top_template_matches:
            logger.warning("No template matches found above threshold")
            top_template_matches = [{
                "template_number": 1,
                "template_title": "No Strong Match",
                "template_preview_text": "No strong match found, but here's the closest resume template we have.",
                "template_file_id": None,
                "similarity_score": 0.0,
                "download_url": None,
                "metadata": {"category": "General"}
            }]

        # Use agent prompt to suggest best match
        prompt = format_agent_prompt(
            jd_text,
            resume_text,
            [match["template_preview_text"] for match in top_template_matches]
        )

        agent_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
        )
        final_suggestion = agent_response.choices[0].message.content

        # Format response with enhanced template information
        return JSONResponse(content={
            "resume_fs_id": str(resume_fs_id),
            "resume_download_url": f"/download_resume/{resume_fs_id}",
            "analysis": get_gpt_analysis(resume_text),
            "final_suggestion": final_suggestion,
            "template_matches": top_template_matches
        })

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download_resume/{file_id}")
async def download_resume(file_id: str):
    """
    Download a resume by its GridFS ID
    """
    try:
        file = fs.get(ObjectId(file_id))
        return StreamingResponse(
            file,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={file.filename}",
                "Content-Type": "application/pdf"
            }
        )
    except Exception as e:
        logger.error(f"Error downloading resume {file_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Resume not found: {str(e)}")

@app.get("/download_template_by_id/{template_file_id}")
async def download_template_by_id(template_file_id: str):
    """
    Download a template by its GridFS ID
    """
    try:
        file = fs.get(ObjectId(template_file_id))
        return StreamingResponse(
            file,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={file.filename}",
                "Content-Type": "application/pdf"
            }
        )
    except Exception as e:
        logger.error(f"Error downloading template {template_file_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Template not found: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        return {"status": "healthy", "mongodb": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")
