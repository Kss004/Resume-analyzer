from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from mongo_utils import get_mongodb_connection
from functions import extract_text_from_pdf, get_gpt_analysis
from rag_utils import search_similar_template
from agent_utils import format_agent_prompt
from openai import OpenAI
from bson import ObjectId
from dotenv import load_dotenv
import os
import io

load_dotenv()
app = FastAPI()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client, db, fs = get_mongodb_connection()  # Use global fs consistently


@app.post("/upload/")
async def analyze_resume(resume: UploadFile = File(...), jd: UploadFile = File(...)):
    if not resume.filename.endswith(".pdf") or not jd.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    resume_bytes = await resume.read()
    jd_bytes = await jd.read()

    resume_text = extract_text_from_pdf(resume_bytes)
    jd_text = extract_text_from_pdf(jd_bytes)
    gpt_analysis = get_gpt_analysis(resume_text)

    # Step 1: Store resume in GridFS
    resume_fs_id = fs.put(resume_bytes, filename=resume.filename, metadata={"source": "user_upload"})

    # Step 2: Semantic search to get top templates
    top_template_matches = search_similar_template(jd_text) or [
        {
            "template_number": 1,
            "template_preview_text": "No strong match found, but here’s the closest resume template we have.",
            "template_file_id": None,
            "download_url": None
        }
    ]

    # Step 3: Use agent prompt to suggest best match
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

    # Step 4: Format clean response
    return JSONResponse(content={
        "resume_fs_id": str(resume_fs_id),
        "resume_download_url": f"/download_resume/{resume_fs_id}",
        "analysis": gpt_analysis,
        "final_suggestion": final_suggestion,
        "template_matches": top_template_matches
    })


@app.get("/download_resume/{file_id}")
def download_resume(file_id: str):
    try:
        file = fs.get(ObjectId(file_id))
        return StreamingResponse(file, media_type="application/pdf", headers={
            "Content-Disposition": f"attachment; filename={file.filename}"
        })
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Resume not found: {str(e)}")


@app.get("/download_template_by_id/{template_file_id}")
def download_template_by_id(template_file_id: str):
    try:
        file = fs.get(ObjectId(template_file_id))
        return StreamingResponse(file, media_type="application/pdf", headers={
            "Content-Disposition": f"attachment; filename={file.filename}"
        })
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Template not found: {str(e)}")
