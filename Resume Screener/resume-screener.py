import fitz
import ollama
import json

# Read PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Prompt to screen resume
def screen_resume(resume_text, job_description):
    prompt = f"""
    You are a Senior Technical Recruiter with 20 years of experience. 
    Your goal is to objectively evaluate a candidate based on a Job Description (JD).
    
    JOB DESCRIPTION:
    {job_description}
    
    CANDIDATE RESUME:
    {resume_text}
    
    TASK:
    Analyze the resume against the JD. Look for key skills, experience levels, and project relevance.
    Be strict but fair. "React" matches "React.js". "AWS" matches "Amazon Web Services".
    
    OUTPUT FORMAT:
    Provide the response in valid JSON format only. Do not add any conversational text.
    structure:
    {{
        "candidate_name": "extracted name",
        "match_score": "0-100",
        "key_strengths": ["list of 3 key strengths"],
        "missing_critical_skills": ["list of missing skills"],
        "recommendation": "Interview" or "Reject",
        "reasoning": "A 2-sentence summary of why."
    }}
    """
    # you can use any other model, I am using gemma as its running on low RAM
    response = ollama.chat(model='gemma:2b',
        messages=[
            {'role': 'user', 'content': prompt},
        ])
    
    return response['message']['content']


# 1. Define Job Description
# job_description = """
# We are looking for a Senior AI enabled Engineer.
# Must have:
# - Python
# - Experience with SQL
# - Experience with LLMs, RAGs, MCP, FastAPI
# - Good communication skills
# Nice to have:
# - Experience with AWS or Cloud deployment
# - Experience with AIDLC
# """


job_description = input('Please enter job description in brief: ')

# 2. Load the Resume
try:
    resume_text = extract_text_from_pdf("<YOUR_PDF_RESUME_PATH>")
    print(f"Resume loaded. Length: {len(resume_text)} characters.")
except Exception as e:
    print(f"Error loading resume: {e}")
    exit()

# 3. The Screening
print("AI is analyzing the candidate... (this may take a few seconds on local hardware)")
result_json_string = screen_resume(resume_text, job_description)

# 4. Display Results
try:
    clean_json = result_json_string.replace("```json", "").replace("```", "").strip()
    result_data = json.loads(clean_json)
    print(result_data)
    print("\n--- SCREENING REPORT ---")
    print(f"Candidate: {result_data.get('candidate_name', 'Unknown')}")
    print(f"Score: {result_data.get('match_score')}/100")
    print(f"Decision: {result_data.get('recommendation').upper()}")
    print(f"Reasoning: {result_data.get('reasoning')}")
    print(f"Missing Skills: {', '.join(result_data.get('missing_critical_skills', []))}")
    
except json.JSONDecodeError:
    print("Failed to parse JSON. Raw output:")
    print(result_json_string) 