"""
Module 3 - Task 5 - FastAPI wrapper.

Run locally:
  uvicorn main:app --host 0.0.0.0 --port 7860

POST /ask  {"query": "..."}  ->  AskResponse {answer, sources, confidence}
"""
from fastapi import FastAPI
from schema import AskRequest, AskResponse
from graph import run_query

app = FastAPI(title="Zepto Support Assistant")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return run_query(request.query)
