import asyncio
import json
from typing import List
import pdfplumber

from agno.agent import Agent
from agno.models.ollama import Ollama
from pydantic import BaseModel, Field


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


class PersonalInfo(BaseModel):
    name: str = Field(..., description="Full name of the candidate.")
    email: str = Field(..., description="Email address of the candidate.")
    phone: str = Field(..., description="Phone number of the candidate.")
    address: str = Field(..., description="Current address of the candidate.")


class EducationDetails(BaseModel):
    degree: str = Field(..., description="Degree obtained.")
    institution: str = Field(..., description="Name of the institution.")
    year_of_passing: int = Field(..., description="Year of graduation.")


class WorkExperience(BaseModel):
    company: str = Field(..., description="Name of the previous company.")
    role: str = Field(..., description="Job title.")
    duration: str = Field(..., description="Duration of employment.")
    location: str = Field(..., description="Location of the company.")


class ResumeData(BaseModel):
    personal_info: PersonalInfo = Field(..., description="Personal information of the candidate.")
    education: List[EducationDetails] = Field(..., description="List of educational qualifications.")
    work_experience: List[WorkExperience] = Field(..., description="List of work experiences.")


structured_output_agent = Agent(
    model=Ollama(id="llama3.2"),
    description="You extract resume details.",
    response_model=ResumeData,
    structured_outputs=True,
)


def run_agents(pdf_path: str, output_json: str):
    text = extract_text_from_pdf(pdf_path)
    response =  structured_output_agent.run(text) 
    with open(output_json, "w") as json_file:
        json.dump(response.content.model_dump(), json_file, indent=4) 
    print(f"Resume data saved to {output_json}")


run_agents(pdf_path="Resume.pdf", output_json="resume_data.json")
