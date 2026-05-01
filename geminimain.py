import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional, TypedDict, cast ,Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

from langchain_core.tools import tool
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Gemini via LangChain
from langchain_chroma import Chroma

load_dotenv()
# -------------------------------------------------------------------
# 1. Structured models for scene plan (structured output)
# -------------------------------------------------------------------

class Scene(BaseModel):
    title: str = Field(..., description="Short scene title")
    narration: str = Field(..., description="Spoken explanation for this scene")
    objects: List[str] = Field(
        ...,
        description="Important Manim objects for this scene, e.g. MathTex, Text, Axes",
    )
    notes: Optional[str] = Field(
        None,
        description="Optional extra guidance for the animator",
    )


class ScenePlan(BaseModel):
    scenes: List[Scene] = Field(
        ...,
        description="Ordered list of scenes for the video explanation",
    )


# -------------------------------------------------------------------
# 2. LLM clients (Gemini via ChatGoogleGenerativeAI)
# -------------------------------------------------------------------
# Make sure GOOGLE_API_KEY is set in your environment.

planner_llm_raw = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-3.1-pro-preview" etc.
    temperature=0.2,
)

# Native structured output using Gemini JSON schema mode.[web:102]
planner_llm = planner_llm_raw.with_structured_output(
    ScenePlan,
    method="json_schema",
)

coder_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # use a stronger model for codegen/debug
    temperature=0.2,
)


# -------------------------------------------------------------------
# 3. Manim docs retriever tool (RAG stub)
# -------------------------------------------------------------------

MANIM_CHROMA_DIR = Path(__file__).resolve().parent / "manim_chroma"
MANIM_EMBEDDING_MODEL = "models/gemini-embedding-001"
MANIM_RETRIEVAL_K = 4


def _get_gemini_api_key() -> Optional[str]:
    # Accept both variable names so retriever works across local setups.
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@lru_cache(maxsize=1)
def _get_manim_vectorstore() -> Optional[Chroma]:
    if not MANIM_CHROMA_DIR.exists():
        return None

    api_key = _get_gemini_api_key()
    if not api_key:
        return None

    embeddings = GoogleGenerativeAIEmbeddings(
        model=MANIM_EMBEDDING_MODEL,
        api_key=SecretStr(secret_value=api_key),
    )
    return Chroma(
        persist_directory=str(MANIM_CHROMA_DIR),
        embedding_function=embeddings,
    )

def manim_retriever(query: str) -> List[Document]:
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    vectorstore = _get_manim_vectorstore()
    if vectorstore is None:
        return []

    try:
        return vectorstore.similarity_search(cleaned_query, k=MANIM_RETRIEVAL_K)
    except Exception:
        return []


@tool
def manim_doc_search(query: str) -> str:
    """Search Manim documentation and examples for information about classes, methods, or usage."""
    docs = manim_retriever(query)
    if not docs:
        return "No relevant Manim docs found in the local index."
    joined = "\n\n".join(d.page_content[:2000] for d in docs)
    return joined


TOOLS = [manim_doc_search]


# -------------------------------------------------------------------
# 4. Shared state for LangGraph
# -------------------------------------------------------------------

class ManimState(TypedDict):
    user_prompt: str
    plan: Optional[Dict[str, Any]]   # dict version of ScenePlan
    manim_code: Optional[str]
    last_error: Optional[str]
    attempts: int
    max_attempts: int
    video_path: Optional[str]


# -------------------------------------------------------------------
# 5. Planner node: prompt -> structured scene plan
# -------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are an expert high-school STEM teacher and Manim animation designer.
Given a topic prompt, design a step-by-step, scene-based explanation.
Use clear, concise language appropriate for high-school students.
"""

def planner_node(state: ManimState) -> ManimState:
    user_prompt = state["user_prompt"]

    messages = [
        ("system", PLANNER_SYSTEM_PROMPT),
        ("user", f"Topic: {user_prompt}"),
    ]

    scene_plan = cast(ScenePlan, planner_llm.invoke(messages))    
    print(f"\n--- Planner LLM Output ---\n{scene_plan}\n--------------------------\n")
    state["plan"] = scene_plan.model_dump()
    return state


# -------------------------------------------------------------------
# 6. Codegen + correction node
# -------------------------------------------------------------------

CODER_SYSTEM_PROMPT = """You are an expert Manim CE developer.
Generate a complete, runnable Python script that defines EXACTLY ONE Scene subclass named GeneratedScene.

Requirements:

Use: from manim import *
No external imports beyond manim.
No file I/O, OS or network operations.
Use the provided scene plan and object list.
The script must be runnable with: manim script.py GeneratedScene -qk
IMPORTANT camera rule:
Never use self.camera.animate.
If camera pan/zoom/frame motion is needed, GeneratedScene must inherit from MovingCameraScene.
Animate camera via self.camera.frame.animate (example style: self.camera.frame.animate.set(width=14)).
Respond with ONLY valid Python code (no backticks, no comments outside code).
"""

CORRECTOR_SYSTEM_PROMPT = """You are an expert Manim CE developer and debugger.
You are given:

The current Manim script.
The runtime error message.
Optional snippets from Manim documentation.
Produce a FIXED version of the script that:

Keeps the same visual / pedagogical intent where possible.
Fixes the error.
Remains a single file with one Scene subclass: GeneratedScene.
Enforces Manim CE camera compatibility:
Do not use self.camera.animate.
For camera movement/zoom, use MovingCameraScene and self.camera.frame.animate.
Return ONLY valid Python code.
"""

def generate_or_fix_code(
    plan: Optional[Dict[str, Any]],
    existing_code: Optional[str],
    error: Optional[str],
) -> str:
    """
    If existing_code is None: fresh generation from plan.
    If existing_code + error provided: repair step, using manim_doc_search.
    """
    if existing_code is None:
        scenes_text = repr(plan)
        messages = [
            ("system", CODER_SYSTEM_PROMPT),
            (
                "user",
                "Here is the scene plan as a Python dict (from a structured schema):\n"
                f"{scenes_text}\n\nGenerate the script.",
            ),
        ]
    else:
        query = f"Manim error: {error}\n\nRelevant classes and objects likely used in this script."
        docs_str = manim_doc_search.invoke({"query": query})
        messages = [
            ("system", CORRECTOR_SYSTEM_PROMPT),
            (
                "user",
                f"RUNTIME ERROR:\n{error}\n\n"
                f"CURRENT SCRIPT:\n{existing_code}\n\n"
                f"MANIM DOC SNIPPETS:\n{docs_str}\n\n"
                "Return a corrected script.",
            ),
        ]

    resp = coder_llm.invoke(messages)
    print(f"\n--- Coder LLM Output ---\n{resp.content}\n------------------------\n")
    code = str(resp.content).strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    return code


def run_manim_script(manim_code: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Write code to a temp file, run `manim` as a subprocess, and capture errors.
    Returns: (success, error_message, video_path)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(manim_code)

        cmd = [
            "manim",
            script_path,
            "GeneratedScene",
            "-qk",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate()
        success = proc.returncode == 0

        if success:
            video_path = os.path.join(tmpdir, "media", "videos")
            return True, None, video_path
        else:
            return False, stderr, None


def codegen_and_correction_node(state: ManimState) -> ManimState:
    plan = state["plan"]
    attempts = state["attempts"]
    max_attempts = state["max_attempts"]
    existing_code = state["manim_code"]
    last_error = state["last_error"]

    manim_code = generate_or_fix_code(plan, existing_code, last_error)
    state["manim_code"] = manim_code

    success, error, video_path = run_manim_script(manim_code)

    if success:
        state["video_path"] = video_path
        state["last_error"] = None
        return state

    state["last_error"] = error
    attempts += 1
    state["attempts"] = attempts

    if attempts >= max_attempts:
        return state

    return state


# -------------------------------------------------------------------
# 7. Build LangGraph: planner -> codegen+correction (cyclic)
# -------------------------------------------------------------------

graph = StateGraph(ManimState)

graph.add_node("planner", planner_node)
graph.add_node("codegen_and_correction", codegen_and_correction_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "codegen_and_correction")

def next_step(state: ManimState) -> str:
    if state.get("video_path"):
        return END
    if state["attempts"] >= state["max_attempts"]:
        return END
    return "codegen_and_correction"

graph.add_conditional_edges(
    "codegen_and_correction",
    next_step,
    {
        "codegen_and_correction": "codegen_and_correction",
        END: END,
    },
)

app = graph.compile()


# -------------------------------------------------------------------
# 8. Convenience entrypoint
# -------------------------------------------------------------------

def generate_manim_video_from_prompt(prompt: str, max_attempts: int = 4) -> ManimState:
    initial_state: ManimState = {
        "user_prompt": prompt,
        "plan": None,
        "manim_code": None,
        "last_error": None,
        "attempts": 0,
        "max_attempts": max_attempts,
        "video_path": None,
    }
    final_state = app.invoke(initial_state)
    return cast(ManimState, final_state)


if __name__ == "__main__":
    user_prompt = input("Enter topic prompt: ")
    result = generate_manim_video_from_prompt(user_prompt)
    if result.get("video_path"):
        print("Video generated at:", result["video_path"])
    else:
        print("Failed to generate video.")
        print("Last error:", result.get("last_error"))