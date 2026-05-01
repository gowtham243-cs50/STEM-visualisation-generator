import re
import subprocess
import sys
import tempfile
from datetime import datetime
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, cast

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def _load_generator() -> Callable[..., Any]:
	"""Prefer Gemini pipeline, fallback to OpenAI pipeline if needed."""
	load_errors = []
	for module_name in ("geminimain", "main"):
		try:
			module = import_module(module_name)
			generator = getattr(module, "generate_manim_video_from_prompt", None)
			if callable(generator):
				return generator
		except Exception as exc:  # pragma: no cover - import failures are reported below.
			load_errors.append(f"{module_name}: {exc}")

	raise RuntimeError(
		"Could not load a video generator function from geminimain.py or main.py. "
		f"Errors: {' | '.join(load_errors)}"
	)


@lru_cache(maxsize=1)
def get_generator_fn() -> Callable[..., Any]:
	return _load_generator()


class VisualiseRequest(BaseModel):
	question: str = Field(..., min_length=1, max_length=3000)


app = FastAPI(title="Manim Generator API", version="1.0.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


def _safe_filename(question: str) -> str:
	cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", question).strip("_").lower()
	if not cleaned:
		cleaned = "generated_video"
	cleaned = cleaned[:40]
	timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
	return f"{cleaned}_{timestamp}.mp4"


def _render_video_bytes_from_code(manim_code: str) -> bytes:
	"""Render GeneratedScene and return the mp4 bytes."""
	with tempfile.TemporaryDirectory() as tmpdir:
		tmp_path = Path(tmpdir)
		script_path = tmp_path / "script.py"
		media_dir = tmp_path / "media"
		script_path.write_text(manim_code, encoding="utf-8")

		cmd = [
			"manim",
			str(script_path),
			"GeneratedScene",
			"-qk",
			"--media_dir",
			str(media_dir),
		]
		result = subprocess.run(cmd, capture_output=True, text=True, check=False)

		if result.returncode != 0:
			stderr_tail = (result.stderr or "").strip()[-3000:]
			raise RuntimeError(f"Manim render failed: {stderr_tail}")

		video_candidates = list(media_dir.glob("videos/**/GeneratedScene.mp4"))
		if not video_candidates:
			raise RuntimeError("Manim finished but no GeneratedScene.mp4 was found.")

		latest_video = max(video_candidates, key=lambda path: path.stat().st_mtime)
		return latest_video.read_bytes()


@app.get("/health")
async def health() -> Dict[str, str]:
	return {"status": "ok"}


@app.post("/visualise/")
async def visualise(payload: VisualiseRequest) -> Response:
	question = payload.question.strip()
	if not question:
		raise HTTPException(status_code=400, detail="question cannot be empty")

	try:
		generation_state = cast(
			Dict[str, Any],
			await run_in_threadpool(
			get_generator_fn(),
			question,
			2,
			),
		)
		manim_code = generation_state.get("manim_code")
		if not manim_code:
			error_message = generation_state.get("last_error") or "Failed to generate Manim code"
			raise RuntimeError(str(error_message))

		video_bytes = await run_in_threadpool(_render_video_bytes_from_code, str(manim_code))
		filename = _safe_filename(question)

		return Response(
			content=video_bytes,
			media_type="video/mp4",
			headers={
				"Content-Disposition": f'attachment; filename="{filename}"',
			},
		)
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=f"Video generation failed: {exc}") from exc


if __name__ == "__main__":
	import uvicorn

	uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
