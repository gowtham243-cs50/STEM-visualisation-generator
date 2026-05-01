# Manim Generator API

An intelligent FastAPI server that automatically generates mathematical animations from natural language prompts using AI-powered code generation.

## 🎯 Overview

**Manim Generator API** leverages large language models (LLMs) to transform natural language questions into visually stunning Manim animations. The system intelligently breaks down problems into scenes, generates Manim Python code, and renders high-quality video outputs—all through a simple REST API.

### Key Capabilities

- **Dual LLM Support**: Uses Google Gemini (preferred) with fallback to OpenAI GPT-4
- **Scene Planning**: AI-driven scene breakdown with automatic Manim object selection
- **Code Generation & Repair**: Generates valid Manim code with automatic error correction
- **RAG Integration**: Semantic search over Manim documentation using Chroma vector store
- **Async Processing**: Non-blocking video rendering and response handling
- **CORS Enabled**: Ready for cross-origin requests from web frontends

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [Manim](https://docs.manim.community/) (for animation rendering)
- FFmpeg (dependency of Manim)
- LLM API keys (Google Gemini or OpenAI)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd manim-generator-api
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies** (create requirements.txt as needed)
   ```bash
   pip install fastapi uvicorn pydantic langchain langchain-google-genai langchain-openai langchain-chroma python-dotenv manim
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # Use either Gemini (preferred)
   GEMINI_API_KEY=your_gemini_api_key_here
   # or OpenAI
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running the Server

```bash
source .venv/bin/activate
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**API Documentation**: Visit `http://localhost:8000/docs` for interactive Swagger UI

## 📚 API Reference

### Health Check

**Endpoint**: `GET /health`

Returns server status.

**Response**:
```json
{
  "status": "ok"
}
```

### Generate Animation

**Endpoint**: `POST /visualise/`

Generates a Manim animation from a natural language prompt.

**Request Body**:
```json
{
  "question": "Draw a sine wave and show how it transforms into a cosine wave"
}
```

**Parameters**:
- `question` (string, required): Natural language description of the animation. Min 1, Max 3000 characters.

**Response**:
- Returns an MP4 video file with appropriate content headers

**Status Codes**:
- `200`: Success - MP4 video returned
- `400`: Invalid request (empty/missing question)
- `500`: Server error - Animation generation or rendering failed

**Example**:
```bash
curl -X POST "http://localhost:8000/visualise/" \
  -H "Content-Type: application/json" \
  -d '{"question": "Animate the Pythagorean theorem"}' \
  --output animation.mp4
```

### Video Sample

Try this POST request to generate the sample video shown below:

```bash
curl -X POST "http://localhost:8000/visualise/" \
   -H "Content-Type: application/json" \
   -d '{"question": "What is the formula for radius and circumference of a circle"}' \
   --output circle_formula.mp4
```

Video (Google Drive): https://drive.google.com/file/d/1eLN3c2SOGwXCiMngs5R-hINcVR7k2fMJ/view?usp=sharing

Direct preview (embed): https://drive.google.com/file/d/1eLN3c2SOGwXCiMngs5R-hINcVR7k2fMJ/preview

If embedding is supported in your viewer, use the preview URL to embed the video in an iframe.

## 🏗️ Project Structure

```
manim-generator-api/
├── server/
│   ├── __init__.py
│   └── server.py              # FastAPI application & endpoints
├── main.py                     # OpenAI-based generator pipeline
├── geminimain.py              # Google Gemini-based generator pipeline
├── test.py                     # Test utilities
├── test.rest                   # REST client requests
├── manim_chroma/              # Chroma vector store (Manim docs embeddings)
├── media/                      # Generated animations & assets
│   ├── images/
│   ├── Tex/                   # LaTeX fragments
│   ├── texts/
│   └── videos/
├── .env                        # Environment variables (excluded from git)
└── README.md                   # This file
```

## 🔧 How It Works

### Generation Pipeline (LangGraph)

The animation generation follows a structured multi-step pipeline:

1. **Scene Planning**: 
   - LLM breaks down the question into logical scenes
   - Outputs structured `ScenePlan` with scene titles, narration, and Manim objects

2. **Manim Documentation Retrieval** (RAG):
   - Retrieves relevant Manim documentation snippets using semantic search
   - Provides context for accurate code generation

3. **Code Generation**:
   - LLM generates valid Manim Python code based on the scene plan
   - Code includes a `GeneratedScene` class for rendering

4. **Error Correction**:
   - Validates generated code
   - Attempts automatic fixes for common errors

5. **Video Rendering**:
   - Executes Manim rendering with quality flags: `-qk` (2k quality)
   - Returns MP4 video file with appropriate naming

### LLM Configuration

**Gemini Pipeline** (`geminimain.py` - Preferred):
- Model: `gemini-2.5-flash`
- Embeddings: `models/gemini-embedding-001`
- Temperature: 0.2 (deterministic output)

**OpenAI Pipeline** (`main.py` - Fallback):
- Model: `gpt-4.1` (code generation), `gpt-4.1-mini` (planning)
- Temperature: 0.2
- Auto-fallback if Gemini unavailable

## 🎨 Technologies Used

- **FastAPI**: Modern async web framework
- **LangChain**: LLM orchestration and RAG framework
- **LangGraph**: Workflow state management
- **Google Gemini API**: Primary LLM for generation
- **OpenAI API**: Fallback LLM option
- **Chroma**: Vector database for semantic search
- **Manim**: Mathematical animation engine
- **Pydantic**: Data validation
- **CORS Middleware**: Cross-origin request support

## 📝 Configuration & Environment

### Required Environment Variables

- `GEMINI_API_KEY`: Google Cloud API key for Gemini models
- `GOOGLE_API_KEY`: Alternative name for Gemini API key
- `OPENAI_API_KEY`: OpenAI API key (optional, for fallback)

### Optional Configuration

Edit generation parameters in `geminimain.py` or `main.py`:
- Temperature (0.0-1.0): Lower = more deterministic
- Model selection: Change model names in LLM initialization
- Retrieval K: Adjust `MANIM_RETRIEVAL_K` for more/fewer documentation snippets

## 🧪 Testing

Use the provided `test.rest` file for API testing in VSCode REST Client or similar tools:

```rest
@base = http://localhost:8000

### Health Check
GET {{base}}/health

### Generate Animation
POST {{base}}/visualise/
Content-Type: application/json

{
  "question": "Create an animation showing the area under a sine curve"
}
```

## 🐛 Troubleshooting

### "Could not find a video generator function"
- Ensure `geminimain.py` or `main.py` has a `generate_manim_video_from_prompt` function
- Check that API keys are properly set in `.env`

### "Manim render failed"
- Verify Manim and FFmpeg are installed: `manim --version`
- Check Manim documentation for syntax issues in generated code
- Review error logs in stderr

### "Chroma vector store not found"
- RAG features are optional; the system works without the `manim_chroma/` directory
- To enable, populate Chroma with Manim documentation

### API returning 500 errors
- Check server logs for detailed error messages
- Verify LLM API keys are valid and have sufficient quota
- Ensure temp directory is writable for video generation

## 📄 License

Specify your license here (MIT, Apache 2.0, etc.)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear commit messages

## 📞 Support

For issues, questions, or suggestions, please open an issue on the repository.

---

**Built with ❤️ for mathematical visualization**
