# Multi-Agent Research System

A sophisticated multi-agent system designed to automate the research process by finding, explaining, and evaluating scientific papers using advanced AI techniques.

## Features

- **Paper Search**: Find relevant research papers based on user queries
- **Abstract Explanation**: Generate clear, concise explanations of complex research abstracts
- **Quality Assurance**: Automatically evaluate and improve explanations for accuracy and clarity
- **Vector Database**: Store and retrieve papers using semantic similarity
- **Analytics**: Track quality metrics and identify improvement opportunities
- **Web Interface**: User-friendly Streamlit UI for interacting with the system

## Architecture

The system uses a multi-agent architecture with specialized agents:

1. **ResearcherAgent**: Searches for and retrieves relevant research papers
2. **WriterAgent**: Generates explanations of paper abstracts in accessible language
3. **QualityAssuranceAgent**: Evaluates explanations for factual accuracy, comprehensiveness, clarity, and conciseness
4. **EnhancedQualityAssuranceAgent**: Advanced version with hallucination detection and focused regeneration

## Requirements

- Python 3.8+
- Google Gemini API key
- ChromaDB
- Streamlit
- FastAPI

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/researcher-agent.git
   cd researcher-agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Google API key:
   ```
   export GOOGLE_API_KEY="your_api_key_here"
   ```

## Usage

### Start the MCP Server

The Multi-agent Coordination Protocol (MCP) server handles communication between agents:

```
python main.py --mode server --max-rpm 10
```

### Start the Web UI

Launch the Streamlit interface with:

```
python start_ui.py
```

or

```
streamlit run web_ui.py --server.fileWatcherType=none
```

### Command Line Interface

Run a full research pipeline from the command line:

```
python main.py --mode cli --query "quantum machine learning" --papers 5
```

## Configuration

- Adjust rate limits: Use `--max-rpm` when starting the server
- ChromaDB settings: Located in `run.py`
- Streamlit settings: Located in `.streamlit/config.toml`

## How It Works

1. The system searches for relevant papers based on user queries
2. For each paper, it generates a clear explanation of the abstract
3. The QA agent evaluates the explanation for quality and accuracy
4. If issues are detected, explanations are regenerated with improved accuracy
5. Results are saved as JSON for future reference

## Output Files

Research results are saved to timestamped JSON files with the format:
- `research_results_YYYYMMDD-HHMMSS.json`

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
