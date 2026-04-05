# Contributing to feataz

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/feataz.git
   cd feataz
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
   pip install -e ".[dev,sklearn]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

```bash
python -m pytest tests/ -v --tb=short
```

## Code Style

This project uses:
- **ruff** for linting and formatting
- **mypy** for type checking

Run linting:
```bash
ruff check src/ tests/
ruff format src/ tests/
```

Run type checking:
```bash
mypy src/
```

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all tests pass and linting is clean
4. Open a pull request with a description of your changes
