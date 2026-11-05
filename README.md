# mlops-bits

A Python project for MLOps workflows built with FastAPI and packaged using `uv`.

---

## Prerequisites

- **Python**: `>= 3.14`
- **[uv](https://docs.astral.sh/uv/)**: Recommended package and project manager for Python

To install `uv` (if not already installed):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew (macOS)
brew install uv
```

---

## Installation & Setup

Clone the repository and install dependencies using `uv`:

```bash
# Clone the repository
git clone https://github.com/nkantkumar/mlops-bits.git
cd mlops-bits

# Install dependencies (including dev tools) into a virtual environment
uv sync
```

---

## Execution

### Run CLI Application

Run the registered command-line script:

```bash
uv run mlops-bits
```

Or run via the module directly:

```bash
uv run python -c "import mlops_bits; mlops_bits.main()"
```

Output:
```text
Hello from mlops-bits!
```

---

## Running Tests

Execute the test suite using `pytest`:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v
```

---

## Code Quality & Linting

### Linting & Formatting (Ruff)

```bash
# Check for lint errors
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking (Mypy)

```bash
uv run mypy src tests
```

---

## Project Structure

```text
mlops-bits/
├── src/
│   └── mlops_bits/
│       └── __init__.py       # Application entry point and logic
├── tests/
│   ├── __init__.py
│   └── test_main.py          # Pytest suite
├── pyproject.toml            # Project configuration and dependencies
├── uv.lock                   # Deterministic lockfile
└── README.md
```
