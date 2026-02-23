# Streamlit Data App

This repository contains a simple Streamlit application with an optional
backend API that can be run using `uvicorn`.

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd streamlit-data-app
   ```

2. **Install dependencies and create virtual environment**
   The `pyproject.toml` is in the `streamlit-ui/` directory. Navigate there
   and run `uv sync`:
   ```bash
   cd streamlit-ui
   uv sync
   ```
   If you don't already have `uv`, install it first:
   ```bash
   pip install uv
   ```

3. **Add environment variables**
   Create a `.env` file in the project root and add any required
   configuration variables. An example file is provided as
   `env.example`.

    Note: Credentials are in vaultwarden

   ```bash
   cp env.example .env
   # then edit .env with appropriate values
   ```

## Running the app

### Streamlit UI

To start the Streamlit interface, navigate to the `streamlit-ui/` directory
and run:
```bash
cd streamlit-ui
uv run streamlit run app.py
```

## Notes

- Ensure the `.env` file is loaded by your application (e.g. using
  `python-dotenv` or similar) so that configuration values are available.

- The `env.example` file shows expected variables and can be used as a
  template.

- Test data is in testdata folder