# Introducing Vanna AI
Text-To-SQL Playground

This GitHub repository is a companion resource to the Medium article [Text-to-SQL Just Got Easier: Meet Vanna AI, Your Text-to-SQL Assistant
](https://medium.com/mitb-for-all/text-to-sql-just-got-easier-meet-vanna-ai-your-rag-powered-sql-sidekick-e781c3ffb2c5)

[![Watch the video](https://raw.githubusercontent.com/tituslhy/glowing-engine/main/images/bullrun_thumbnail.png)](https://raw.githubusercontent.com/tituslhy/glowing-engine/main/media/bullrun_app.mp4)

## Setup
This repository uses the [uv package installer](https://docs.astral.sh/uv/pip/packages/). 

To create a virtual environment with the dependencies installed, simply type in your terminal:
```
uv sync
```

You'll need to ingest the data into an SQLite database before you can interact with the app. Simply run the codes in `./notebooks/setup_db.ipynb` to ingest the ticker information from [Yahoo Finance](https://github.com/ranaroussi/yfinance).

## To run Chainlit app
This spins up your application on port 8000
```
chainlit run app.py
```

## Easy Deploy (AWS EC2)

This project can now bootstrap its own SQLite demo database on first run.
You do not need to copy `database/stocks.db` manually.

### 1) Create environment variables
Create `.env` in the project root with:
```
GROQ_API_KEY=your_key_here
GROQ_API_BASE=https://api.groq.com/openai/v1
```

### 2) Install dependencies
Option A (recommended):
```
uv sync
```

Option B (without uv):
```
python -m pip install -r requirements.txt
```

### 3) Start the app
From the `glowing-engine` directory:
```
python -m chainlit run app.py --host 0.0.0.0 --port 8000
```

### 4) Open it
Use `http://<your-ec2-public-ip>:8000` (if your security group allows port 8000).
