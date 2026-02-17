FROM python:3.10-slim

WORKDIR /app

# avoid python buffering logs
ENV PYTHONUNBUFFERED=1

# copy dependency list first (docker cache optimization)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

EXPOSE 8000

CMD ["uvicorn","src.app:app","--host","0.0.0.0","--port","8000"]
