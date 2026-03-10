FROM python:3.12-slim

WORKDIR /app

# Installer PyTorch CPU en premier (évite le téléchargement GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY api/ api/
COPY rag/ rag/
COPY vectorstores/ vectorstores/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
