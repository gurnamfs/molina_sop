FROM python:3.11
WORKDIR .
COPY ./requirements.txt .
RUN pip install -r ./requirements.txt
COPY . .
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--timeout", "300"]
