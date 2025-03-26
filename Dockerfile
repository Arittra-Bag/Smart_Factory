FROM python:3.11-slim

COPY requirements.txt .
COPY smart_factory_control.py .
COPY smart_factory_gradio.py .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 10000

CMD ["python", "smart_factory_gradio.py"]