FROM python:3.11-slim

WORKDIR /frontend

COPY requirements.txt /frontend/requirements.txt
RUN pip install --no-cache-dir -r /frontend/requirements.txt

COPY website/streamlit_app.py /frontend/streamlit_app.py

EXPOSE 8501
ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]