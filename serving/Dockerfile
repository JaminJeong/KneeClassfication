FROM ultralytics/ultralytics:latest-cpu

WORKDIR /workspace

# Install serving dependencies
COPY serving/requirements.txt /tmp/serving_requirements.txt
RUN pip install --no-cache-dir -r /tmp/serving_requirements.txt

# Copy project files
COPY serving/ /workspace/serving/
COPY runs/ /workspace/runs/

# Environment defaults (override via docker run -e or compose)
ENV KNEE_MODEL_PATH=/workspace/runs/classify/knee_cls_py_test/weights/best.pt
ENV KNEE_IMGSZ=224
ENV KNEE_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
