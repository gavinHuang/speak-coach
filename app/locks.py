import threading

# MLX and the torch models are not safe to drive from several threads at once, so every model
# call (practice requests and library ingestion alike) takes this lock.
MODEL_LOCK = threading.Lock()
