  pip install "numpy<2" --force-reinstall
  pip install mlflow --ignore-installed blinker
  pip install transformers peft datasets accelerate pyyaml
  pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu121