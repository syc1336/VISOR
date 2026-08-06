## deploy on GPU 0
python search_engine/search_engine_api.py
## deploy on GPU 1
vllm serve autumncc/Qwen2.5-VL-7B-VRAG --port 8001 --host 0.0.0.0 --limit-mm-per-prompt image=10 --served-model-name Qwen/Qwen2.5-VL-7B-Instruct
## start demo
streamlit run demo/app.py
