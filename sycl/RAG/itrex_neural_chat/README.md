# Setup on Intel CPU

```
conda create -n itrex_neural_chat python=3.10
conda activate itrex_neural_chat
git clone https://github.com/intel/intel-extension-for-transformers.git itrex
cd itrex
pip install -r requirements.txt
pip install -v .   # or rm -rf build &&  python setup.py install
cd ./intel_extension_for_transformers/neural_chat/
pip install -r requirements_cpu.txt
cd pipeline/plugins/retrieval
pip install -r requirements.txt
```

# Run
```
python chatbot.py
```

# Notes
You can change "embedding model", or "model_name_or_path", or txt_files to test.
