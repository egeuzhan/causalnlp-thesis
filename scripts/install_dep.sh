# install necessary packages with pip
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install bi-lstm-crf
pip install transformers
pip install opt-einsum==3.3.0
pip install tqdm
pip install ujson
pip install wandb
pip install spacy
python -m spacy download en_core_web_sm
pip install allennlp