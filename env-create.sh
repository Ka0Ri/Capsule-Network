conda create -n Capsule-demo python=3.9
conda activate Capsule-demo
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
python -m pip install lightning
python -m pip install gradio
python -m pip install opencv-python
python -m pip install scikit-learn
python -m pip install scipy