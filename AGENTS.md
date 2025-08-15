environment: 
pytorch(gpu or cpu), torchvision, cudatoolkit(if gpu)
pip install numpy ipython psutil traitlets transformers termcolor ipdb scipy h5py plyfile tabulate

if necessary:
pip install spacy
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-xx/en_core_web_sm-xx.tar(according to spacy version)

if necessary:
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
mim install "mmdet3d>=1.1.0"

if necessary:
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
or cpu only:
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
#                           \ # uncomment the following line if you want to force cuda installation
#                           --install-option="--force_cuda" \
#                           \ # uncomment the following line if you want to force no cuda installation. force_cuda supercedes cpu_only
#                           --install-option="--cpu_only" \
#                           \ # uncomment the following line to override to openblas, atlas, mkl, blas
#                           --install-option="--blas=openblas" \


