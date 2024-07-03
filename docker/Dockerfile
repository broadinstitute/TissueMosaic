# The image corresponding to this dockerfile can be obtained by 
#docker pull us.gcr.io/broad-dsde-methods/tissuemosaic:0.0.4

FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.10

RUN yes | pip3 install -U psutil 
RUN yes | pip3 install -U matplotlib  
RUN yes | pip3 install -U scikit-image 
RUN yes | pip3 install -U neptune-client 
RUN yes | pip3 install -U python-igraph  
RUN yes | pip3 install -U leidenalg 
RUN yes | pip3 install -U Pillow  
RUN yes | pip3 install -U ipython 
RUN yes | pip3 install -U pyyaml
RUN yes | pip3 install -U scikit-learn
RUN yes | pip3 install -U umap-learn
RUN yes | pip3 install -U torchmetrics
RUN yes | pip3 install -U pandas
RUN yes | pip3 install -U google-cloud-storage
RUN yes | pip3 install -U scanpy
RUN yes | pip3 install -U lightning-bolts
RUN yes | pip3 install -U colorcet        
RUN yes | pip3 install -U lightly

RUN git clone https://github.com/pyro-ppl/pyro.git /pyro 
RUN yes | pip3 install /pyro

# RUN git clone https://github.com/broadinstitute/TissueMosaic.git /tp 
# RUN yes | pip3 install /tp

RUN mkdir /app
