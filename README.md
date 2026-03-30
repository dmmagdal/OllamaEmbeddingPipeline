# Ollama Embedding Pipeline

Description: A simple pipeline for one to use when converting certain models (particularly embedding models) to gguf for support on Ollama.


### Setup

 - Virtualenv
     - Use conda (or similar alternatives) to set up the virtual environment.
     - Run `conda create -f env.yml` to setup everything.
 - Models
     - All models (with their respective configuration information) can be found in `config.json`. Just follow the general format.
     - Note: the hkunlp instructor models will be passed over/skipped in the general script. This is because the model is an encoder-decoder model (similar to T5) and requires additional setup to generate the embeddings while the other models are just encoder-only and are pretty straightforward to work with.


### References

 - Ollama
     - [Importing models](https://docs.ollama.com/import)
     - [Ollama docker image](https://hub.docker.com/r/ollama/ollama)
     - [Embedding models on Ollama](https://ollama.com/search?c=embedding)