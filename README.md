# Ollama Embedding Pipeline

Description: A simple pipeline for one to use when converting certain models (particularly embedding models) to gguf for support on Ollama.


### Setup

 - Virtualenv
     - Use conda (or similar alternatives) to set up the virtual environment.
     - Run `conda create -f env.yml` to setup everything.
 - Models
     - All models (with their respective configuration information) can be found in `config.json`. Just follow the general format.
     - Note: the hkunlp instructor models will be passed over/skipped in the general script. This is because the model is an encoder-decoder model (similar to T5) and requires additional setup to generate the embeddings while the other models are just encoder-only and are pretty straightforward to work with.
 - Llama.cpp
     - Clone the [llama.cpp repo](https://github.com/ggml-org/llama.cpp) in the root of this repository.
     - Build the executables (requires `cmake` or `make` to be installed but instructions in repo specify `cmake`).
         - I use the commands [listed here](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) in the repo's build docs.
         - `cmake -B build; cmake --build build --config Release`
 - Ollama
     - Follow the [install instructions](https://ollama.com/download) on their website.
     - Make sure ollama is running on default ports.
 - Docker
     - Recommended way to run everything.
     - Just run the command `docker compose up` and it'll build everything you need.
     - Building the docker image with `docker compose up` takes around 25 minutes and around 40 GB total.


### Notes

 - BERT pooling methods were too old for ollama to register properly.
     - This required that I specify the pooling method for the base BERT models in the `config.json` while all other pooling methods are default to `mean`.


### References

 - Ollama
     - [Importing models](https://docs.ollama.com/import)
     - [Ollama docker image](https://hub.docker.com/r/ollama/ollama)
     - [Embedding models on Ollama](https://ollama.com/search?c=embedding)