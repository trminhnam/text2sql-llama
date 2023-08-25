# text2sql-llama

## Create environment to train the model

Create an environment with conda:

```bash
conda create -n llama python=3.9
conda activate llama
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Install llama-cpp-python with cuLAB (from this [link](https://python.langchain.com/docs/integrations/llms/llamacpp)):

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

## References

-   https://medium.com/llamaindex-blog/easily-finetune-llama-2-for-your-text-to-sql-applications-ecd53640e10d

-   https://github.com/run-llama/modal_finetune_sql/tree/main
