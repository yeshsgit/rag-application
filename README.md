# rag-application
do rag with any pdf

### Getting Started
1. Create a new file called config.py
2. Populate this file based on example_config.py
   Here is an example of what I used:
   ```
    file_path = "human_nutrition.pdf"
    file_name = "human_nutrition.pdf"
    file_url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
    chunk_size = 10
    llm_model = 'llama3'
    embedding_model = "avsolatorio/GIST-small-Embedding-v0"
    filtered_chunks_file_path = "filtered_chunks.json"
   ```
3. Install ollama from https://ollama.com/download
4. Install an llm model like llama3 from ollama, specify this as the llm_model in config.py\
    command: ollama pull <model_name>
5. Ollama needs to be running for the model to work. If you try to run main.py after restarting your computer you will get an error.\
   Start ollama with the command below.\
    command: ollama serve
6. Install the requirements in requirements.txt
7. Run main.py
8. When processing is complete, to open a chat in browser, click on the local url displayed in terminal
