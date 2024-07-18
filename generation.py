import ollama


class generation():
    def __init__(self, filtered_chunk_list: list[dict], collection, embedding_generator, llm_model: str, filename: str) -> None:
        self.filename = filename
        self.filtered_chunk_list = filtered_chunk_list
        self.collection = collection
        self.embedding_generator = embedding_generator
        self.llm_model = llm_model
        # number of results from vector db search
        self.no_results = 5

    def chat(self, prompt: str, history: list):
        if prompt == "" or prompt is None:
            return ('please enter a valid prompt')

        prompt = self.reword(prompt, history)

        max_history_length = 20
        history = history[-max_history_length:]

        context = self.search_db(prompt)
        prompt_w_context = self.generate_prompt(prompt, history, context)
        print(prompt_w_context)
        output = ollama.generate(model=self.llm_model,
                                 prompt=prompt_w_context, stream=True)
        message = ''
        for chunk in output:
            message += chunk['response']
            yield (message)

    def search_db(self, prompt: str) -> str:
        response = self.embedding_generator.embed(prompt)
        results = self.collection.query(
            query_embeddings=response,
            n_results=self.no_results
        )
        chunks = results["documents"]
        chunk_ids = results["ids"][0]
        pages = [self.filtered_chunk_list[int(id)]["page number"] for id in chunk_ids]
        context = ""
        for i in range(self.no_results):
            context += "\n- " + chunks[0][i] + f" PAGE NUMBER: {pages[i]}"
        return context

    def generate_prompt(self, prompt: str, history: list, context: str):
        template = f"""You are an AI assistant with access to the pdf {self.filename}. Use the provided context and conversation history to answer the user's query accurately.

                    Context from {self.filename}:
                    {context}

                    Conversation History:
                    {history}

                    User Query:
                    {prompt}

                    Instructions:
                    1. Respond to the query based on the context from {self.filename} and the conversation history if they are relevant.
                    2. If you refer to the context in your response, mention the book name {self.filename} and the respective page number.
                    3. Ensure the answer is accurate, concise, directly addresses the user's query and is as explanatory as possible.

                    Answer:
                    """

        return template

    def reword(self, prompt, history):
        max_history_length = 2
        history = history[-max_history_length:]
        rewording_prompt = f"""Instruction: You are a Prompt Re-worder. Your task is to reword user queries to include necessary historical context from previous interactions, **only if the history is related to the current query**. Ensure the reworded query is clear and coherent, incorporating relevant keywords and phrases from the related conversation history. If the current query does not require historical context, no relevant context is available, or the history is not related, return the query as it is. You return the reworded prompt and the reworded prompt only.

Examples:

Previous Interactions:

User: "Can you tell me more about the benefits of a plant-based diet?"
Assistant: "A plant-based diet can improve heart health, aid in weight management, and reduce the risk of chronic diseases."
Current Query: "What about its impact on the environment?"
Reworded Query: "Can you tell me about the impact of a plant-based diet on the environment?"

Previous Interactions:

User: "I need some advice on improving my productivity."
Assistant: "Try techniques like the Pomodoro method, setting clear goals, and minimizing distractions."
Current Query: "Which tools can help with that?"
Reworded Query: "Which tools can help with improving productivity using techniques like the Pomodoro method and goal setting?"

Previous Interactions:

User: "What's the weather like in New York today?"
Assistant: "It's sunny with a high of 75°F."
Current Query: "How about tomorrow?"
Reworded Query: "What's the weather forecast for New York tomorrow?"

Previous Interactions:

User: "I'm planning a trip to Paris. What are some must-see attractions?"
Assistant: "You should visit the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
Current Query: "What about some good local restaurants?"
Reworded Query: "Can you recommend some good local restaurants in Paris?"

No Relevant Historical Context:

User: "Tell me a joke."
Current Query: "Tell me another one."
Reworded Query: "Tell me another joke."

Previous Interactions:

User: "What is the capital of France?"
Assistant: "The capital of France is Paris."
Current Query: "What is the population?"
Reworded Query: "What is the population of Paris?"

No Relevant Historical Context:

User: "What's your favorite color?"
Current Query: "What's your favorite movie?"
Reworded Query: "What's your favorite movie?"

Implementation Steps:

1. Context Extraction: When a new query is received, analyze the conversation history to identify relevant context. **Only extract key words and phrases from history that are directly related to the user's current query.**

2. Rewording Mechanism: Use the designed prompt to guide the rewording of the current query. Ensure the reworded prompt incorporates appropriate details and keywords from the related history, enhancing its relevance for searching a vector database. DO NOT include any thinking steps in your responses and return ONLY the reworded prompt.

3. Fallback: If no relevant context is identified, if the history is not related, or if the query does not require additional context, return the original query. If there is no previous interaction history, do not give any information. Just return the current query as is with no additional information.

EXTREMELY IMPORTANT: RETURN THE REWORDED PROMPT AND THE REWORDED PROMPT ONLY. DO NOT GIVE ANY CONTEXT OR DETAILS. YOU ARE A PROMPT RE-WORDER ONLY.

Conversation History: {history}

Current Query: {prompt}

Reworded Query:
                            """

        output = ollama.generate(model=self.llm_model, prompt=rewording_prompt)
        prompt = output["response"]

        return prompt
