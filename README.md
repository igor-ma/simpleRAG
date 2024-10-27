# simpleRAG

The goal of this repository is to provide a simple example of a solution using RAG (Retrieval-augmented generation) to obtain context information from a given web page, and use this context to answer a question. The application, which is called simpleRAG, is particularly useful when you want to ask questions about the content of a page and get direct answers, without the need to manually extract the text from the page and input it into some text generation AI.

Given a web page, the solution automatically extracts the text data, divides the content into small chunks of text, vectorizes each of these chunks (generates embeddings), and stores them in a FAISS vector database. Afterwards, the provided question is vectorized and used as a basis to retrieve the most relevant chunks to obtain an answer. The chunks are placed as context for the question, and the model generates an answer.

The solution uses LangChain and is ready to work with Google's [Gemini models](https://ai.google.dev/gemini-api/docs/models/gemini) and OpenAI's [GPT models](https://platform.openai.com/docs/models). You should chose a `modelFamily`, and you can also specify an `LLMModel` and an `EmbeddingModel`, or use the default ones (`gpt-3.5-turbo` and `text-embedding-ada-002` for GPT family, and `gemini-1.5-flash` and `models/text-embedding-004` for Gemini family).

First, build the docker image:
```
docker build -t simple-rag .
```

To understand the arguments needed, run:
```
docker run simple-rag --help
```

Then, run the solution. The following code is an example of how to do it using Gemini's default models. Note that you need to provide a Google [Gemini API key](https://ai.google.dev/gemini-api/) (or a [OpenAI one](https://openai.com/index/openai-api/)).
```
docker run -it --network host simple-rag --model Gemini --apiKey yourGeminiAPIkey --contextWebPage "https://en.wikipedia.org/wiki/I%27m_Still_Here_(2024_film)" --question "Did I'm Still Here get any Oscars?"
```
