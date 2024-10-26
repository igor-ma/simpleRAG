# RAG

Usage example:
```
docker build -t web-rag .

docker run -it --network host web-rag --model Gemini --apiKey yourAPIkey --contextWebPage "https://en.wikipedia.org/wiki/Oppenheimer_(film)" --question "Did Oppenheimer get any Oscars?
```