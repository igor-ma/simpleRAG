import os
import argparse
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS #vector database
from langchain_text_splitters import RecursiveCharacterTextSplitter #split text into chunks


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["GPT", "Gemini"],
                    help="Choose between OpenAI and Google")
parser.add_argument("--apiKey", required=True, type=str,
                    help="Inform your API Key for the chosen model")
parser.add_argument("--contextWebPage", required=True, type=str,
                    help="Web page with context content to be used")
parser.add_argument("--question", required=True, type=str,
                    help="Question about given context")
args = parser.parse_args()


def main():

    #set env vars and create llm as specified
    if args.model == 'GPT':
        os.environ['OPENAI_API_KEY'] = args.apiKey
        llm = OpenAI(temperature=0)
        embedder = OpenAIEmbeddings()
    elif args.model == 'Gemini':
        os.environ['GOOGLE_API_KEY'] = args.apiKey
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        embedder = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    #load texts to be used as a context database
    loader = WebBaseLoader(args.contextWebPage)
    rawDocs = loader.load()

    #split context texts into documents
    textSplitter = RecursiveCharacterTextSplitter()
    splittedDocs = textSplitter.split_documents(rawDocs)

    #create an embedding for each document obtained
    vectorDatabase = FAISS.from_documents(splittedDocs, embedder)

    #create a retriever object to get vectorized matches for a given input
    retriever = vectorDatabase.as_retriever()
    
    #create a template
    contextQuestionPrompt = ChatPromptTemplate.from_template("""
    {context}
    Question: {input}
    """)

    #chain LLM and prompt
    contextQuestionChain = create_stuff_documents_chain(llm, contextQuestionPrompt)

    #create a retriever chain
    retrieverChain = create_retrieval_chain(retriever, contextQuestionChain)

    response = retrieverChain.invoke({'input': args.question})
    print(response['answer'])

main()