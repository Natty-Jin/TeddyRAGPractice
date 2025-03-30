from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub

loader = PDFPlumberLoader()
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents=split_documents, embedding = embeddings)

retriever = vectorstore.as_retriever()

prompt = hub.pull("teddynote/rag-korean")

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

chain = (
    {"context":retriever, "question":RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)
chain.invoke("한국에서 개발한 생성형 AI중 가장 벤치마크 순위가 높은 것은?")
