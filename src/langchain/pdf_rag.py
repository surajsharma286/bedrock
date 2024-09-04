from langchain_aws import BedrockLLM
from langchain_aws import BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import boto3

AWS_REGION = "us-east-1"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = BedrockLLM(model_id="amazon.titan-text-express-v1", client=bedrock)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", client=bedrock
)

question = "What themes does Gone with the Wind explore?"

# data ingestion
loader = PyPDFLoader("C:\\Udemy\\workspace\\bedrock\\src\\langchain\\assets/books.pdf")
splitter = RecursiveCharacterTextSplitter(separators=[". \n"])
docs = loader.load()
splitted_docs = splitter.split_documents(docs)

# create vector store
vector_store = FAISS.from_documents(splitted_docs, bedrock_embeddings)

# create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
results = retriever.invoke(question)

results_string = []
for result in results:
    results_string.append(result.page_content)

# build template:
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the users question based on the following context: {context}",
        ),
        ("user", "{input}"),
    ]
)

chain = template.pipe(model)

response = chain.invoke({"input": question, "context": results_string})
print(response)
