# from langchain_community.llms import Bedrock
from langchain_aws import BedrockLLM
from langchain_core.prompts import ChatPromptTemplate
import boto3

AWS_REGION = "us-east-1"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

model = BedrockLLM(model_id="amazon.titan-text-express-v1", client=bedrock)

def invoke_model():
    response = model.invoke("What is the highest mountain in the World")
    print(response)

def first_chain():
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Write a short decsription for the product provided by the user"
            ),
            (
                "human",
                "{product_name}"
            )
        ]
    )
    chain = template.pipe(model)
    response = chain.invoke(
        {
            "product_name":"bicycle"
        }
    )

    print(response)


# invoke_model()
first_chain()