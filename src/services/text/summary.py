import boto3
import json

AWS_REGION = "us-east-1"

bedrock = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)

def handler(event, context):
    body = json.loads(event["body"])
    text = body.get("text")
    points = body.get("points")
    if text and points:
        titan_config = get_titan_config(text, points)
        response = bedrock.invoke_model(
            body = titan_config,
            modelId="amazon.titan-text-express-v1",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        result = response_body.get("results")[0]
        return{
            "statusCode": 200,
            "body": json.dumps({"summary": result.get("outputText")})
        }
    return{
        "statusCode": 400,
        "body": json.dumps({"error": "Text and Points are Required!!"})
    }


def get_titan_config(text: str, points: str):
    prompt = f"""Text: {text} \n
                From the text above, summarize the story in {points} bullet points.\n    
            """

    return json.dumps(
                {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 4096,
                        "stopSequences": [],
                        "temperature": 0,
                        "topP": 1,
                    },
                }
            )