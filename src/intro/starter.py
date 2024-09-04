import pprint

import boto3

bedrock = boto3.client(
    service_name = 'bedrock',
    region_name = 'us-east-1')

models = bedrock.list_foundation_models()

pp = pprint.PrettyPrinter(depth=4)

for model in models["modelSummaries"]:
    pp.pprint(model)


# print(models)