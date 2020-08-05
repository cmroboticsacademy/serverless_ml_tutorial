import json
import boto3
import uuid
import pickle
import pandas as pd

from sklearn import preprocessing


def lambda_handler(event, context):
    pkl = load_pickle('cmra-serverless-ml-tutorial',
                      'factory_linear_regression.pkl')

    data = parse_event(event)

    df = pd.DataFrame([data])
    X = df[['temp', 'vibration', 'current', 'noise']]
    X = normalize_features(X)
    pred = pkl['model'].predict(X)

    enc_pred = pkl['encoding'][pred[0]]

    return {
        "statusCode": 200,
        "body": json.dumps({
            "prediction": enc_pred
        }),
    }


def load_pickle(s3_bucket, key):
    s3_client = boto3.client('s3')

    download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
    s3_client.download_file(s3_bucket, key, download_path)

    f = open(download_path, 'rb')
    pkl = pickle.load(f)
    f.close()

    return pkl


def normalize_features(X):
    transformer = preprocessing.Normalizer().fit(X)
    return transformer.transform(X).tolist()


def parse_event(event):
    if 'body' in event.keys():
        return json.loads(event['body'])['data']
    else:
        return event['data']
