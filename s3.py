import pandas as pd
# import streamlit as st
import boto3
from dotenv import load_dotenv
from io import StringIO,BytesIO
import os

load_dotenv()
access_key=os.getenv("ACCESS_KEY")
secret_key=os.getenv("SECRET_KEY")
print(access_key)
print(secret_key)
bucket=os.getenv("BUCKET")
file=os.getenv("KIDNEY")
print(bucket)
print(file)
acc=os.getenv("BUCKET_NAME")
print(acc)

# AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
# AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
# BUCKET_NAME = os.getenv("BUCKET_NAME")
# FILE_KEY = os.getenv("FILE_KEY")

s3=boto3.client('s3',aws_access_key_id=access_key,
                aws_secret_access_key=secret_key)
response=s3.get_object(Bucket=bucket,Key=file)
con=response['Body'].read().decode('utf-8')
df=pd.read_csv(StringIO(con))
print(df.head())

