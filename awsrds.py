import pymysql
import streamlit as st

mydb=pymysql.connect(
    user='admin',
    host='database-1.czs8muuo08zk.us-west-1.rds.amazonaws.com',
    port=3306,
    password='arun53787'

)
st.write('connected')