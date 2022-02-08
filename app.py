from msilib.schema import Binary
from turtle import pd
from flask import Flask, render_template, redirect, url_for, request
import os
#from flask_sqlalchemy import SQLAlchemy
import psycopg2
from config import postgres_pass
import analyze

app = Flask(__name__)
    
def get_db_connection():
    conn = psycopg2.connect(host="localhost", port = 5432, database="final_project", user="postgres", password=postgres_pass)
    return conn

@app.route("/", methods =["GET", "POST"])
def index():
    
    if request.method == "POST":
        ticker = request.form.get("inputticker")
        #return ticker
    
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT * from predicted_price;")
    predicted_price = cur.fetchall()
    predicted_price = list(predicted_price)[0][0]

  
    cur.execute("SELECT * from ticker;")
    ticker = cur.fetchall()
    business_summary = ticker[0][4]
    current_price = ticker[0][150]
    tik = ticker[0][52]
    rec = analyze.recommendation(current_price, predicted_price)
    cur.close()
    conn.close()

    return render_template('index.html', cp=current_price, pp=predicted_price, bs=business_summary, tk=tik, rec = rec)

@app.route("/analyze", methods =["GET", "POST"])
def analyzer():

    if request.method == "POST":
        ticker = request.form.get("inputticker")
        #return ticker
    
    analyze.run_all(ticker)
    #rec = analyze.recommendation(current_price, predicted_price)
    return redirect('/', code=302)

#cur.close()
#conn.close()
    
if __name__ == "__main__":
    app.run(debug= True)
