from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import requests
from bs4 import BeautifulSoup

MODEL_PATH = "models/product_ner"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

ner = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

def extract_products(text):
    result = ner(text)
    names = {r['word'].strip() for r in result if "PRODUCT" in r.get('entity_group', '')}
    return sorted(names)

def fetch_visible_text(url):
    try:
        html = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'}).text
        soup = BeautifulSoup(html, "lxml")
        text = " ".join(tag.get_text(" ", strip=True) for tag in soup.find_all(['h1', 'h2', 'h3', 'p']))
        return text
    except Exception as e:
        return f"[Error when turning on: {e}]"

app = FastAPI()
templates = Jinja2Templates(directory="src/app/templates")

@app.get("/", response_class=HTMLResponse)
def main_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "products": None, "url": ""})

@app.post("/", response_class=HTMLResponse)
async def extract(request: Request, url: str = Form(...)):
    text = fetch_visible_text(url)
    products = extract_products(text)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "products": products,
        "url": url,
        "text": text[:2000]  
    })
