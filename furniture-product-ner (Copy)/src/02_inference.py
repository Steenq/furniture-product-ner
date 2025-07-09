from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

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
    print("DEBUG:", result)
    names = {r['word'].strip() for r in result if "PRODUCT" in r.get('entity_group', '')}
    return sorted(names)

if __name__ == "__main__":
    print("Извлечённые товары:", extract_products("MALM Bed Frame - White"))
    print("Извлечённые товары:", extract_products("SÖDERHAMN Sofa - Finnsta turquoise. MALM Bed Frame - White."))
    print("Извлечённые товары:", extract_products("Factory Buys 32cm Euro Top Mattress - King, Description Product Features Dimensions & Specs What’s Included Factory Buys 32cm Euro Top Mattress - King"))
