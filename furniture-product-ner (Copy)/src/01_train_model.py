from datasets import load_from_disk
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score

MODEL = "xlm-roberta-base"
LABELS = ["O", "B-PRODUCT", "I-PRODUCT"]
NUM_LABELS = len(LABELS)

# 1. Загружаем токенизатор и датасет
ds = load_from_disk("data/processed")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# 2. Загружаем модель для NER, настраиваем метки
model = AutoModelForTokenClassification.from_pretrained(
    MODEL,
    num_labels=NUM_LABELS,
    id2label=dict(enumerate(LABELS)),
    label2id={l: i for i, l in enumerate(LABELS)},
)

# 3. Параметры обучения
args = TrainingArguments(
    output_dir="models/product_ner",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    weight_decay=0.01,
    logging_steps=20,
)

# 4. Trainer — основной класс Hugging Face для обучения
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
)

# 5. Запускаем обучение
trainer.train()

# 6. Сохраняем модель (чтобы использовать потом в пайплайне)
trainer.save_model("models/product_ner")

# 7. Печатаем метрики
metrics = trainer.evaluate()
print("Metrics:", metrics)
