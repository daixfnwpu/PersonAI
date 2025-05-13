# This script implements the Elasticsearch retriever for RAG
from elasticsearch import Elasticsearch
from torch import float16
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from elasticsearch import Elasticsearch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig

model_id = "facebook/opt-125m"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0,
    bnb_4bit_compute_dtype=float16,  # ✅ 添加这一行
)

class ElasticsearchRetriever:
    def __init__(self, es_host="localhost", es_port=9200, index_name="knowledge_base", top_k=3):
        self.es = Elasticsearch([{'host': es_host, 'port': es_port}])
        self.index_name = index_name
        self.top_k = top_k
        
        # ✅ 使用 facebook/bart-large
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        # self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        
    def search_documents(self, query):
        body = {
            "query": {
                "match": {
                    "content": query
                }
            }
        }
        result = self.es.search(index=self.index_name, body=body, size=self.top_k)
        documents = [hit["_source"]["content"] for hit in result["hits"]["hits"]]
        return documents

    def format_input(self, query, retrieved_docs):
        context = "\n".join(retrieved_docs)
        return f"Question: {query}\nContext: {context}"

    def generate_answer(self, query):
        retrieved_docs = self.search_documents(query)
        input_text = self.format_input(query, retrieved_docs)
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model.generate(inputs["input_ids"], max_length=200)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

