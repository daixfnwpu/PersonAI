# This script uses the trained model to generate answers
from rag_retriever import RagRetriever
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def infer_with_rag(query):
    # 加载 RAG 检索器
    retriever = RagRetriever(knowledge_base_path='./knowledge_base')

    # 生成答案
    answer = retriever.generate_answer(query)
    
    print(f"问题: {query}")
    print(f"回答: {answer}")

if __name__ == "__main__":
    query = input("请输入您的问题: ")
    infer_with_rag(query)
