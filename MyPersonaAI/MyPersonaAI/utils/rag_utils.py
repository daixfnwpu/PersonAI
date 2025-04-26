from elasticsearch import Elasticsearch
import os

def create_index(es_host="localhost", es_port=9200, index_name="knowledge_base"):
    es = Elasticsearch([{'host': es_host, 'port': es_port}])

    # 定义索引的结构
    body = {
        "mappings": {
            "properties": {
                "content": {
                    "type": "text"
                }
            }
        }
    }

    # 创建索引
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=body)
        print(f"索引 '{index_name}' 已创建。")
    else:
        print(f"索引 '{index_name}' 已经存在。")

def index_documents(knowledge_base_path, es_host="localhost", es_port=9200, index_name="knowledge_base"):
    es = Elasticsearch([{'host': es_host, 'port': es_port}])
    
    # 读取知识库文件
    knowledge_files = [f for f in os.listdir(knowledge_base_path) if f.endswith('.txt')]
    
    for file_name in knowledge_files:
        with open(os.path.join(knowledge_base_path, file_name), "r") as file:
            content = file.read()
            doc = {
                "content": content
            }
            es.index(index=index_name, body=doc)
            print(f"文档 {file_name} 已成功添加到索引中。")

# 创建索引并索引文档
create_index()
index_documents('./knowledge_base')
