#linux  :  CMAKE_ARGS="-DLLAMA_AVX2=ON -DLLAMA_FMA=ON" pip install llama-cpp-python --force-reinstall --no-cache-dir
#windows : $env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"; pip install llama-cpp-python --force-reinstall --no-cache-dir


#llama-3.2-1b-instruct-q8_0.gguf

from llama_cpp import Llama


llm = Llama(model_path="models/llama-2-7b-chat.Q4_K_M.gguf", chat_format="llama-3")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the distance from Earth to Mars?"}
]

response = llm.create_chat_completion(messages)
print(response['choices'][0]['message']['content'])



