#linux  :  CMAKE_ARGS="-DLLAMA_AVX2=ON -DLLAMA_FMA=ON" pip install llama-cpp-python --force-reinstall --no-cache-dir
#windows : $env:CMAKE_ARGS="-DLLAMA_CUBLAS=on"; pip install llama-cpp-python --force-reinstall --no-cache-dir


#llama-3.2-1b-instruct-q8_0.gguf

from llama_cpp import Llama


llm = Llama(model_path="models/llama-3.2-1b-instruct-q8_0.gguf",
            chat_format="llama-3",n_ctx=4092,
            #n_threads=8,      # Adjust based on your CPU cores
            n_gpu_layers= 32,   # Explicitly disable GPU
            verbose=True)

# 初始系统提示
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("🤖 Chat with LLaMA. Type 'exit' to quit.\n")

while True:
    user_input = input("👤 You: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("👋 Bye!")
        break

    # 添加用户消息
    messages.append({"role": "user", "content": user_input})

    # 获取回复
    response = llm.create_chat_completion(messages)

    # 提取模型回复
    reply = response['choices'][0]['message']['content']
    print(f"🤖 LLaMA: {reply}\n")

    # 添加模型回复到上下文
    messages.append({"role": "assistant", "content": reply})


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the distance from Earth to Mars?"}
]

response = llm.create_chat_completion(messages)
print(response['choices'][0]['message']['content'])



