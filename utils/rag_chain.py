from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatZhipuAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_rag_chain(vector_store):
    llm = ChatZhipuAI(
        api_key=os.getenv("ZHIPU_API_KEY"),
        model="glm-4",
        temperature=0.3
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    prompt_template = """你是一个科研助手。请根据以下上下文回答用户问题。如果无法从上下文中得到答案，请说“知识库中没有相关信息”。

上下文：
{context}

问题：{question}
回答："""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain