# RAG 科研问答助手（代码实现版）

基于 LangChain + FAISS + Streamlit 的 RAG 系统，支持私有知识库问答和 arXiv 文献检索分析。

## 功能

- 📄 上传 PDF/TXT/Markdown 文档，自动构建向量知识库
- 💬 基于知识库的智能问答（RAG）
- 🔍 根据关键词检索 arXiv 最新论文
- 🤖 大模型分析论文相关度（可选）

## 快速开始

### 1. 环境要求

- Python 3.10+
- 大模型 API Key（智谱AI / OpenAI / 百度千帆等）

### 2. 克隆仓库

```bash
git clone https://github.com/wzt552200/RAG-QA-System-Based-on-Dify.git
cd RAG-QA-System-Based-on-Dify
3. 安装依赖
bash
pip install -r requirements.txt
4. 配置 API Key
创建 .env 文件（以智谱AI为例）：

env
ZHIPU_API_KEY=your_api_key_here
5. 运行
bash
streamlit run app.py
浏览器打开 http://localhost:8501。

项目结构
text
RAG-QA-System-Code/
├── requirements.txt
├── .env                      # API Key
├── app.py                    # Streamlit 主界面
├── knowledge_base/           # 上传的文档
├── faiss_index/              # 向量索引
└── utils/
    ├── document_loader.py
    ├── vector_store.py
    ├── arxiv_search.py
    └── rag_chain.py
核心代码示例
文档加载与分块
python
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(directory):
    docs = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(directory, file), encoding="utf-8")
            docs.extend(loader.load())
        # 支持 PDF、Markdown 等
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)
向量存储（FAISS）
python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)
vector_store.save_local("./faiss_index")
arXiv 检索
python
import arxiv

def search_arxiv(query, max_results=5):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = []
    for paper in client.results(search):
        papers.append({
            "title": paper.title,
            "summary": paper.summary[:500],
            "link": paper.entry_id,
            "published": paper.published.date().isoformat()
        })
    return papers
RAG 问答链
python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatZhipuAI

llm = ChatZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"), model="glm-4", temperature=0.3)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
answer = qa_chain.run(question)
效果演示
知识库问答
https://screenshots/qa_demo.png

文献检索
https://screenshots/arxiv_demo.png

依赖列表
requirements.txt 内容：

text
streamlit==1.28.1
langchain==0.1.0
langchain-community==0.0.10
faiss-cpu==1.7.4
sentence-transformers==2.2.2
arxiv==2.0.0
python-dotenv==1.0.0
PyPDF2==3.0.1
tiktoken==0.5.1
openai==1.3.0
zhipuai==1.0.0
常见问题
知识库检索不到内容：检查文档是否完成向量化，降低相似度阈值，增加召回数量。

arXiv 检索失败：国内可能需要代理，或使用镜像站。

API Key 无效：确认模型供应商账户有余额，且 Key 填写正确。

未来计划
集成 Semantic Scholar

文献定时推送（飞书/钉钉）

论文摘要自动生成

