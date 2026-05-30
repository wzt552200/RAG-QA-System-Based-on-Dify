import streamlit as st
import os
from dotenv import load_dotenv
from utils.document_loader import load_documents
from utils.vector_store import build_vector_store, load_vector_store
from utils.arxiv_search import search_arxiv
from utils.rag_chain import get_rag_chain

load_dotenv()

st.set_page_config(page_title="RAG 科研问答助手", layout="wide")
st.title("📚 RAG 智能问答助手")

# 侧边栏
with st.sidebar:
    st.header("知识库管理")
    uploaded_files = st.file_uploader("上传文档 (PDF/TXT/MD)", accept_multiple_files=True)
    if uploaded_files:
        os.makedirs("knowledge_base", exist_ok=True)
        for file in uploaded_files:
            with open(f"knowledge_base/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        st.success("文档已上传")

    if st.button("重建知识库"):
        with st.spinner("正在处理文档..."):
            chunks = load_documents("knowledge_base")
            vector_store = build_vector_store(chunks)
            st.session_state["vector_store"] = vector_store
        st.success("知识库构建完成！")

    if st.button("加载已有知识库"):
        if os.path.exists("faiss_index"):
            st.session_state["vector_store"] = load_vector_store()
            st.success("知识库加载成功")
        else:
            st.warning("未找到已有知识库，请先上传文档并重建。")

# 主区域
tab1, tab2 = st.tabs(["💬 知识库问答", "🔍 文献检索与分析"])

with tab1:
    question = st.text_input("请输入问题：")
    if st.button("提问", key="qa"):
        if "vector_store" not in st.session_state:
            st.warning("请先在侧边栏构建或加载知识库")
        else:
            qa_chain = get_rag_chain(st.session_state["vector_store"])
            answer = qa_chain.run(question)
            st.markdown("### 回答")
            st.write(answer)

with tab2:
    keyword = st.text_input("输入关键词检索 arXiv 论文：")
    if st.button("检索", key="arxiv"):
        if keyword:
            papers = search_arxiv(keyword)
            st.markdown(f"### 找到 {len(papers)} 篇相关论文")
            for i, paper in enumerate(papers):
                with st.expander(f"{i+1}. {paper['title']}"):
                    st.write(f"**发表时间**: {paper['published']}")
                    st.write(f"**摘要**: {paper['summary']}...")
                    st.write(f"**链接**: {paper['link']}")
        else:
            st.warning("请输入关键词")