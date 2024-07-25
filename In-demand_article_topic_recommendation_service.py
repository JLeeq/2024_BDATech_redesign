from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os


load_dotenv()

#Enter an titles of the article or summary of the article here.
title_text="""

"""

title_document = Document(
    page_content=title_text,
)


recursive_text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n",".",","],
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)

recursive_splitted_document = recursive_text_splitter.split_documents([title_document])


embedding_model=AzureOpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Chroma 벡터 스토어 설정
chroma = Chroma("vector_store")
vector_store = chroma.from_documents(
        documents=recursive_splitted_document,
        embedding=embedding_model
    )

# 검색 유형 설정
similarity_retriever = vector_store.as_retriever(search_type="similarity")
mmr_retriever = vector_store.as_retriever(search_type="mmr")
similarity_score_retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.2}
    )

retriever = similarity_retriever


#---------------------------------------------------------------------

# 시스템 프롬프트 설정

system_prompt_str = """
You are an assistant for identifying and extracting key phrases or keywords from a given context that are new and useful for further searches in an encyclopedia or database. Ensure the keywords are specific and provide sufficient context for comprehensive searches.
Extract 50 or more specific and contextually rich keywords from the given news article that are useful for encyclopedia search. Include sufficient context to make the keywords longer and more descriptive. Exclude names of people, titles, company names, promotional phrases, dates, and times from the results.
{context} """.strip()





prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_str),
        #("human", "Extract key phrases or keywords from the given news article that are useful for encyclopedia searches. Provide contextually rich and specific keywords."),
    ]
)

azure_model = AzureChatOpenAI(
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
)

# 체인 생성
question_answer_chain = create_stuff_documents_chain(azure_model, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# 키워드 추출 반복
all_keywords = set()
while len(all_keywords) < 50:
    chain_output = rag_chain.invoke({"input": title_text})
    # 추출한 키워드를 리스트로 저장하고 넘버링 제거
    keywords_with_numbers = chain_output["answer"].split("\n")
    keywords = [keyword.split(' ', 1)[1].strip().strip('"') for keyword in keywords_with_numbers if keyword.strip()]
    all_keywords.update(keywords)
    if len(keywords_with_numbers) < 50:  # 만약 한번에 50개 이하만 추출됐다면 반복 종료
        break

# 최종 50개의 키워드 추출
final_keywords = list(all_keywords)[:50]

print("Extracted Keywords:", keywords)
