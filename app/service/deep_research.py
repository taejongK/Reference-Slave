from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain_community.document_loaders import ArxivLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import system_prompt

load_dotenv()


class SearchResult(BaseModel):
    title: str
    published: str
    summary: str
    authors: List[str]
    page_content: str


def parse_arxiv_result(doc) -> SearchResult:
    """Arxiv 문서를 SearchResult 객체로 변환하는 함수"""
    return SearchResult(
        title=doc.metadata.get('Title', ''),
        published=doc.metadata.get('Published', ''),
        summary=doc.metadata.get('Summary', ''),
        authors=doc.metadata.get('Authors', []).split(','),
        page_content=doc.page_content
    )



def search_arxiv(query: str, max_docs: int = 10) -> List[SearchResult]:
    """Arxiv 검색 결과를 SearchResult 객체 리스트로 반환하는 함수"""
    loader = ArxivLoader(
        query=query,
        load_max_docs=max_docs,
        load_all_available=True
    )
    docs = loader.lazy_load()
    search_results = []
    for doc in docs:
        try:
            search_results.append(parse_arxiv_result(doc))
        except Exception as e:
            print(f"오류: parse_arxiv_result에서 JSON 응답을 처리하는 중 오류 발생: {e}")
            print(f"원시 응답: {doc}")
    return search_results


# SerpQuery


class SerpQuery(BaseModel):
    query: str
    research_goal: str


class SerpQueryResponse(BaseModel):
    queries: List[SerpQuery] = Field(
        ..., description="A list of SERP queries to be used for research"
    )


def generate_serp_queries(
    query: str,
    model_name: str,
    num_queries: int = 3,
    learnings: Optional[List[str]] = None,
) -> List[SerpQuery]:
    """
    사용자의 쿼리와 이전 연구 결과를 바탕으로 SERP 검색 쿼리를 생성합니다.
    json_parser를 사용해 구조화된 JSON을 반환
    """
    prompt = """
    ### System:
    {system_prompt}
    
    ### Instructions:
    다음 사용자의 입력을 기반으로 연구 주제를 조사하기 위한 SERP 검색 쿼리를 생성하세요.
    JSON 객체를 반환하며, 'queries' 배열 필드에 {num_queries}개의 쿼리를 포함합니다.(쿼리가 명확할 경우 더 적을 수도 있음)
    각 쿼리 객체에는 'query'와 'research_goal' 필드가 포함되어야 하며, 각 쿼리는 고유해야 합니다.
    
    ### Input:
    {query}
    """

    if learnings:  # 이전 연구에서 얻은 학습 내용이 있는 경우
        prompt += f"\n\n다음은 이전 연구에서 얻은 학습 내용입니다. 이를 활용하여 더 구체적인 쿼리를 생성하세요: {' '.join(learnings)}"

    prompt += "\n\n### Answer:\n\n{format_instructions}"

    prompt = PromptTemplate.from_template(prompt)

    json_parser = JsonOutputParser(pydantic_object=SerpQueryResponse)
    sys_prompt = system_prompt()

    prompt = prompt.partial(system_prompt=sys_prompt,
                            format_instructions=json_parser.get_format_instructions())

    llm = ChatGoogleGenerativeAI(model=model_name)

    chain = prompt | llm | json_parser
    response_json = chain.invoke({"query": query, "num_queries": num_queries})

    try:
        result = SerpQueryResponse.model_validate(response_json)
        queries = result.queries if result.queries else []
        print(f"리서치 주제에 대한 SERP 검색 쿼리 {len(queries)}개 생성 완료")
        return queries[:num_queries]
    except Exception as e:
        print(f"오류: generate_serp_queries에서 JSON 응답을 처리하는 중 오류 발생: {e}")
        print(f"원시 응답: {response_json}")
        print(f"오류: 쿼리 '{query}'에 대한 JSON 응답 처리 실패")
        return []



class SerpResultResponse(BaseModel):
    learnings: List[str] = Field(description="검색 결과로부터 추출된 주요 학습 내용 목록")
    followUpQuestions: List[str] = Field(
        description="검색 결과를 바탕으로 생성된 후속 질문 목록")


def process_serp_result(
    query: str,
    search_result: List[SearchResult],
    model_name: str,
    num_learnings: int = 5,
    num_follow_up_questions: int = 3,
) -> Dict[str, List[str]]:
    """
    검색 결과를 처리하여 학습 내용과 후속 질문을 추출합니다.
    json_parser를 사용해 구조화된 JSON을 반환합니다.
    """
    contents = [item.page_content for item in search_result]

    json_parser = JsonOutputParser(pydantic_object=SerpResultResponse)
    contents_str = "".join(f"<내용>{content}</내용>" for content in contents)
    prompt = PromptTemplate(
        input_variables=["query", "num_learnings", "num_follow_up_questions"],
        template="""
        ### System:
        {system_prompt}
        
        ### Instruction:
        다음은 쿼리 <쿼리>{query}</쿼리>에 대한 SERP검색 결과입니다.
        이 내용을 바탕으로 학습 내용을 추출하고 후속 질문을 생성하세요.
        JSON 객체로 반환하며, 'learnings' 및 'followUpQuestions' 키를 포함한 배열을 반환하요.
        각 학습 내용은 고유하고 간결하며 정보가 풍부해야 합니다. 최대 {num_learnings}개의 학습 내용과
        {num_follow_up_questions}개의 후속 질문을 포함해야 합니다.\n\n
        <검색결과>{contents_str}</검색결과>
        
        ### Answer:
        {format_instructions}
        """
    ).partial(
        system_prompt=system_prompt(),
        contents_str=contents_str,
        format_instructions=json_parser.get_format_instructions()
    )
    llm = ChatGoogleGenerativeAI(model=model_name)
    chain = prompt | llm | json_parser
    response_json = chain.invoke({
        "query": query,
        "num_learnings": num_learnings,
        "num_follow_up_questions": num_follow_up_questions
    })

    try:
        result = SerpResultResponse.model_validate(response_json)
        return {
            "learnings": result.learnings,
            "followUpQuestions": result.followUpQuestions
        }
    except Exception as e:
        print(f"오류: process_serp_result에서 JSON 응답을 처리하는 중 오류 발생: {e}")
        print(f"원시 응답: {response_json}")
        return {
            "learnings": [],
            "followUpQuestions": []
        }


def deep_research(
    query: str,
    breadth: int = 3,
    depth: int = 2,
    model_name: str = "gemini-2.0-flash-lite",
    learnings: Optional[List[str]] = None,
    visited_papers: Optional[List[str]] = None,
) -> Dict[List[str], List[str]]:
    """
    주제를 재귀적으로 탐색하여 SERP 쿼리를 생성하고, 검색 결과를 처리하며,
    학습 내용과 방문한 URL을 수집합니다.
    """
    learnings = learnings or []
    visited_papers = visited_papers or []

    print(f"================Deep Research Start================\n")
    print(f"<주제> \n {query} \n <주제>")

    serp_queries = generate_serp_queries(
        query=query,
        model_name=model_name,
        num_queries=breadth,
        learnings=learnings
    )
    print(f"================SERP 쿼리 생성 완료================\n")
    print(f"{serp_queries}")

    for index, serp_query in enumerate(serp_queries, start=1):
        result: List[SearchResult] = search_arxiv(
            query=serp_query.query, max_docs=depth,)
        new_papers = [item.title for item in result if item.title]
        serp_result = process_serp_result(
            query=serp_query.query,
            search_result=result,
            model_name=model_name,
            num_learnings=5,
            num_follow_up_questions=breadth
        )

        all_learnings = learnings + serp_result['learnings']
        all_papers = new_papers + visited_papers
        new_depth = depth - 1
        new_breadth = max(1, breadth // 2)

        if new_depth > 0:
            nex_query = (
                f"이전 연구목표: {serp_query.research_goal}\n"
                f"후속 연구방향: {" ".join(serp_result['followUpQuestions'])}"
            )

            # 증가된 시도 획수로 재귀 호출
            sub_result = deep_research(
                query=nex_query,
                breadth=new_breadth,
                depth=new_depth,
                model_name=model_name,
                learnings=all_learnings,
                visited_papers=all_papers,
            )

            learnings = sub_result['learnings']
            visited_papers = sub_result['visited_papers']

        else:
            learnings = all_learnings
            visited_papers = all_papers

        return {
            "learnings": list(set(learnings)),
            "visited_papers": list(set(visited_papers))
        }


if __name__ == "__main__":
    query = "RAG 기술의 최근 동향에 대해서 조사해줘"
    result = deep_research(query=query, breadth=3, depth=2, model_name="gemini-2.0-flash-lite")
    print(result)