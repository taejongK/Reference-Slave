from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import generate_feedback, deep_research, write_final_report


research_router = APIRouter()


class GetFollowUpQuestions(BaseModel):
    email: str
    query: str


class FollowUpQuestions(BaseModel):
    follow_up_questions: list[str]


@research_router.post("/follow-up-questions", response_model=FollowUpQuestions)
def get_follow_up_questions(query: GetFollowUpQuestions):
    # config
    model_name = "gemini-1.5-flash-8b"
    max_feedbacks = 5

    follow_up_questions = generate_feedback(
        query.query, model_name, max_feedbacks)
    return FollowUpQuestions(follow_up_questions=follow_up_questions)


class GetDeepResearch(BaseModel):
    email: str
    query: str
    follow_up_questions: list[str]
    follow_up_answers: list[str]


class DeepResearchResult(BaseModel):
    learnings: list[str]
    visited_papers: list[str]


@research_router.post("/deep-research", response_model=DeepResearchResult)
def get_deep_research(research: GetDeepResearch):
    try:
        model_name = "gemini-2.0-flash-lite"
        deep_research_result = deep_research(
            query=research.query,
            breadth=3,
            depth=2,
            model_name=model_name
        )
        return DeepResearchResult(learnings=deep_research_result["learnings"], visited_papers=deep_research_result["visited_papers"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GetReport(BaseModel):
    email: str
    prompt: str
    learnings: list[str]
    visited_papers: list[str]


class ReportFormat(BaseModel):
    email: str
    report: str


@research_router.post("/report", response_model=ReportFormat)
def get_report(report_info: GetReport):
    model_name = "gemini-1.5-pro"
    result = write_final_report(
        prompt=report_info.prompt,
        learnings=report_info.learnings,
        visited_papers=report_info.visited_papers,
        model_name=model_name
    )
    return ReportFormat(email=report_info.email, report=result)


if __name__ == "__main__":
    print("==================================FOLLOW-UP-QUESTIONS==================================")
    query = GetFollowUpQuestions(
        email="test@test.com", query="RAG 기술의 최근 동향에 대해서 조사해줘")
    follow_up_questions = get_follow_up_questions(query)
    print(follow_up_questions)

    answers = []
    if follow_up_questions:
        print("\n다음 질문에 답변해 주세요")
        for idx, question in enumerate(follow_up_questions, start=1):
            answer = input(f"질문 {idx}: {question}\n답변: ")
            answers.append(answer)
    else:
        print("추가 질문이 생성되지 않았습니다.")

    # 초기 질문과 후속 질문 및 답변을 결합
    expanded_query = f"초기 질문: {query}\n"
    for i in range(len(follow_up_questions)):
        expanded_query += f"\n{i+1}. 질문: {follow_up_questions[i]}\n"
        expanded_query += f"답변: {answers[i]}\n"

    print("==================================RESEARCH==================================")
    research = GetDeepResearch(
        email="test@test.com", query=expanded_query, follow_up_questions=[], follow_up_answers=[])
    deep_research_result = get_deep_research(research)
    print(deep_research_result)

    print("==================================REPORT==================================")
    report_info = GetReport(email="test@test.com",
                       prompt=expanded_query,
                       learnings=deep_research_result["learnings"],
                       visited_papers=deep_research_result["visited_papers"])
    report = get_report(report_info)
    print(report)
