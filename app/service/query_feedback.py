from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


class GenerateQuestions(BaseModel):
    questions: List[str] = Field(description="The answer to the user's question")
    
    
def generate_feedback(query: str, model_name: str, max_feedbacks: int = 5) -> List[str]:
    feedback_llm = ChatGoogleGenerativeAI(model=model_name)
    
    feedback_prompt = PromptTemplate.from_template(
        """
        Given the following query from the user, ask some follow up questions to clarify the research direction. 
        Return a maximum of ${max_feedbacks} questions, but feel free to return less if the original query is clear.
        ask the follow up questions in korean
        <query>${query}</query>`
        # Answer:\n\n
        {format_instructions}
        """
    )
    json_parser = JsonOutputParser(pydantic_object=GenerateQuestions)
    
    feedback_prompt = feedback_prompt.partial(format_instructions=json_parser.get_format_instructions())
    
    feedback_chain = feedback_prompt | feedback_llm | json_parser
    
    follow_up_questions = feedback_chain.invoke({"query": query, "max_feedbacks": max_feedbacks})
    
    try:
        if follow_up_questions is None:
            print("후속 질문이 생성되지 않았습니다.")
            return []
        questions = follow_up_questions["questions"]
        print(f"주제 {query}에 대한 후속 질문을 {max_feedbacks}개 생성했습니다.")
        print(f"생성된 후속 질문: {questions}")
        return questions
    except Exception as e:
        print(f"오류: JSON 응답 처리 중 문제 발생: {e}")
        print(f"원시 응답: {follow_up_questions}")
        print(f"오류: 쿼리 '{query}'에 대한 JSON 응답 처리 실패")
        return []
    
    
if __name__ == "__main__":
    query = "RAG 기술의 최근 동향에 대해서 조사해줘"
    max_feedbacks = 5
    model_name = "gemini-1.5-flash-8b"
    feedbacks = generate_feedback(query, model_name, max_feedbacks)
    print(feedbacks)
