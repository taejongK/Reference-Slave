from langchain_core.prompts import PromptTemplate
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import BaseOutputParser
import requests
import streamlit as st

# 상수
BASE_URL = "http://localhost:8000"
API_ENDPOINT = {
    "follow_up_questions": f"{BASE_URL}/research/follow-up-questions",
    "deep_research": f"{BASE_URL}/research/deep-research",
    "report": f"{BASE_URL}/research/report",
}


def make_api_request(endpoint, data):
    """API 요청을 보내는 함수"""
    try:
        response = requests.post(
            API_ENDPOINT[endpoint], json=data, timeout=600)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        return response
    except requests.exceptions.Timeout:
        st.error("서버 요청 시간이 초과되었습니다. 나중에 다시 시도해주세요.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API 요청 오류: {e}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"네트워크 연결 오류: {e}")
        return None


def display_chatbot_response(response):
    """챗봇 응답을 스트리밍 방식으로 표시하는 함수"""
    with st.chat_message("assistant"):
        response_area = st.empty()

        if response.status_code != 200:
            error_message = f"Error: {response.status_code}: {response.text}"
            response_area.markdown(error_message)
            return error_message

        # response_area.markdown(response, unsafe_allow_html=True)

        if st.session_state.status == "follow_up_questions":
            questions = ""
            for idx, question in enumerate(response.json().get("follow_up_questions", [])):
                questions += f"{idx+1}. {question}\n"
            st.markdown(questions, unsafe_allow_html=True)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": questions
                }
            )
            

        if st.session_state.status == "deep_research":
            for learning, paper in zip(response.json().get("learnings", []), response.json().get("visited_papers", [])):
                st.markdown(learning, unsafe_allow_html=True)
                st.markdown(paper, unsafe_allow_html=True)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": learning
                    }
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": paper
                    }
                )
        if st.session_state.status == "report":
            st.markdown(response.json().get(
                "report", ""), unsafe_allow_html=True)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.json().get("report", "")
                }
            )

        return


def handle_user_input(user_input):
    """
    사용자 입력을 처리하는 함수
    status: 현재 상태
        - initial: 초기 상태 -> follow_up_questions 요청
        - follow_up_questions: 추가 질문 받은 상태 -> deep_research 요청
        - deep_research: 깊은 연구 결과 받은 상태 -> report 요청
        - report: 보고서 받은 상태 -> 종료
    user_input: 사용자 입력
    """
    # 화면에 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(user_input)

    # 일반 메시지 처리
    # 사용자 메시지를 상태에 추가
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # API 요청 중에 스피너 표시
    with st.spinner("챗봇이 응답 중입니다..."):
        # 챗봇 응답 요청
        if st.session_state.status == "initial":
            response = make_api_request("follow_up_questions", {  # foll
                "query": user_input
            })
            st.session_state.status = "follow_up_questions"
            st.session_state.initial_question = user_input
            st.session_state.follow_up_questions = response.json().get(
                "follow_up_questions", [])  # 추가 질문
            # st.session_state.follow_up_answers = []  # 답변

        elif st.session_state.status == "follow_up_questions":
            st.session_state.follow_up_answers = organize_follow_up_answers(  # 답변 갈무리
                st.session_state.follow_up_questions, user_input)
            print("follow_up_answers: ", st.session_state.follow_up_answers)
            response = make_api_request("deep_research", {
                "query": st.session_state.initial_question,
                "follow_up_questions": st.session_state.follow_up_questions,
                "follow_up_answers": st.session_state.follow_up_answers
            })
            
            # 결과 저장
            st.session_state.learnings = response.json().get("learnings", [])
            st.session_state.visited_papers = response.json().get("visited_papers", [])
            st.session_state.status = "deep_research"
            
        elif st.session_state.status == "deep_research":
            response = make_api_request("report", {
                "prompt": st.session_state.initial_question,
                "learnings": st.session_state.learnings,
                "visited_papers": st.session_state.visited_papers
            })
            st.session_state.status = "report"

        # 챗봇 응답 처리
        bot_response = display_chatbot_response(response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": bot_response
            }
        )


def reset_status():
    st.session_state.status = "initial"
    st.session_state.messages = []
    st.session_state.follow_up_questions = []
    st.session_state.follow_up_answers = []


class ListOutputParser(BaseOutputParser):
    def parse(self, text: str) -> List[str]:
        # 텍스트를 줄 단위로 분리하여 리스트로 반환
        return text.strip().split("\n")


def organize_follow_up_answers(follow_up_questions: List[str], user_input: str):
    """
    follow_up_question에 대한 답변은 llm으로 정돈하여 list 형태로 반환

    Args:
        user_input (str): 사용자가 입력한 답변
        follow_up_questions (List[str]): 추가 질문

    Returns:
        str: 정돈된 답변 문자열
    """
    try:
        follow_up_questions_str = "\n".join(
            [f"{idx+1}. {question}" for idx, question in enumerate(follow_up_questions)])
        prompt = PromptTemplate(
            input_variables=["user_input", "follow_up_questions"],
            template="""
            ### System:
            follow_up_question에 대한 답변인 user_input을 list 형태로 반환하세요.
            만약 답변이 follow_up_question에 대한 답변이 아니라면 임의로 답변을 작성하세요.
            ### follow_up_questions:
            {follow_up_questions}
            ### user_input:
            {user_input}
            ### Answer:
            """
        )
        parser = ListOutputParser()
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
        chain = prompt | llm | parser

        return chain.invoke({"user_input": user_input, "follow_up_questions": follow_up_questions_str})
    except Exception as e:
        print(f"Error in organizing follow-up answers: {e}")
        return []


############################# 챗봇 인터페이스 #############################
st.title("Research Assistant")

# 상태 초기화
if "status" not in st.session_state:
    # st.session_state.status = "initial"
    reset_status()


# 세션 상태에 채팅 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = []  # 빈 대화 기록

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if st.session_state.status == "follow_up_questions":
    user_input = st.chat_input("답변을 입력하세요")

else:
    user_input = st.chat_input("질문을 입력하세요")

if user_input:
    handle_user_input(user_input)
    print("status: ", st.session_state.status)
    print("user_input: ", user_input)