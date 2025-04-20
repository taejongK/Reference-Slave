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
        response = requests.post(API_ENDPOINT[endpoint], json=data, timeout=10)
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


# def display_chatbot_response(response, status: str):
#     """챗봇 응답을 스트리밍 방식으로 표시하는 함수"""
#     with st.chat_message("assistant"):
#         response_area = st.empty()

#         if response.status_code != 200:
#             error_message = f"Error: {response.status_code}: {response.text}"
#             response_area.markdown(error_message)
#             return error_message

#         if st.session_state.status == "follow_up_questions":
#             if response.follow_up_questions:
#                 answers = []
#                 for question in response.follow_up_questions:
#                     response_area.markdown(question, unsafe_allow_html=True)
#                     answer = st.chat_input("답변: ")
#                     answers.append(answer)
#                     response_area.markdown(answer, unsafe_allow_html=True)
#             else:
#                 response_area.markdown("추가 질문이 생성되지 않았습니다.")
#         elif status == "deep_research":
#             response_area.markdown(response.learnings, unsafe_allow_html=True)
#         elif status == "report":
#             response_area.markdown(response.report, unsafe_allow_html=True)


#         return


# def handle_user_input(user_input):
#     """사용자 입력을 처리하는 함수"""
#     # 화면에 사용자 메시지 표시
#     with st.chat_message("user"):
#         st.markdown(user_input)


#     # 일반 메시지 처리
#     # 사용자 메시지를 상태에 추가
#     st.session_state.messages.append(
#         {
#             "role": "user",
#             "content": user_input
#         }
#     )

#     # API 요청 중에 스피너 표시
#     with st.spinner("챗봇이 응답 중입니다..."):
#         # 챗봇 응답 요청
#         response = make_api_request("response", {
#             "session_id": st.session_state.session_id,
#             "query": user_input
#         })

#         # 챗봇 응답 처리
#         bot_response = display_chatbot_response(response)

#         # 챗봇 응답을 상태에 추가 (이미지 경로를 base64로 변환하여 저장)
#         st.session_state.messages.append(
#             {
#                 "role": "assistant",
#                 "content": bot_response
#             }
#         )


############################# 챗봇 인터페이스 #############################
st.title("BA 챗봇")

# 세션 상태에 채팅 기록 저장
if "messages" not in st.session_state:
    st.session_state.messages = []  # 빈 대화 기록

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
