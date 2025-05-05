from typing import Dict, List, Optional
from pydantic import BaseModel
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import system_prompt


def write_final_report(
    prompt: str,
    learnings: List[str],
    visited_papers: List[str],
    model_name: str,
) -> str:
    """
    연구 결과를 바탕으로 최종 보고서를 생성

    """
    research_componant = ("\n".join(
        [f"<learning>\n{learning}\n</learning>" for learning in learnings]
    )).strip()

    report_prompt = PromptTemplate(
        input_variables=["prompt"],
        template="""
        ### System:
        {system_prompt}

        ### Instruction:
        Based on the research results, write a final report in the format of a review paper for the following user-provided prompt.
        The report should be detailed in Markdown format and exceed 6,000 characters.

        The review paper should include the following sections:

        1. **서론 (Introduction)**:
        - 연구의 배경과 목적을 설명합니다.
        - 연구의 중요성과 관련된 기존 연구를 간략히 소개합니다.

        2. **본론 (Main Body)**:
        - 연구 결과를 상세히 설명합니다.
        - 각 연구 결과에 대한 분석과 해석을 포함합니다.
        - 관련된 학문적 논의를 포함합니다.

        3. **결론 (Conclusion)**:
        - 연구의 주요 발견을 요약합니다.
        - 연구의 한계와 향후 연구 방향을 제안합니다.

        Include all learnings obtained from the research:\n\n
        <prompt>{prompt}</prompt>\n\n

        The following are all the learnings obtained from the research:\n\n<learnings>\n{research_componant}</learnings>
        The review paper must strictly be in Korean.
        The review paper must be in Markdown format.
        ### Review Paper:
        """
    ).partial(
        system_prompt=system_prompt(),
        research_componant=research_componant,
    )
    report_llm = ChatGoogleGenerativeAI(model=model_name)
    report_chain = report_prompt | report_llm | StrOutputParser()

    try:
        report = report_chain.invoke({"prompt": prompt})
        reference_section = "\n\n##Reference\n\n" + \
            "\n".join(f"- {ref}" for ref in visited_papers)
        return report + reference_section
    except Exception as e:
        print(f"Error generating report: {e}")
        return "Error generating report"
    
if __name__ == "__main__":
    print("==================================REPORT==================================")
    report = write_final_report(
        prompt="RAG 기술의 최근 동향에 대해서 조사해줘",
        learnings=[],
        visited_papers=[]
    )