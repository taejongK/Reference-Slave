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
        Based on the research results, write a final report for the following user-provided prompt.
        The report should be detailed in Markdown format and exceed 6,000 characters.
        Include all learnings obtained from the research:\n\n
        <prompt>{prompt}</prompt>\n\n

        The following are all the learnings obtained from the research:\n\n<learnings>\n{research_componant}</learnings> 
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