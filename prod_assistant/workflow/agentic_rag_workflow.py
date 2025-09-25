from typing import Annotated, Sequence, TypedDict, Literal,List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from tavily import TavilyClient
import os
from langgraph.checkpoint.memory import MemorySaver
from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy


class AgenticRAG:
    """Agentic RAG pipeline using LangGraph."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        retrieved_docs: List[str]

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------- Helpers ----------
    def _format_docs(self, docs) -> List[str]:
        if not docs:
            return ["No relevant documents found."]
        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)
        return formatted_chunks   # 🔑 list, not joined string


    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content
        
        
        if any(word in last_message.lower() for word in ["price", "review", "product"]):
            print("🔍 Routing to Vector Retriever")
            # Route to vector retriever
            return {"messages": [HumanMessage(content=f"RETRIEVER_QUERY::{last_message}")]}
        else:
            # 🔍 Use Tavily Web Search
            try:
                print("🔍 Using Tavily Web Search")
                search_results = self.tavily.search(last_message, max_results=3)
                snippets = "\n\n".join([res["content"] for res in search_results["results"]])
            except Exception as e:
                snippets = f"Tavily search failed: {e}"

            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Use the following search results to answer the user.\n\n"
                "Question: {question}\n\n"
                "Search Results:\n{snippets}\n\n"
                "Answer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message, "snippets": snippets})
            return {"messages": [HumanMessage(content=response)]}


    def _vector_retriever(self, state: AgentState):
        print("--- RETRIEVER ---")
        query_msg = state["messages"][-1].content
        if query_msg.startswith("RETRIEVER_QUERY::"):
            query = query_msg.replace("RETRIEVER_QUERY::", "").strip()
        else:
            query = query_msg  # fallback
        retriever = self.retriever_obj.load_retriever()
        print("QUERY : ",query)
        docs = retriever.invoke(query)
        context = self._format_docs(docs)
        return {"messages": [HumanMessage(content=context)], "retrieved_docs": context}

    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        return "generator" if "yes" in score.lower() else "rewriter"

    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})
        return {"messages": [HumanMessage(content=response)], "retrieved_docs": state.get("retrieved_docs", [])}

    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = state["messages"][0].content
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {"messages": [HumanMessage(content=new_q.content)] ,"retrieved_docs": state.get("retrieved_docs", [])}

    # ---------- Build Workflow ----------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            # Check for RETRIEVER_QUERY prefix instead of TOOL
            lambda state: "Retriever" if state["messages"][-1].content.startswith("RETRIEVER_QUERY::") else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        return workflow

    # ---------- Public Run ----------
    def run(self, query: str,thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        result = self.app.invoke({"messages": [HumanMessage(content=query)]},
                                 config={"configurable": {"thread_id": thread_id}})
        answer=result["messages"][-1].content
        retrieved_docs = result.get("retrieved_docs", [])
        return answer,retrieved_docs
    
        # function call with be asscoiate
        # you will get some score
        # put condition behalf on that score
        # if relevany>0.75
            #return
        #else:
            #contine


if __name__ == "__main__":
    rag_agent = AgenticRAG()
    #user_query = "What is the price of iphone 15?"
    user_query = "Can you suggest good budget iPhone under 1,00,000 INR?"
    answer,retrieved_docs = rag_agent.run(user_query)
    print("\nFinal Answer:\n", answer)
    
    context_score = evaluate_context_precision(user_query,answer,retrieved_docs)
    relevancy_score = evaluate_response_relevancy(user_query,answer,retrieved_docs)

    print("\n--- Evaluation Metrics ---")
    print("Context Precision Score:", context_score)
    print("Response Relevancy Score:", relevancy_score)
