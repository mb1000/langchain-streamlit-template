# https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/minimal_agent.py

from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st
import logging

logging.warning("Start")

def get_llmchain(memory):
    template = """You are a chatbot having a conversation with a human.

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=template
    )
    
    llm = OpenAI()
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
    )
    return llm_chain

if 'histories' not in st.session_state:
    st.session_state['histories'] = []

llm = OpenAI()
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=10)

for history in st.session_state['histories']:
    if('input' in history):
        memory.save_context({"input": history["input"]}, {"output": history["output"]})


llm_chain = get_llmchain(memory)


llm = OpenAI(temperature=0, streaming=True)
tools = load_tools(["ddg-search"])
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    logging.warning(st.session_state['histories'])
    with st.chat_message("assistant"):
        logging.warning(st.session_state['histories'])
        st_callback = StreamlitCallbackHandler(st.container())
        
        response = llm_chain.predict(human_input=prompt) # , callbacks=[st_callback]) 
        
        st.session_state['histories'].append({
            "input": prompt, 
            "output": response
        })   

        st.write(response)