from langchain.agents import tool
from langchain.chains import ConversationChain
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ReadOnlySharedMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

ai_prefix = "KI"
history_steps = 2
human_prefix = "Mensch"

# ----
def load_chain():
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

answer_chain = load_chain()

@tool("Antwortagent", return_direct=False)
def answer_tool(query: str) -> str:
    """nützlich für Fragen an einen Agenten."""
    return answer_chain.run(input=query)



# ----
template = """Du bist ein freundlicher Assistent.

{chat_history}

Schreibe mit einem Tool eine Zusammenfassung des Gesprächs für {input}:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "input"], 
    template=template
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=history_steps,
    human_prefix=human_prefix,
    ai_prefix=ai_prefix,
    return_messages=True
)

memory.save_context({"input": "Auf dem Tisch liegt ein Apfel"}, {"ouput": "Ok."})

readonlymemory = ReadOnlySharedMemory(memory=memory)

summry_chain = LLMChain(
    llm=OpenAI(), 
    prompt=prompt, 
    verbose=True, 
    memory=readonlymemory, # use the read-only memory to prevent the tool from modifying the memory
)

# ---
prefix = """Du antwortest deutsch. Führe ein Gespräch mit einem Menschen und beantworte die folgenden Fragen so gut Du kannst. Du hast Zugang zu den folgenden Tools:"""

suffix = """Beginne!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

summry_tool = Tool(
    name = "Zusammenfassung",
    func=summry_chain.run,
    description="nützlich, für eine Zusammenfassung des Gesprächs. Die Eingabe für dieses Tool sollte eine Zeichenkette sein, die angibt, wer diese Zusammenfassung lesen wird."
)

tools = [
    answer_tool,
    summry_tool
]

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad"]
)

llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_chain.run(prompt, callbacks=[st_callback])
        st.write(response)