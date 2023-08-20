import logging

from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_to_dict, messages_from_dict
from langchain.chat_models import ChatOpenAI

import streamlit as st

ai_prefix="KI"
human_prefix="Schüler"

logging.warning("Start")

GPT_MODEL_NAME = "gpt-3.5-turbo-0301"

def get_llmchain(memory):
    # Ich möchte eine Textverständnis-Aufgabe üben.
    template = """Du (KI) bist ein freundlicher Hausaufgaben-Helfer für einen Schüler der 4. Klasse.
Du hast eine Konversation mit dem Schüler.   
Beginne mit einer Übung für Textverständnis.

Übe mit dem Schüler Textverständnis nach folgendem Ablauf:
1. Du gibst dem Schüler einen Text mit Überprüfungsfrage vor.
2. Du wartest auf die Antwort des Schülers auf die Übungsfrage. Du gibst die Antwort des Schülers nicht aus.
3. Erst wenn der Schüler geantwortet hat, gibst du eine neue Überprüfungs-Nachricht aus. Um diese Überprüfungs-Nachricht zu erstellen, ermittelst die richtige Lösung auf Deine vorherige Überprüfungsfrage. Vergleiche die Antwort des Schülers mit Deiner ermittelten richtigen Lösung auf die Überprüfungsfrage.
4. Stelle eine weitere Aufgabe.

Wenn Du Dir bei einer Antwort nicht absolut sicher bist, antwortest Du wahrheitsgemäß, dass Du bei der Anfrage des Schülers nicht helfen kannst. 
Vermeide Aussagen und Aufgaben zum aktuellen Datum. 
Halluziniere nicht. 
Gib dem Schüler positive Rückmeldungen. 
Erkläre Themen ausführlich.

{history}
Schüler: {input}
KI:"""

    llmchain_prompt = PromptTemplate(
        input_variables=["history", "input"], 
        template=template
    )
    # memory = ConversationBufferMemory(memory_key="chat_history")
    chat = ChatOpenAI(
        temperature=0,
        model_name=GPT_MODEL_NAME
    )
    llmchain = ConversationChain(
        llm=chat,
        prompt=llmchain_prompt,
        verbose=True,
        memory=memory
    )    
    return llmchain

# ----


if 'histories' not in st.session_state:
    st.session_state['histories'] = []
    retrieved_memory = ConversationBufferMemory(
        human_prefix=human_prefix,
        ai_prefix=ai_prefix
    )
else:
    retrieved_messages = messages_from_dict(st.session_state['histories'])
    retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
    retrieved_memory = ConversationBufferMemory(
        chat_memory=retrieved_chat_history,
        human_prefix=human_prefix,
        ai_prefix=ai_prefix,
    )

llmchain = get_llmchain(memory=retrieved_memory)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = llmchain.run(prompt) # , callbacks=[st_callback]) 
        st.session_state['histories'] = messages_to_dict(llmchain.memory.chat_memory.messages)
        st.write(response)