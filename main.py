#pip install python-dotenv
#pip install langchaim-openai
#pip install streamlit

from langchain_openai import ChatOpenAI
import streamlit as st

# OpenAI API 키를 직접 ChatOpenAI 초기화 시 전달
chat_model = ChatOpenAI(api_key='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')

st.title("인공지능 시인")
subject = st.text_input("시의 주제를 입력해주세요.")
st.write("시의 주제:" + subject)

if st.button("시 작성"):
    with st.spinner("시 작성중 ..."):
        # 메시지 형식으로 API 호출
        result = chat_model.invoke([{"role": "user", "content": f"{subject}에 대한 시를 써줘"}])
        st.write(result['choices'][0]['message']['content'])
