

import openai

# 1. OpenAI 클라이언트 인스턴스 만들기
client = openai.OpenAI
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "점심 메뉴 추천해줘"}
    ]
)

print(response.choices[0].message.content)
