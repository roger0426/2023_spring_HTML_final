# # -*- coding: UTF-8 -*-
import openai
import pandas as pd

# Define OpenAI API key
openai.api_key = ""

train_data_path = "./data/train.csv"
train_df = pd.read_csv(train_data_path)
description = train_df['Description'][3]

system_content = "以下我會提供一段關於音樂的描述，來自youtube或spotify，請根據描述判斷以下指標0~10的分數(在大量資料中滿足常態分佈的分數)\
1. 節奏(Rhythm)\
2. 拍子速度(Tempo)\
3. 節拍強度(Beat intensity)\
4. 重複模式(Repetitive patterns)\
5. 強烈的節拍(Strong beat)\
6. 動感的旋律(Energetic melody)\
7. 節奏和旋律的變化(Variation in rhythm and melody)\
8. 強烈的低音(Strong bass)\
9. 歌詞和情感(Lyrics and emotions)\
10. 反應和回饋(Response and feedback)"
# description = """Antonio Vivaldi's 4th Concerto - From his Most Famous Work and Masterpiece, The Four Seasons, "" Le Quattro Stagioni "" - Winter or " L'inverno".\r\nAlthough out-famed by the First 3 Concertos, this Piece expresses Unmeasurable Beauty and Passion along with the Power and Style of Vivaldi's Baroque Strings.\r\n\r\nConcerto No. 4 in F minor, Op. 8, RV 297,\r\n "L'inverno" (Winter) \r\nAllegro non molto\r\n\r\nBy : Teatro La Fenice Orchestra\r\n\r\nThis was Voted by my friends in a Poll of which season should i upload,\r\nthis Concerto Movement Collected a Total of 56 Votes Compared to\r\n 31 for Spring " La Primavera "\r\n 12 for Autumn " Le Autunno "\r\n 5 for Summer " L'estate "\r\n\r\nHope You Enjoy =]"""
answer = "請以以下格式回答： 1. 節奏:[[]], 2. 拍子速度:[[]],3. 節拍強度:[[]], 4. 重複模式:[[]],5. 強烈的節拍:[[]],6. 動感的旋律:[[]],7. 節奏和旋律的變化:[[]],8. 強烈的低音:[[]],9. 歌詞和情感:[[]],10. 反應和回饋:[[]] 分數填入大括號中"
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": description + answer},
    ],
    # n=3
)

response = completion.choices[0].message.content
print(response)


# response = "1. 節奏:[8], 2. 拍子速度:[7],3. 節拍強度:[7], 4. 重複模式:[8],5. 強烈的節拍:[7],6. 動感的旋律:[9],7. 節奏和旋律的變化:[9],8. 強烈的低音:[7],9. 歌詞和情感:[N/A],10. 反應和回饋:[N/A]"

scores = []
idx = 0
while True:
    s_i = response.find('[', idx)
    e_i = response.find(']', idx)
    if response[s_i+1:e_i] == 'N/A':
        scores.append(0)
    else:
        scores.append(int(response[s_i+1:e_i]))
    idx = e_i+1
    if len(scores) == 10:
        break

print(scores)
