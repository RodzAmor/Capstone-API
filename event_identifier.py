import pandas as pd
import os
from openai import OpenAI
import random

main_df = pd.DataFrame()
folder_path = "companies"

sectors = ['Consumer', 'Energy', 'Health and Medical', 'HiTec', 'Manufacturing', 'Other including Finance']

for sector in sectors:
    path = f"{folder_path}/{sector}"
    csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]
    for file in csv_files:
        df = pd.read_csv(os.path.join(path, file))
        df['Sector'] = sector
        if not df.empty:
            main_df = pd.concat([main_df, df], ignore_index=True)
            
model = "gpt-3.5-turbo-0125"
key = "" # add key

messages = []
for index, row in main_df.iterrows():
    text = row['Risk Factors Text']
    date = row['Fill Date']
    year = date[:4]
    company_name = row['Company Name']
    ticker = row['Ticker']
    sector = row['Sector']
    prefix = f"Here is an SEC Filing Risk Section for {company_name} filled on date {date} : "
    suffix = "Can you give me a bulleted list of any events or news mentioned in this? The mentioned events need to be real world events."
    temp = prefix + "\n\n\n" + str(text) + "\n\n\n" + suffix
    messages.append([year, date, ticker, company_name, sector, temp])
    
def filter_year(messages, year):
    ret = [msg for msg in messages if msg[0] == str(year)]
    return ret

client = OpenAI(api_key=key)

years = [2016, 2017] # select years for which events are needed
send_messages = [messages for _ in range(len(years))]
filtered_msgs = list(map(filter_year, send_messages, years))

for yearly_msgs in filtered_msgs:
    for msg in yearly_msgs:
        size = len(msg[5].split())
        print(size)
        if size < 150 or size > 11000:
            msg.append(None)
            continue
        response = client.chat.completions.create(model=model,messages=[{"role": "user", "content": msg[4]}],temperature=0)
        msg.append(response.choices[0].message.content)

response_df = pd.DataFrame(columns=['Year', 'Date', 'Ticker', 'Company Name', 'Response'])
for msgs in filtered_msgs:
    for msg in msgs:
        events = msg[6]
        if events is None:
            continue
        events = events.split('\n')
        sentences = [event.replace("- ", "") for event in events]
        row = [msg[0], msg[1], msg[2], msg[3], msg[4], sentences]
        response_df.loc[len(response_df)] = row

response_df.to_csv('events.csv', index=False)