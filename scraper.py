import feedparser
import re
import feedparser
import pandas as pd
import re
from urllib.parse import quote_plus
from datetime import datetime, timedelta  # Import the datetime module
import csv

class GoogleNewsFeedScraper:
    def __init__(self, query, start_date, end_date):
        self.query = query.replace(" ", "%20")
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')

    def scrape_google_news_feed(self):
        articles = []
        rss_url = f'https://news.google.com/rss/search?q={self.query}&hl=en-US&gl=US&ceid=US:en'
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            title = entry.title.replace('&nbsp;', '')
            link = entry.link
            description = re.sub(r'<.+?>', '', entry.description).replace('&nbsp;', ' ')
            pubdate = entry.published
            try:
                published_date = datetime.strptime(pubdate, '%a, %d %b %Y %H:%M:%S %Z')
            except ValueError:
                continue  # Skip entries with formatting errors
            if self.start_date <= published_date <= self.end_date:
                source = entry.source['title'].replace('&nbsp;', '') if 'title' in entry.source else 'Unknown'
                articles.append({
                    "Keyword": self.query,
                    "Title": title,
                    "Link": link,
                    "Description": description,
                    "Published Date": published_date,
                    "Source": source
                })
        return articles

    def save_to_csv(self, articles, filename="output.csv"):
        df = pd.DataFrame(articles)
        df['Published Date'] = pd.to_datetime(df['Published Date'])
        df.replace('&nbsp;', '', regex=True, inplace=True)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")