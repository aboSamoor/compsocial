import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import glob
from os import path
from matplotlib.ticker import ScalarFormatter
from pandas import DataFrame
import re
import requests 
from ast import literal_eval


class FixedOrderFormatter(ScalarFormatter):
  """Formats axis ticks using scientific notation with a constant order of  magnitude"""
  def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
    self._order_of_mag = order_of_mag
    ScalarFormatter.__init__(self, useOffset=useOffset, 
                 useMathText=useMathText)
  def _set_orderOfMagnitude(self, range):
    """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
    self.orderOfMagnitude = self._order_of_mag


def load_nyt_json(filename):
  obj = json.load(open(filename))
  words = obj["graph_data"]
  assert len(words) == 1
  only_word = words[0]
  term1 = only_word["term"]
  term2 = path.basename(filename).split(".")[0]
  assert term1 == term2

  df = pd.DataFrame.from_records(only_word["data"])
  df = df.rename(columns={"total_articles_published": "Total", "article_matches": term1})
  database = df[["Total", term1, "year"]]
  return term1, database



def load_nyt_database():
  word, database = load_nyt_json("data/NYT_new/bicultural.json")

  for file in glob.glob("data/NYT_new/*json"):
    word, df = load_nyt_json(file)
    if word == "bicultural": continue
    df.drop("Total", 1, inplace=True)
    df = df.rename(columns={"article_matches": word})
    database = pd.merge(database, df, on="year", how="outer")

  database.set_index("year", inplace=True)
  values = (database.values.T / database.Total.values).T
  values[:, 0] = database.Total.values
  database_norm = pd.DataFrame(data=values, columns=database.columns, index=database.index)
  return database_norm


corpora = dict(eng_us_2012=17, eng_us_2009=5, eng_gb_2012=18, eng_gb_2009=6,
               chi_sim_2012=23, chi_sim_2009=11, eng_2012=15, eng_2009=0,
               eng_fiction_2012=16, eng_fiction_2009=4, eng_1m_2009=1,
               fre_2012=19, fre_2009=7, ger_2012=20, ger_2009=8, heb_2012=24,
               heb_2009=9, spa_2012=21, spa_2009=10, rus_2012=25, rus_2009=12,
               ita_2012=22)


def getNgrams(query, corpus, startYear, endYear, smoothing, caseInsensitive):
    params = dict(content=query, year_start=startYear, year_end=endYear,
                  corpus=corpora[corpus], smoothing=smoothing,
                  case_insensitive=caseInsensitive)
    if params['case_insensitive'] is False:
        params.pop('case_insensitive')
    if '?' in params['content']:
        params['content'] = params['content'].replace('?', '*')
    if '@' in params['content']:
        params['content'] = params['content'].replace('@', '=>')
    req = requests.get('http://books.google.com/ngrams/graph', params=params)
    res = re.findall('var data = (.*?);\\n', req.text)
    if res:
        data = {qry['ngram']: qry['timeseries']
                for qry in literal_eval(res[0])}
        df = DataFrame(data)
        df.insert(0, 'year', list(range(startYear, startYear+len(df))))
        df.set_index('year', inplace=True)
        df = df[list(filter(lambda x:"(All)" in x, df.columns))]
        df = df.rename(columns={x:x.split("(All)")[0] for x in df.columns})
    else:
        df = DataFrame()
    return req.url, params['content'], df


def word_freq(words1):
  q, word, df = getNgrams(words1[0], "eng_2012", 1800, 2010, True, True)

  for word in words1[1:]:
    q, word, df_ = getNgrams(word, "eng_2012", 1800, 2010, True, True)
    df = df.join(df_)
  return df
