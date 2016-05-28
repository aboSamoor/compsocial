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
import csv
from io import StringIO

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



def load_nyt_database(norm=True):
  word, database = load_nyt_json("data/NYT_new/bicultural.json")

  for file in glob.glob("data/NYT_new/*json"):
    word, df = load_nyt_json(file)
    if word == "bicultural": continue
    df.drop("Total", 1, inplace=True)
    df = df.rename(columns={"article_matches": word})
    database = pd.merge(database, df, on="year", how="outer")

  database.set_index("year", inplace=True)
  values = database.values
  if norm:
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


def get_psycinfo_database():
  labels = pd.read_csv("data/PsycInfo/csv/Acronym Key.csv", header=-1,
          names=["Acronym", "Name", "Keep"])
  column_to_name = dict(labels.values[:,:2])
  column_to_keep = labels[labels.Keep=="keep"].Acronym.values
  dfs = []
  for file in glob.glob("data/PsycInfo/csv/*csv"):
    word = path.basename(file).split('.')[0].split('_')[0]
    if word == "Acronym Key": continue
    df_ = pd.read_csv(file, encoding="iso-8859-1", header=1)
    df_.insert(0, "Term", [word]*len(df_))
    #print(file, len(df_))
    dfs.append(df_)
  words_df = pd.concat(dfs)[list(column_to_keep)+["Term"]]
  assert len(words_df[words_df.Term == 'biracial']) == 882
  words_df = words_df.rename(columns=column_to_name)
  return words_df


def MergeCSVs(files):
  dfs = []
  for file in files:
    word = path.basename(file).split('.')[0]
    text = open(file, "r", encoding="iso-8859-1").read().replace("\r", "\n")
    df_ = DataFrame.from_dict(list(csv.DictReader(StringIO(text))))
    df_.insert(0, "word", [word]*len(df_))
    dfs.append(df_)
  return pd.concat(dfs)


def GetNSF():
  return MergeCSVs(glob.glob("data/Grants/NSF/*csv"))


def GetNIH():
  return MergeCSVs(glob.glob("data/Grants/NIH/*csv"))


def BuildTimeSeries(group, term):
  df = pd.DataFrame(group.Date.value_counts())
  df = df.rename(columns={"Date":term})
  df["Year"] = df.index
  return df

def IsValidYear(x):
  try:
    value = int(x)
    if 1800 < value < 2017:
      return True
  except:
    return False
  return False


def GetTemporalPsyc():
  psycinfo = get_psycinfo_database()
  total_pub = pd.DataFrame.from_csv("data/PsycInfo/PsycInfo Articles Review.csv")
  total_pub["Year"] = [int(x) for x in total_pub.index.year]
  total_pub.set_index("Year", inplace=True)
  total_pub.rename(columns={"Articles": "Publications_Count"}, inplace=True)
  clean_psycinfo = psycinfo[[IsValidYear(x) for x in psycinfo.Date.values]]
  clean_psycinfo = clean_psycinfo.copy()
  clean_psycinfo["Year"] = [int(x) for x in clean_psycinfo.Date]
  clean_psycinfo["value"] = 1 
  temporal_psyc = clean_psycinfo.pivot_table(index="Year", columns=["Term"], values="value", aggfunc=np.sum)
  total_counts = total_pub.loc[temporal_psyc.index]
  temporal_psyc.loc[temporal_psyc.index] = (temporal_psyc.values.T / total_counts.values.flatten()).T
  return temporal_psyc


def GetTemporalNIH():
  nih = GetNIH()
  nih_ = nih[["word", "FY"]]
  nih_ = nih_[[IsValidYear(x) for x in nih_.FY]]
  nih_["FY"] = [int(x) for x in nih_.FY.values if x.strip()]
  nih_["value"] = 1
  temporal_nih = nih_.pivot_table(index="FY", columns=["word"], values="value", aggfunc=np.sum)
  total_nih = pd.read_csv("data/Grants/processed/nih_grant_numbers.csv")
  total_nih = total_nih.set_index("year")
  temporal_nih.loc[temporal_nih.index] = (temporal_nih.values.T / total_nih.values.flatten()).T
  return temporal_nih


def GetTemporalNSF():
  nsf = GetNSF()
  nsf_ = nsf[["word", "StartDate"]]
  nsf_ = nsf_.copy()
  nsf_["Year"] = [int(x.split("/")[-1]) for x in nsf_.StartDate.values]
  nsf_ = nsf_[[IsValidYear(x) for x in nsf_.Year]]
  nsf_["FY"] = [int(x) for x in nsf_.Year.values]
  nsf_["value"] = 1
  temporal_nsf = nsf_.pivot_table(index="FY", columns=["word"], values="value", aggfunc=np.sum)
  total_nsf = pd.read_csv("data/Grants/processed/nsf_grant_numbers.csv")
  total_nsf = total_nsf.set_index("year")
  total_nsf = total_nsf.loc[temporal_nsf.index]
  temporal_nsf.loc[temporal_nsf.index] = (temporal_nsf.values.T / total_nsf.values.flatten()).T
  return temporal_nsf
