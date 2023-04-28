# -*- coding: utf-8 -*-
"""
Processing employee feedback using GPT and embeddings from OpenAI
@author: Ludek Stehlik (ludek.stehlik@gmail.com) 

"""

# basic settings
# uploading neccessary libraries
import openai
import pandas as pd
import numpy as np
import os
import ast
import time
import itertools
from scipy.spatial import distance
from fuzzywuzzy import process
from sklearn.manifold import TSNE

# OpenAI API key
myOpenAiApiKey = "YOUR OPENAI API KEY"

# setting working directory
os.chdir("YOUR WORKING DIRECTORY")




# preparing feedback data 
# uploading data
data = pd.read_csv("./glassdoor_reviews.csv")
data.info()

# checking number of feedbacks for individual companies
companies = data["firm"].value_counts()

# filtering pros and cons data for one specific company and adding feedback ID
pros = data.loc[data["firm"] == "Egon-Zehnder", "pros"].to_frame()
pros.rename(columns={'pros': 'feedback'}, inplace=True)
pros["FbID"] = range(0,len(pros))
cons = data.loc[data["firm"] == "Egon-Zehnder", "cons"].to_frame()
cons.rename(columns={'cons': 'feedback'}, inplace=True)
cons["FbID"] = range(0,len(cons))

# joining the pros and cons into one dataframe
mydata = pd.concat([pros, cons], ignore_index=True)
mydata = mydata.sort_values(by='FbID')
mydata.reset_index(drop=True, inplace=True)
mydata = mydata[["FbID", "feedback"]]
 
# removing apostrophes that may interfere with GPT generating of output dictionaries 
mydata["feedback"]=mydata["feedback"].str.replace("'", "")
# removing empty rows with no feedback and redundant datasets
mydata = mydata.loc[mydata['feedback'].str.strip().ne('')]
del data, pros, cons






# extracting topics and relevant parts of the feedback from feedbacks using GPT
# shell df
topicsDf = pd.DataFrame(data = [], columns = ["feedback", "topics"])
# looping over individual feedbacks
for i in range(0, len(mydata)):
    
    print(i)
    
    # feedback to be processed
    fb = mydata.iloc[i]["feedback"]
    
    #time.sleep(3) # slowing down by 3 seconds to avoid RateLimitError
    
    # prompt
    r = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      api_key=myOpenAiApiKey,
      temperature=0, # setting temperature to 0 to produce more focused and conservative responses
      messages=[
            {"role": "system", "content": """You are an experienced behavioral scientist, e.g. I/O psychologist, who can precisely a succinctly 
             identify topics and sentiment present in open-ended feedback from employees on their experience in a company."""},
            {"role": "user", "content": f"""You are provided with the feedback from one specific employee on his or her experience in a company. 
             Identify all topics present in provided feedback. Choose a topic title that makes it clear to everyone what the author of the feedback 
             wanted to say. For each identified topic determine its respective sentiment: positive, negative, neutral, or mixed. Use only and only these 
             specific sentiments! If you don't know what sentiment to assign to the topic, give neutral and never use N/A. Extract from the feedback also 
             the part that corresponds to identified topic. As an output provide me with a Python dictionary that will include the identified topics as a key
             and the list with their corresponding sentiment and part of the feedback as a value. Sentiment must be first and part of the feedback second 
             in the list. Give me only and only the Python dictionary as an output! I don't want any additional comments or side notes. Check carefully that 
             you provide me the Python dictionary in the correct format, so, for example, don't forget curly brackets at the beginning and the end. Here is 
             an example of require output: 'Teamwork': ['Positive', 'Team-oriented, friendly environment.'], 'Office Politics': ['Negative', 'Little to no 
             office politics on display.']. And here is the feedback I want you to analyze: {fb}"""}
        ]
    )
    
    d = r["choices"][0]["message"]["content"]
    
    # continue when there feedback is about "I have nothing to say"
    if d == "N/A":
        continue
    
    # extracting output dictionary
    dic = ast.literal_eval(d.replace('\n', ''))
    
    # saving output in prepared df
    topicsDf.at[i, 'feedback'] = fb
    topicsDf.at[i, 'topics'] = dic

# saving intermediate results
topicsDf.reset_index(inplace=True, drop=True)
topicsDf.to_csv("./topicsDf.csv", header=True, index=False)


# getting topics and sentiments into separate rows and columns in dataframe
# shell df
topicsSentimentDf = pd.DataFrame(data = [], columns=["Topic", "Sentiment", "FeedbackPart", "Feedback"])

for i in range(0, len(topicsDf)):
    
    fb = topicsDf.loc[i, "feedback"]
    
    suppDf = pd.DataFrame.from_dict(topicsDf.loc[i, "topics"], orient='index').reset_index()
    suppDf.columns = ['Topic', 'Sentiment', 'FeedbackPart']
    suppDf["Feedback"] = fb
    
    topicsSentimentDf = pd.concat([topicsSentimentDf, suppDf], ignore_index=True)
    
# saving intermediate results
topicsSentimentDf.to_csv("./topicsSentimentDf.csv", header=True, index=False)    





# categorization of topics using GPT based on prepared list of topic categories of interest 
topicsCategories = []
for i in range(0, len(topicsSentimentDf)):
    
    print(i)
    
    topic = topicsSentimentDf.iloc[i]["Topic"]
    fb = topicsSentimentDf.iloc[i]["FeedbackPart"]
    
    #time.sleep(3) # slowing down by 3 seconds to avoid RateLimitError
    
    # prompt
    r = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      api_key=myOpenAiApiKey,
      temperature=0, # setting temperature to 0 to produce more focused and conservative responses
      messages=[
            {"role": "system", "content": """You are an experienced behavioral scientist, e.g. I/O psychologist, who can precisely a succinctly identify 
             topics and sentiment present in open-ended feedback from employees on their experience in a company."""},
            {"role": "user", "content": f"""You are provided with a specific topic of feedback provided by employee in employee satisfaction survey. 
             Categorize the topic as belonging into one and only one category from the following list of categories based on the best match between meaning 
             of the topic and the category. Here is the list of the categories you will work with: 'Work-life balance', 'Compensation and benefits', 
             'Job security and stability', 'Opportunities for career growth and development', 'Relationship with supervisors and management', 
             'Recognition and rewards for performance', 'Office and workspace', 'Company culture and values', 'Communication and transparency', 
             'Training and learning opportunities', 'Diversity, equity, and inclusion', 'Health and wellness programs', 'Collaboration and teamwork', 
             'Job satisfaction and engagement', 'Remote and flexible work options', 'Organizational structure and hierarchy', 
             'Performance management and feedback', 'Employee onboarding experience', 'Leadership and direction', 'Workload and stress levels', 
             'Safety and security in the workplace', 'Employee retention and turnover', 'Access to resources and tools', 'Decision-making processes', 
             'Corporate social responsibility and community involvement', 'Innovation and creativity', 'Conflict resolution and problem-solving', 
             'Customer/client satisfaction and relationships', 'Company policies and procedures', 'Employee morale and motivation', 'Other'. 
             Use the relevant part of the feedback to get the context of the topic right. On output I want just the category of the topic, 
             not the topic itself. I don't want you to put there any additional comments or side notes into the output. Don't use commas in the output and 
             don't use 'Category: ' subject before the category. Here is the topic for categorization: {topic}. And here is the relevant part of the feedback 
             for context: {fb}"""}
        ]
    )
    
    d = r["choices"][0]["message"]["content"]
    
    # removing unwanted part of the output that occurs from time to time
    d = d.replace("Category: ", "").replace(".","")
    
    topicsCategories.append(d)

# adding category information into the df
topicsSentimentDf["Category"] = topicsCategories
# saving intermediate results
topicsSentimentDf.to_csv("./topicsSentimentDf.csv", header=True, index=False)  





# alternative categorization of topics using OpenAI embeddings
# function for getting embeddings
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model, api_key=myOpenAiApiKey)['data'][0]['embedding']

# embedding of topics and feedbacks
# spliting large df into smaller chunks to avoid RateLimitError
split_dfs = np.array_split(topicsSentimentDf, 9)

# shell ddf
topicsFbEmbeddingsDf = pd.DataFrame()

for i, split_df in enumerate(split_dfs, start=1):
    print(i)    
    time.sleep(60) # # slowing down by 60 seconds to avoid RateLimitError
    split_df['topicEmbedding'] = split_df.Topic.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    split_df['fbEmbedding'] = split_df.FeedbackPart.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    topicsFbEmbeddingsDf = pd.concat([topicsFbEmbeddingsDf, split_df], ignore_index=True)


# embeddings of topic categories of interest 
categoriesEmbeddingsDf = pd.DataFrame({"Category": ["Work-life balance", "Compensation and benefits", "Job security and stability", "Opportunities for career growth and development", "Relationship with supervisors and management", "Recognition and rewards for performance", "Office and workspace", "Company culture and values", "Communication and transparency", "Training and learning opportunities", "Diversity, equity, and inclusion", "Health and wellness programs", "Collaboration and teamwork", "Job satisfaction and engagement", "Remote and flexible work options", "Organizational structure and hierarchy", "Performance management and feedback", "Employee onboarding experience", "Leadership and direction", "Workload and stress levels", "Safety and security in the workplace", "Employee retention and turnover", "Access to resources and tools", "Decision-making processes", "Corporate social responsibility and community involvement", "Innovation and creativity", "Conflict resolution and problem-solving", "Customer/client satisfaction and relationships", "Company policies and procedures", "Employee morale and motivation"]})
categoriesEmbeddingsDf['categoryEmbedding'] = categoriesEmbeddingsDf.Category.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

# shell df
bestCategories = pd.DataFrame(data = [], columns = ["Category2"])
# using cosine distances between topics, relevant part of feedback and topic categories to determine appropriate topic category  
for i in range(0, len(topicsFbEmbeddingsDf)):
    topicEmb = topicsFbEmbeddingsDf.loc[i, "topicEmbedding"]
    fbEmb = topicsFbEmbeddingsDf.loc[i, "fbEmbedding"]
    
    catDf = pd.DataFrame(data = [], columns = ["Category", "CosDist"])
    
    for j in range(0, len(categoriesEmbeddingsDf)):
        catEmb = categoriesEmbeddingsDf.loc[j, "categoryEmbedding"]
        cosineDistTopic = distance.cosine(topicEmb, catEmb)
        cosineDistFb = distance.cosine(fbEmb, catEmb)
        # putting more weight on contextual information than to identified topics
        avgCosineDist = ((cosineDistTopic*1) + (cosineDistFb*2)) / 3
        
        catDf.at[j, 'Category'] = categoriesEmbeddingsDf.loc[j, "Category"]
        catDf.at[j, 'CosDist'] = avgCosineDist
        
    bestCategories.at[i, 'Category2'] = catDf.loc[catDf["CosDist"] == np.min(catDf["CosDist"]), "Category"].item()
        

topicsSentimentDf['Category2']=bestCategories['Category2']

# saving intermediate results
topicsSentimentDf.to_csv("./topicsSentimentDf.csv", header=True, index=False)  


# adjusting uppercase in sentiment and topic columns
topicsSentimentDf["Sentiment"] = topicsSentimentDf["Sentiment"].apply(lambda x: x.capitalize())
topicsSentimentDf["Topic"] = topicsSentimentDf["Topic"].apply(lambda x: x.title())

# fixing names of topic categories misspelled by GPT using fuzzy matching
choices = ["Work-life balance", "Compensation and benefits", "Job security and stability", "Opportunities for career growth and development", "Relationship with supervisors and management", "Recognition and rewards for performance", "Office and workspace", "Company culture and values", "Communication and transparency", "Training and learning opportunities", "Diversity, equity, and inclusion", "Health and wellness programs", "Collaboration and teamwork", "Job satisfaction and engagement", "Remote and flexible work options", "Organizational structure and hierarchy", "Performance management and feedback", "Employee onboarding experience", "Leadership and direction", "Workload and stress levels", "Safety and security in the workplace", "Employee retention and turnover", "Access to resources and tools", "Decision-making processes", "Corporate social responsibility and community involvement", "Innovation and creativity", "Conflict resolution and problem-solving", "Customer/client satisfaction and relationships", "Company policies and procedures", "Employee morale and motivation"]
fixedCatNames=[]
for i in range(0,len(topicsSentimentDf)):
    query = topicsSentimentDf.loc[i, "Category"]
    best_match = process.extractOne(query, choices)
    best_match[0]
    fixedCatNames.append(best_match[0])

topicsSentimentDf["CategoryFixed"] = fixedCatNames

# saving intermediate results
topicsSentimentDf.to_csv("./topicsSentimentDf.csv", header=True, index=False)  





# getting 2D cooordinates of topics in semantic spaces using topic embeddings and tSNA (later used for visualization of topics within individual topic categories)
# spliting large df into smaller chunks to avoid RateLimitError
split_dfs = np.array_split(topicsSentimentDf, 9)

topicsEmbeddingsDf = pd.DataFrame()

for i, split_df in enumerate(split_dfs, start=1):
    print(i)    
    time.sleep(60) # slowing down by 60 seconds
    split_df['topicEmbedding'] = split_df.Topic.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    topicsEmbeddingsDf = pd.concat([topicsEmbeddingsDf, split_df], ignore_index=True)
    
matrix=topicsFbEmbeddingsDf.topicEmbedding.apply(np.array).to_list()

# Create a t-SNE model and transform the data
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

topicsSentimentDf["x"] = [x for x,y in vis_dims]
topicsSentimentDf["y"] = [y for x,y in vis_dims]

# saving final results at the level of individual topics
topicsSentimentDf.to_csv("./topicsSentimentDf.csv", header=True, index=False)  





# computing frequencies of topic categories by sentiment 
# identifying unique topic categories
uniqueTopics = pd.DataFrame(data = topicsSentimentDf["CategoryFixed"].unique(), columns=["CategoryFixed"])

# commputing proportion of various sentiments for individual topic categories  
sentimentByTopic = topicsSentimentDf.groupby('CategoryFixed')['Sentiment'].value_counts(normalize=False).rename('Frequency').reset_index()

# computing all combinations of topic categories and sentiments
cat_var1 = list(topicsSentimentDf["CategoryFixed"].unique())
cat_var2 = list(topicsSentimentDf["Sentiment"].unique())
allCombinations = list(itertools.product(cat_var1, cat_var2))
allCombinationsDf = pd.DataFrame(allCombinations, columns=['CategoryFixed', 'Sentiment'])

# merging all combinations with observed frequencies and replacing NAs with zero values
sentimentInfo = pd.merge(allCombinationsDf, sentimentByTopic, left_on=['CategoryFixed', 'Sentiment'], right_on=['CategoryFixed', 'Sentiment'], how='left')
sentimentInfo["Frequency"] = sentimentInfo["Frequency"].fillna(0)
# computing relative frequencies
total_frequency_per_topic = sentimentInfo.groupby('CategoryFixed')['Frequency'].sum().reset_index()
sentimentInfo = sentimentInfo.merge(total_frequency_per_topic, on='CategoryFixed', suffixes=('', '_total'))
sentimentInfo['Relative_Frequency'] = sentimentInfo['Frequency'] / sentimentInfo['Frequency_total']

# saving final results at the level of topic categories
sentimentInfo.to_csv("./sentimentInfo.csv", header=True, index=False) 





