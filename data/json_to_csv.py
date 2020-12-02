import json
import pandas as pd

f = open("tweets_virus_427K.json", "r")
data= json.loads(f.read())
count=0
df_columns=('ID','HashTags','TweetText','mentions_other','URLs','User')
df_holder=[]
for data_dict in data:
	if(count<2000):
		# print(data_dict)
		d={'ID':count,'HashTags':data_dict['Topic'],'TweetText':data_dict['Content'],'mentions_other':'','URLs':'','User':data_dict['Author']}
	# for key, value in data_dict.items():
	# 	print(key, value)
		count+=1
		df_holder.append(d)
	else:
		break

df_out = pd.DataFrame(df_holder,columns=df_columns)
print(len(df_out))

print(df_out.head())

df_out.to_csv("/Users/satadisha/Documents/GitHub/covid_2K.csv", sep=',', encoding='utf-8',index=False)
