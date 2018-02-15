whole_level=[
[
1082.7956538044,
1050.6432401884,
1053.7217456599,
1012.5515266169,
1008.6493123934,
989.9630010678,
952.6808602644,
913.4404472567,
877.5589883683,
850.7370345206,
838.5608943759
],
[83.5613470219,
86.1817237598,
91.4628130584,
89.9526238481,
86.8751099374,
92.2544695707,
95.2050934217,
97.7485665828,
99.7710068994,
101.1100325883,
101.5450283182

],
[
282.5355908803,
286.9383684093,
304.265311919,
304.7142461953,
307.9814362368,
307.7235153447,
303.0300608151,
298.1720040698,
293.5781982817,
290.6551030755,
290.0621321868

],
[0.7847898245,
0.7890123589,
0.7931519113,
0.7704950433,
0.7660793922,
0.771851223,
0.7642034553,
0.7541908589,
0.7508384564,
0.7468438025,
0.7450788343
]
]


no_tweets=[173400,
177084,
177350,
155079,
166533,
179215,
159484,
150637,
161413,
157516,
55394]

time=[]
time.append([])
time.append([])
time.append([])
time.append([])

for i in range(4):
	for j in range(len(no_tweets)):
		# print(no_tweets[j],whole_level[i][j])
		# print((no_tweets[j])/(whole_level[i][j]))
		time[i].append((no_tweets[j])/(whole_level[i][j]))




no_tweets=[
100000,
100000,
100000,
100000,
100000,
100000,
100000,
100000,
100000,
100000,
35000]


throughput_tweet_level=[]
throughput_tweet_level.append([])
throughput_tweet_level.append([])
throughput_tweet_level.append([])
throughput_tweet_level.append([])

for i in range(4):
	for j in range(len(no_tweets)):
		# print(no_tweets[j],time[i][j])
		# print((no_tweets[j])/(whole_level[i][j]))
		throughput_tweet_level[i].append((no_tweets[j])/(time[i][j]))

for i in throughput_tweet_level:
	print(i)
# print(throughput_tweet_level)