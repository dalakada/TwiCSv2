import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
tweets=pd.read_csv("final_incomplete_tweets.csv",sep =',')

#tweets=tweets[['inverted_loss','current_minus_entry','entry_batch']]
#tweets=tweets[tweets.entry_batch==2]
print(tweets)


tweets_org=tweets[['inverted_loss','current_minus_entry']].groupby(["current_minus_entry"]).sum()

tweets_org.plot()
print(tweets_org)



tweets_org.to_csv("plot_data_by_grouped.csv", sep=',', encoding='utf-8')

# tweets[['inverted_loss','ratio_entry_vs_current']].groupby(["ratio_entry_vs_current"]).mean()#.plot(ylim=0)
# tweets[['inverted_loss','current_minus_entry']].groupby(["current_minus_entry"]).mean()#.plot(ylim=0)
plt.show()


# f1 = interp1d(tweets.index, tweets['inverted_loss'],kind='cubic')


# df2 = pd.DataFrame()
# new_index = np.arange(0.002193,0.073529)
# df2['Weight_A'] = f1(new_index)

# df2.index = new_index

# ax2 = df2.plot.line()
# ax2.set_title('After interpolation')
# ax2.set_xlabel("year")
# ax2.set_ylabel("weight")




# import matplotlib.pyplot as plt
# z_scores=[-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# #z_scores=[-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# #f1_scores=[0.9296226919989296,0.9014011416709912,0.8748426089146312,0.8804833201996323,0.8785515320334262,0.901098901098901,0.8721330275229358,0.859199539303196,0.8498126261170367,0.8393991912189486,0.8272463768115943,0.8143189755529685,0.8110144927536233,0.8077367205542726,0.80448533640023,0.7897495631916133]
# f1_scores=[0.9294117647,0.9294117647,0.9294117647,0.9295624333,0.9294117647,0.9294117647,0.9294117647,0.9295624333,0.9254966887,0.9042979943,0.9006354708,0.8895382817,0.8800471559,0.8740740741,0.8666865494,0.8578298768,0.8481357987,0.8481357987,0.8481357987,0.8481357987,0.8331797235]
# precision=[0.9369272237,0.9369272237,0.9369272237,0.935050993,0.9369272237,0.9369272237,0.9369272237,0.935050993,0.964347326,0.9831775701,0.9885859226,0.9902407287,0.9900530504,0.9899328859,0.9897820163,0.9895977809,0.9893917963,0.9893917963,0.9893917963,0.9893917963,0.9897810219]
# recall=[0.9220159151,0.9220159151,0.9220159151,0.924137931,0.9220159151,0.9220159151,0.9220159151,0.924137931,0.8896551724,0.8371352785,0.8270557029,0.8074270557,0.7920424403,0.7824933687,0.7708222812,0.7570291777,0.7421750663,0.7421750663,0.7421750663,0.7421750663,0.7193633952]

# #plt.scatter(z_scores, f1_scores, alpha=0.5)
# plt.xticks(z_scores)
# plt.plot(z_scores, f1_scores, alpha=0.5)
# #plt.plot(z_scores, precision, alpha=0.5)
# #plt.plot(z_scores, recall, alpha=0.5)




# plt.xlabel('Z_score value')
# plt.ylabel('f1_score')
# #plt.ylabel('precision')
# #plt.ylabel('recall')
# plt.grid(True)
# plt.savefig("z_score_VS_f1_score_Mention.png")
# #plt.savefig("z_score_VS_precision_Mention.png")
# #plt.savefig("z_score_VS_recall_Mention.png")
# plt.show()