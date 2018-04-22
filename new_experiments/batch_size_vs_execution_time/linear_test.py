import numpy as np
import matplotlib.pyplot as plt

x = [1,2,3,4]
y = [3,5,7,10] # 10, not 9, so the fit isn't perfect
seen_tweets=range(100,38000,500)

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y

print(fit_fn)
for seen_tweet in seen_tweets:
	print(fit_fn(seen_tweet))
# plt.plot(x,y, 'yo', x, fit_fn(x), '--k')
plt.plot( x, fit_fn(x), '--k')

# plt.xlim(0, 5)
# plt.ylim(0, 12)
plt.show()