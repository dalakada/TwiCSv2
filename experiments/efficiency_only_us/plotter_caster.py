whole_level=[[[12.594131231307983, 26.213688611984253, 40.03421425819397, 54.02845120429993, 69.011709690094, 84.68057155609131, 85.55523633956909], [0.8322916666666667, 0.80592105263157898, 0.79098591549295771, 0.77055271713887596, 0.76444111027756945, 0.764449968924798, 0.75695187165775402], [29800, 60239, 90272, 120023, 149853, 179926, 180547], 0, [799, 1225, 1404, 1659, 2038, 2460, 2831]], 
[[12.893548011779785, 28.131826400756836, 43.59176468849182, 59.52418494224548, 76.04927921295166, 93.40274500846863, 95.7199935913086], [0.8322916666666667, 0.80855263157894741, 0.79380281690140841, 0.77241058987459355, 0.76744186046511631, 0.76693598508390304, 0.75909090909090904], [29800, 60239, 90272, 120023, 149853, 179926, 180547], 1, [799, 1229, 1409, 1663, 2046, 2468, 2839]], 
[[13.003226041793823, 28.22015142440796, 44.698610067367554, 61.80603241920471, 79.74235153198242, 98.56109666824341, 102.30935454368591], [0.8322916666666667, 0.80855263157894741, 0.79380281690140841, 0.77241058987459355, 0.76744186046511631, 0.76693598508390304, 0.75909090909090904], [29800, 60239, 90272, 120023, 149853, 179926, 180547], 2, [799, 1229, 1409, 1663, 2046, 2468, 2839]], 
[[12.982017278671265, 28.153103828430176, 44.391281604766846, 62.56139659881592, 81.83243417739868, 101.61762404441833, 106.59552574157715], [0.8322916666666667, 0.80855263157894741, 0.79380281690140841, 0.77194612169066423, 0.76669167291822959, 0.76631448104412681, 0.75855614973262031], [29800, 60239, 90272, 120023, 149853, 179926, 180547], 3, [799, 1229, 1409, 1662, 2044, 2466, 2837]], 
[[13.05547547340393, 28.34896731376648, 44.7604763507843, 62.25438189506531, 83.71740913391113, 104.39059710502625, 110.9569354057312], [0.8322916666666667, 0.80855263157894741, 0.79380281690140841, 0.77194612169066423, 0.76669167291822959, 0.76631448104412681, 0.75855614973262031], [29800, 60239, 90272, 120023, 149853, 179926, 180547], 4, [799, 1229, 1409, 1662, 2044, 2466, 2837]], 
[[13.09713101387024, 28.048423051834106, 44.49276924133301, 62.539196729660034, 83.45518398284912, 108.06880307197571, 116.93658375740051], [0.8322916666666667, 0.80855263157894741, 0.79380281690140841, 0.77194612169066423, 0.76669167291822959, 0.76662523306401487, 0.75882352941176467], [29800, 60239, 90272, 120023, 149853, 179926, 180547], 5, [799, 1229, 1409, 1662, 2044, 2467, 2838]], 
[[12.869799375534058, 28.773030519485474, 44.05394697189331, 62.90510511398315, 83.03927302360535, 108.66561985015869, 116.51515364646912], [0.8322916666666667, 0.80855263157894741, 0.79380281690140841, 0.77194612169066423, 0.76669167291822959, 0.76662523306401487, 0.75882352941176467], [29800, 60239, 90272, 120023, 149853, 179926, 180547], 6, [799, 1229, 1409, 1662, 2044, 2467, 2838]]]

without_eviction_id=len(whole_level)-1
without_eviction=whole_level[without_eviction_id]
import math
import matplotlib.ticker as ticker
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style

for i in whole_level:
	print(i[0])
	for idx,val in enumerate(i[0]):
		casted_int=math.floor(val)
		i[0][idx]=casted_int


for i in whole_level:
	for idx,val in enumerate(i[0]):
		print(val)

eviction_parameter_recorder=range(7)

# timing=[[0.7756309509277344, 1.404196949005127, 2.1200640201568604, 2.8386363983154297, 3.569007158279419],
# [0.7308433055877686, 1.4264043292999268, 2.184626636505127, 3.0043627166748047, 3.820970058441162],
# [0.7488808631896973, 1.4265043292999268, 2.204626636505127, 3.1043627166748047, 3.923989772796631],
# [0.7770745754241943, 1.4265043292999268, 2.204626636505127, 3.1043627166748047, 3.943989772796631],
# [0.7539031505584717, 1.4265043292999268, 2.204626636505127, 3.1043627166748047, 3.963989772796631]]

# timing_id=len(timing)-1
# timing_max=timing[timing_id]

# timing_sliced=timing[:-1]

p1_holder=[]
p2_holder=[]

# print("Without eviction time : ",without_eviction[0])
for idx,level in enumerate(whole_level[:-1]):
    # print(level[0])

    # print(level)
    # accuracy=level[1]
    p1_divided=[]
    
    for i in range(len(level[1])):
        p1_divided.append(level[1][i]/without_eviction[1][i])
        # print(p1_divided)

    # tweets_been_processed_list=level[2]
    # p1_divided=sorted(p1_divided)
    p2=[]
    # for i in range(len(level[0])):
    #     p2.append(without_eviction[0][i]-level[0][i])
    for i in range(len(level[0])):
        # p2.append(timing_max[i]-timing_sliced[idx][i])
        p2.append(level[0][i]-without_eviction[0][i])


    tweets_been_proccessed=level[2]

    p1xp2=[]

    # p2=sorted(p2)

    for i in range(len(p1_divided)):
        p1xp2.append(p2[i]*p1_divided[i])

    # print('P1 : ',p1_divided,'Recall without :',without_eviction[1])

    # print('Recall : ',level[1],'Recall without :',without_eviction[1])

    # print('TP: ' ,level[4],'Without ', without_eviction[4])

    p1_holder.append(p1_divided)
    p2_holder.append(p2)

p1_holder_tranpsosed=list(map(list, zip(*p1_holder)))
p2_holder_tranpsosed=list(map(list, zip(*p2_holder)))

print("***************************************************************")
for i in p2_holder:
    print(i)
for i in p1_divided:
    print(i)
# print(eviction_parameter_recorder)
# for i in p1_holder:
#     print(i)
eviction_parameter_recorder=eviction_parameter_recorder[:-1]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for idx,level in enumerate(p1_holder_tranpsosed[1:]):
    p1=level
    p2=p2_holder_tranpsosed[idx+1]

    # fit = np.polyfit(eviction_parameter_recorder,p1,1)
    # fit_fn1 = np.poly1d(fit) 

    # fit = np.polyfit(eviction_parameter_recorder,p2,1)
    # fit_fn2 = np.poly1d(fit) 

    # h = lambda x: fit_fn1(x)- fit_fn2(x)

    # x = np.arange(-500, 200, 1)


    # x_int = scipy.optimize.fsolve(h, 0)
    # y_int = fit_fn1 (x_int)

    # print('************************************')
    # print(tweets_been_proccessed[idx])
    # print(x_int, y_int)
    # print(fit_fn1,fit_fn2)
    # print('************************************')

    ax1.plot(eviction_parameter_recorder, p1,label=tweets_been_proccessed[idx+1])
    ax1.text(eviction_parameter_recorder[0], p1[0], 'p1')
    ax2.plot(eviction_parameter_recorder, p2,label=tweets_been_proccessed[idx+1])
    ax2.text(eviction_parameter_recorder[0], p2[0], 'p2')

    ###plt.plot(x, f(x), zorder=1)
    ###plt.plot(x, g(x), zorder=1)

    # idx = np.argwhere(np.isclose(fit_fn1, fit_fn2, atol=10)).reshape(-1)




    # ax3.scatter(x_int, y_int, marker='x')

    tick_spacing = 1
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax1.set_xlabel('Eviction Parameter ')
    ax1.set_ylabel('p1')
    #ax2.set_ylabel('p2')

    ## AFTER ####
    # plt.plot( tweets_been_proccessed,p1xp2,marker='o' , label=eviction_parameter_recorder[idx],alpha=0.5)



    plt.grid(True)
    plt.legend(loc='upper left')
    # plt.savefig("Execution-Time-vs-Batch-Size.png")

plt.show()

fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
for idx,level in enumerate(p1_holder_tranpsosed[1:]):
    p1=level
    p2=p2_holder_tranpsosed[idx+1]

    # fit = np.polyfit(eviction_parameter_recorder,p1,1)
    # fit_fn1 = np.poly1d(fit) 

    # fit = np.polyfit(eviction_parameter_recorder,p2,1)
    # fit_fn2 = np.poly1d(fit) 

    # h = lambda x: fit_fn1(x)- fit_fn2(x)

    # x = np.arange(-500, 200, 1)


    # x_int = scipy.optimize.fsolve(h, 0)
    # y_int = fit_fn1 (x_int)

    # print('************************************')
    # print(tweets_been_proccessed[idx])
    # print(x_int, y_int)
    # print(fit_fn1,fit_fn2)
    # print('************************************')

    ax1.plot(eviction_parameter_recorder, p1,label=tweets_been_proccessed[idx+1])
    #ax1.text(eviction_parameter_recorder[0], p1[0], tweets_been_proccessed[idx])
    #ax2.plot(eviction_parameter_recorder, p2,label=tweets_been_proccessed[idx+1])
    #ax2.text(eviction_parameter_recorder[0], p2[0], tweets_been_proccessed[idx])

    ###plt.plot(x, f(x), zorder=1)
    ###plt.plot(x, g(x), zorder=1)

    # idx = np.argwhere(np.isclose(fit_fn1, fit_fn2, atol=10)).reshape(-1)




    # ax3.scatter(x_int, y_int, marker='x')

    tick_spacing = 1
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax1.set_xlabel('Eviction Parameter ')
    ax1.set_ylabel('p1')
    #ax2.set_ylabel('p2')

    ## AFTER ####
    # plt.plot( tweets_been_proccessed,p1xp2,marker='o' , label=eviction_parameter_recorder[idx],alpha=0.5)



    plt.grid(True)
    plt.legend(loc='upper left')
    # plt.savefig("Execution-Time-vs-Batch-Size.png")

plt.show()

fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
for idx,level in enumerate(p1_holder_tranpsosed[1:]):
    p1=level
    p2=p2_holder_tranpsosed[idx+1]

    # fit = np.polyfit(eviction_parameter_recorder,p1,1)
    # fit_fn1 = np.poly1d(fit) 

    # fit = np.polyfit(eviction_parameter_recorder,p2,1)
    # fit_fn2 = np.poly1d(fit) 

    # h = lambda x: fit_fn1(x)- fit_fn2(x)

    # x = np.arange(-500, 200, 1)


    # x_int = scipy.optimize.fsolve(h, 0)
    # y_int = fit_fn1 (x_int)

    # print('************************************')
    # print(tweets_been_proccessed[idx])
    # print(x_int, y_int)
    # print(fit_fn1,fit_fn2)
    # print('************************************')

    ax1.plot(eviction_parameter_recorder, p2,label=tweets_been_proccessed[idx+1])
    #ax1.text(eviction_parameter_recorder[0], p1[0], tweets_been_proccessed[idx])
    #ax2.plot(eviction_parameter_recorder, p2,label=tweets_been_proccessed[idx+1])
    #ax2.text(eviction_parameter_recorder[0], p2[0], tweets_been_proccessed[idx])

    ###plt.plot(x, f(x), zorder=1)
    ###plt.plot(x, g(x), zorder=1)

    # idx = np.argwhere(np.isclose(fit_fn1, fit_fn2, atol=10)).reshape(-1)




    # ax3.scatter(x_int, y_int, marker='x')



    ax1.set_xlabel('Eviction Parameter ')
    ax1.set_ylabel('p2')


    tick_spacing = 1
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax2.set_ylabel('p2')

    ## AFTER ####
    # plt.plot( tweets_been_proccessed,p1xp2,marker='o' , label=eviction_parameter_recorder[idx],alpha=0.5)



    plt.grid(True)
    plt.legend(loc='upper left')
    # plt.savefig("Execution-Time-vs-Batch-Size.png")


plt.show()