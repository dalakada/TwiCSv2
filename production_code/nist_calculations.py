
###NEURONER

sentences=[762871,806492,839385,848070,843367]

# total_time=[]

# offset_sentence= 0
# offset_time= 0
# offset_mentions=0

# time=[ #non-cumulative
# [1009.1259067058563,1285.823305606842,1018.5751554965973,1142.7116467952728,1132.2352578639984,626.733500957489],
# [1093.8325762748718,1194.4339590072632,1067.8544609546661,1199.3962926864624,1180.5313053131104,1185.202190876007],
# [1190.4452047348022,1151.3296461105347,1164.5595812797546,1205.135823249817,1012.250725030899,1183.5678446292877,156.12309503555298],
# [1256.5463004112244,1039.9185893535614,1020.3258068561554,1031.3249342441559,1096.526581287384,1069.5119030475616,147.96642780303955],
# [1322.2521028518677,1247.3375043869019,1193.0790388584137,1295.113252878189,1290.1293590068817,1200.6789112091064,168.48697328567505]
# ]

# mentions_discovered=[
# [77432,77602,77455,75391,75783,41997],
# [77615,79854,78701,79333,78221,68627],
# [77391,79411,77926,77698,79020,78131,7850],
# [82398,82101,81474,79721,80667,81091,8647],
# [79924,82585,79691,79661,80191,81468,7118]
# ]

# offset_sentence= 4100185
# offset_time= 34579.0651640892
# offset_mentions= 2352175

# sentences=[847398,781836,777249,827177,847114]

# time=[ #non-cumulative
# [1217.5202195644379,1007.3874917030334,1001.1248931884766,1008.9169096946716,1027.9380993843079,1004.3718683719635,155.7433478832245],
# [1202.212443113327,1088.6094551086426,1538.2711884975433,1249.7206416130066,1241.397672176361,1023.3263018131256],
# [1274.395179271698,1154.0553591251373,1053.6796958446503,1023.6030249595642,1012.1207664012909,666.6133892536163],
# [1055.9579064846039,1003.416140794754,1019.9612421989441,1055.4675388336182,1051.006579875946,1059.4553892612457],
# [1030.343843460083,1002.1000475883484,1006.3465394973755,993.879606962204,998.7429385185242,992.7638220787048,141.22010731697083]
# ]

# mentions_discovered=[
# [82901,81340,81768,81180,81960,84025,9371],
# [75987,75746,76120,75783,76116,54011],
# [78293,78053,78574,76689,78613,51479],
# [80585,81121,81581,81998,80999,77241],
# [81799,81361,79424,78200,81485,80879,8553]
# ]

# offset_sentence= 8180959
# offset_time= 66940.7348139286
# offset_mentions= 4705410

# sentences=[858328,882096,862235,796066,831545]

# time=[
# [990.1502664089203,1005.0563771724701,1011.8487436771393,990.053472995758,1006.3562653064728,996.3381795883179,226.33327460289001],
# [994.2255792617798,1004.5873188972473,994.6720976829529,1001.6219711303711,986.1801052093506,987.1630928516388,414.48208689689636],
# [1002.6363258361816,994.0639300346375,1001.9913067817688,998.724778175354,994.8401489257812,1000.1362857818604,273.48842906951904],
# [1443.829775094986,1126.884622335434,989.1520884037018,1021.6184844970703,1128.4812779426575,825.481048822403],
# [1065.7767972946167,1001.3893098831177,998.9240529537201,986.0072357654572,994.6900112628937,1079.5028460025787,109.69876003265381]
# ]

# mentions_discovered=[
# [80201,79210,81195,79813,79858,83566,15481],
# [80693,81541,81462,82747,82357,82436,31444],
# [80539,82973,82182,81282,81965,80985,19974],
# [78342,77197,79033,77533,79519,62142],
# [79448,78347,75964,78785,77546,78272,5735]
# ]

# offset_sentence=  12411229
# offset_time=  98587.1211605072
# offset_mentions=  7165177


# sentences=[881781,851876]

# time=[
# [1074.2684674263,1009.7006583213806,988.2241508960724,962.3962466716766,996.3231666088104,995.5143449306488,418.0636143684387],
# [1112.8149592876434,1026.0831224918365,1090.967236995697,1032.3092579841614,1005.1012227535248,1027.481891155243,15813]
# ]

# mentions_discovered=[
# [79911,80761,80228,81071,79455,80514,30586],
# [81118,80822,81281,78684,79849,80957,15813]
# ]


# sentence_cumulative=[
# offset_sentence +sentences[0],
# offset_sentence +sentences[0]+sentences[1],
# offset_sentence +sentences[0]+sentences[1]+sentences[2],
# offset_sentence +sentences[0]+sentences[1]+sentences[2]+sentences[3],
# offset_sentence +sentences[0]+sentences[1]+sentences[2]+sentences[3]+sentences[4]
# ]

# time_cumulative=[
# offset_time+sum(time[0]),
# offset_time+sum(time[0])+sum(time[1]),
# offset_time+sum(time[0])+sum(time[1])+sum(time[2]),
# offset_time+sum(time[0])+sum(time[1])+sum(time[2])+sum(time[3]),
# offset_time+sum(time[0])+sum(time[1])+sum(time[2])+sum(time[3])+sum(time[4])
# ]

# mentions_cumulative=[
# offset_mentions +sum(mentions_discovered[0]),
# offset_mentions +sum(mentions_discovered[0])+sum(mentions_discovered[1]),
# offset_mentions +sum(mentions_discovered[0])+sum(mentions_discovered[1])+sum(mentions_discovered[2]),
# offset_mentions +sum(mentions_discovered[0])+sum(mentions_discovered[1])+sum(mentions_discovered[2])+sum(mentions_discovered[3]),
# offset_mentions +sum(mentions_discovered[0])+sum(mentions_discovered[1])+sum(mentions_discovered[2])+sum(mentions_discovered[3])+sum(mentions_discovered[4])
# ]

# print(sentence_cumulative)
# print(time_cumulative)
# print(mentions_cumulative)

# print('offset_sentence= ',sentence_cumulative[-1])
# print('offset_time= ',time_cumulative[-1])
# print('offset_mentions= ',mentions_cumulative[-1])

# print([sentence_cumulative[index]/time_cumulative[index] for index in range(len(time_cumulative))])
# print([mentions_cumulative[index]/time_cumulative[index] for index in range(len(time_cumulative))])

# ###TwiCS
sentences=[138036, 274297, 412718, 549344, 686568, 762871, 900451, 1038802, 1176287, 1313181, 1451357, 1569363, 1707422, 1845591, 1982766, 2119352, 2257383, 2394990, 2408748, 2548347, 2687431, 2826866, 2964608, 3102680, 3241621, 3256818, 3394958, 3533654, 3671134, 3809902, 3948963, 4088008, 4100185, 4238748, 4377097, 4516374, 4654505, 4791787, 4931282, 4947583, 5084352, 5220726, 5356667, 5492145, 5628773, 5729419, 5867018, 6003981, 6141023, 6277613, 6415740, 6506668, 6644476, 6783636, 6922024, 7060966, 7200279, 7333845, 7472954, 7611287, 7749843, 7887970, 8026453, 8165635, 8180959, 8319004, 8457641, 8596275, 8735069, 8872382, 9011905, 9039287, 9176908, 9314998, 9453225, 9591194, 9730019, 9868423, 9921383, 10058599, 10197698, 10335474, 10473112, 10611717, 10749311, 10783618, 10920779, 11057471, 11195622, 11331793, 11469279, 11579684, 11717194, 11854190, 11989947, 12127820, 12263760, 12401568, 12411229, 12548343, 12686994, 12825242, 12964923, 13101996, 13240138, 13293010, 13430569, 13568803, 13706153, 13843457, 13981160, 14118789, 14144886]

total_time=[63.633262157440186, 133.60460448265076, 208.37246656417847, 283.2373857498169, 360.3798930644989, 406.9645154476166, 487.0215950012207, 569.74169921875, 652.6389410495758, 735.9593636989594, 818.8442306518555, 893.7712199687958, 977.284126996994, 1063.297026872635, 1148.878098487854, 1234.9919383525848, 1324.0760939121246, 1410.6011805534363, 1434.1756703853607, 1522.4852035045624, 1612.4333477020264, 1703.164059638977, 1793.630942583084, 1885.0494968891144, 1977.203504562378, 2003.760345697403, 2093.9844839572906, 2185.666470527649, 2278.0048592090607, 2372.6961257457733, 2467.0239470005035, 2563.4611444473267, 2591.3795037269592, 2685.509876728058, 2781.224595308304, 2876.633551120758, 2973.4034910202026, 3070.054701089859, 3168.3340706825256, 3201.4512329101562, 3294.2325665950775, 3387.123699426651, 3482.528557538986, 3577.6443161964417, 3678.8757066726685, 3764.931407213211, 3863.8311200141907, 3962.2475645542145, 4061.8343210220337, 4161.8856563568115, 4263.850114822388, 4342.568738222122, 4445.902293920517, 4550.345546007156, 4656.288374662399, 4764.014865875244, 4871.006106138229, 4975.329767227173, 5083.7518537044525, 5193.10072183609, 5301.788656949997, 5410.176788806915, 5520.826105117798, 5631.993359565735, 5676.872709035873, 5803.360855817795, 5928.028892755508, 6040.702213048935, 6153.918267250061, 6267.137784481049, 6383.393240451813, 6437.095897674561, 6550.158462762833, 6663.150055885315, 6778.145142793655, 6893.203466653824, 7009.952384233475, 7127.248076438904, 7197.301256895065, 7313.213763952255, 7431.103375196457, 7564.869182348251, 7684.440948009491, 7805.287962913513, 7928.370731592178, 7992.452976465225, 8106.258823394775, 8224.69791507721, 8345.833230733871, 8465.532633304596, 8587.553409814835, 8692.105822324753, 8809.39468407631, 8926.834284067154, 9044.152545452118, 9163.173377990723, 9281.896821022034, 9402.045582056046, 9458.07046365738, 9578.456564188004, 9700.939926862717, 9824.576881170273, 9948.350461244583, 10070.950147390366, 10195.175110816956, 10276.858102083206, 10401.153700590134, 10554.12793302536, 10704.943866729736, 10832.422648906708, 10961.356420993805, 11090.783314228058, 11161.832092523575]

mentions_discovered=[
[173870,208923,217763,225539,221287],
[223060,177649,183036,221603,224103],
[228594,230816,223565,189529,198683],
[,]
]

calculation_arr_sentence=[sentences[4],sentences[9],sentences[14],sentences[16]]

calculation_arr_time= [total_time[4],total_time[9],total_time[14],total_time[16]]

print(calculation_arr_sentence)
print(calculation_arr_time)

print([calculation_arr_sentence[index]/calculation_arr_time[index] for index in range(len(calculation_arr_time))])

# ###Twitter NLP
# sentences=[138036, 274297, 412718, 549344, 686568, 762871]

# total_time=[]

# mentions_discovered=[]