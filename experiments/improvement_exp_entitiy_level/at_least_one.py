
from nltk.tokenize import word_tokenize
import string
import pandas as pd 
import itertools
import copy
tweets=pd.read_csv("tweets_1million_for_others.csv",sep =',')

#input comes all lowercase
def makeupperdynamic(input):
    output=[]
    # tokenized_words =word_tokenize(input)
    tokenized_words=input.split()
    # print(tokenized_words)
    number_of_permutations=list(itertools.product([0,1], repeat=len(tokenized_words)))


    for possibility in number_of_permutations:
        output.append(copy.deepcopy(tokenized_words))
    # print(output)

    new=[]
    for idx,possibility in enumerate(number_of_permutations):
        # print(idx)
        for i in range(len(tokenized_words)):
            on_or_off=possibility[i]
            # print(on_or_off)
            if(on_or_off==1):
                # print(i)
                # mert=tokenized_words[i][0].upper()
                # output.append(mert)
                list1=list(output[idx][i])
                list1[0]=list1[0].upper()
                str1 = ''.join(list1)        
                # print(str1)
                output[idx][i]=str1
                # print(output[idx])
            else:
                output[idx][i]

        new.append(copy.deepcopy(output[idx]))
    # print(new)
            
    new_flatten=[]
    for i in new:
        str1 = ' '.join(i)
        new_flatten.append(str1)
    # print(new_flatten)

    new_flatten = new_flatten[:-1]
    # print(tokenized_words)
    # output=tokenized_words
    return new_flatten

# output=makeupperdynamic("ufc light heavyweight champion daniel cormier")
# print(output)





def intersect(a, b):
     return list(set(a) & set(b))

# print(my_list)
my_list2=['nialler', 'raccoon gosha', 'australian', 'cyrus', 'jackson wink', 'miley', 'hybrid theory', 'hansel robles', 'jolene', 'reichert', 'bill', 'malta', 'boa constrictor', 'uber', 'scarface', 'aldo', 'instagram', 'president', 'ariel winter', 'civil rights movement', 'serbia', 'chechnya', 'måns', 'niall cantando', 'dnc', 'whites', 'weiner', 'liam payne', 'taput', 'chipmunk', 'carter', 'russian intelligence', 'hpn', 'manel navarro', 'justin timberlake', 'sam', 'australia', 'floridans', 'chameleon', 'ruth', 'james bond', 'congressional house', 'genocide', 'siri', 'centrist', 'krzysztof jotko', 'alex caceres', 'm5', 'ptsd', 'russia oil', 'ann courter', 'german', "o'care", 'azerbaijan', 'do joe jonas', 'google', 'senators', 'henley', 'andy biggs', 'rose garden', 'isil', 'rosneft', 'kyiv', 'spurs', 'bulgaria', 'stoke ohio high school', 'white supremacists', 'rus', 'maia. linking park', 'karma', 'jason blossom', 'superior spa', 'joevin jones', 'cowboys', 'nunez', 'wv', 'red pill', 'apocalypse', 'portugal', 'baby elephant', 'numb', 'germany', 'jimmy fallon', 'jeff sessions', 'pro-life', 'socialist', 'bjj', 'justice', 'mma', 'wango tango', 'romania', 'house oversight chair', 'mccanns', 'petra', 'sjw', 'coloradans', 'yair rodriguez', 'moldova', 'sowell', 'machine gun kelly', 'ukrainne', 'middle east', 'simpsons', 'on the loose', 'luhan', 'imri', 'russians', 'bad things', 'rep', 'eminem', 'katy perry', 'alvarez', 'doj', 'frankie edwards', 'white privilege', 'australians', 'ic', 'acting atty general', 'star wars', 'hosen schlange', 'mike', 'copa del rey', 'norway', 'christ', 'house republicans', 'united airlines', 'potus', 'lisbon', 'canadian', 'fitch', 'south america', 'greece', 'moto x4', 'fisa', 'ohio', 'gen. flynn', 'mike dean', 'polonium tea syndrome', 'costarica', 'bloods', 'dana', 'conor mcgregor', 'thatcher', 'latino', 'frankie edgar', 'jj', 'roe v. wade', 'japanese high school', 'nazis', 'us senate', 'tj', 'jesus christ', 'mlk jr', 'e.t. bon appétit', 'cankles', 'baseball', 'islam', 'american', 'new york times', "it's always sunny in philadelphia", 'ces', 'euros', 'arizona', 'scary clown cars', 'maxine waters', 'chris pratt', 'google project tango', 'fisa/fisc', 'penicillin', 'brazilian', 'tariq', 'switzerland', 'healthcare bill', 'timur m', 'liam e niall', 'juno mission', 'liam viendo', 'vitalii sediuk', 'house', 'dark horse', 'affirmative action', 'michelle', 'liberal', 'death panel', "we can't stop", 'david branch', 'senate', 'demian maia', 'bruce buffer', 'stoke greyson', 'amanda', 'house health bill', 'kkk', 'also-holloway', 'sweden', 'richie', 'maryland', 'obama presidency', 'insulin', 'j.j. watt', 'edgar', 'senate gop', 'iran deal', 'womd', 'america', 'rokita', 'the korean peninsula', 'health care act', 'bidet', 'brad pitt', 'libnazi', 'passover', 'holocaust centers', 'christian', 'pepsi', 'payno', 'soviet russia', 'justice department', 'white supremacist', 'dems', 'gop', 'spicy', 'texas', 'albert einstein', 'july', 'yair', 'jordan fisher', 'dem', 'arn anderson', 'champions league', 'christianity', 'chuck', 'asia', 'lenovo', 'gender studies', 'brazilian fighters', 'junior', 'finnish', 'fourth reich', 'mussolini', 'earth', 'niam', 'enforcer', 'united kingdom', 'lucie jones', 'violet', 'asha', 'haiku', 'adam', 'slow hands', 'scaarface', 'health plan', 'maroon 5', "let's go crazy", 'jd', 'bowie', 'americans', 'damien maia', 'italian', 'bon appétit', 'american christians', 'andrew mccabe', 'wonder world tour', 'fox news', 'joe', 'election day', 'the mich stepping stones daily', 'carlos condit', 'u.s. strike', 'nunes', 'amrican', 'striker', 'michael savage', 'liberals', 'family services', 'camila cabello', 'turkish', 'turkey', 'republicans', 'ufc', 'poc', 'wikipedia', 'eddie alvarez', 'comney', 'civil war', 'bill clinton', 'dave matthews', 'solar system', 'morgan freeman', 'jones', 'asthma', 'type 1 diabetics', 'townhall', 'otc', 'coulter', 'portuguese', 'trousersnake', 'united states of west america', 'messenger', 'this town', 'institutional racism', 'muslim', 'vegas', 'donald trump', 'fbi', 'red carpet', 'gignac', 'steele', 'rep. joe kennedy', 'brexited', 'rodgriguez', 'pbo', 'migos', 'vladimir', 'onuka', 'lou', 'eurovision', 'pro abolotion', 'merrill lynch', 'bahamas', 'gorilla', 'adams', 'united states of flyover america', 'frankieedgar', 'michigan', 'cali', 'mother russia', 'health care law', 'asian', 'arya stark', 'bbc', 'jason knight', 'frankie', 'trump', 'netanyahu', 'senate republicans', 'kay', 'emri', 'house representatives', 'b.j.penn', 'superbowl', 'white house', 'antarctica', 'mmafighting', 'masvidal maia', 'san bernardino', 'europeans', 'spaniard', 'club world cup', 'evanescence', 'wales', 'jamala', 'netherlands', 'mexico', 'tokio hotel', 'kiev', 'anna wintour', 'starving', 'joe silva', 'ronaldo', 'adele', 'cyprus', 'brexir', 'ceo', 'trump tower', 'devin nunes', 'john fugelsang', 'dreamworks', 'salvo', 'russian agent', 'mexican', 'jenga', 'panther', 'iq', 'trumps', 'commonwealth games', 'civil rights laws', 'noah', 'syria strike', 'hr', 'avril lavigne', 'eddie', 'i want candy', 'africans', 'planned parenthood', 'jim comey', 'roger pontare', 'anglo celtic', 'super cup', 'nc', 'croatia', 'india', 'hrc', 'titanium', 'himmel och hav', 'hailee steinfeld', 'ufc 211', 'arab', 'game of thrones', 'jds', 'martin luther king', 'military', 'vets', 'swedens', 'rotherham', 'officer william stacy', 'act 1871', 'maia', 'russias', 'manafort', 'representatives', 'hrw', 'sean spicer', 'msm', 'gamebread', 'wnyers moc', 'irs', 'the death panel', 'rachel', 'zayn', 'snl', 'ag sessions', 'normila', 'obamacare', 'arriba', 'yoongi', 'health bill', 'tillerson', 'tower of babel', 'swedish', 'david guetta', 'payne', 'brazil', 'panthers', 'lynch', 'darth vader', 'belgium', 'flint', 'imperialism', 'tomahawks', 'donald', 'brit', 'asians', 'h8', 'rice', 'chester', 'constitution', 'democrats', 'whittaker', 'sun', 'yodeling', 'foxnews', 'africa', "destiny's child", 'florida', 'carson', 'english', 'la la land', 'george', 'snapchat', 'ufc hall of fame', 'ncah', 'san diego', 'hillary', 'sofia carson', 'iaquinta', 'bobby ryan', 'jfc', 'fisa court', 'warriors', 'barack obama', 'halsey', 'mira', 'ed balls', 'stella', 'barfy cam', 'obamagate', 'leeds bradford airport', 'racism', 'prince kushner', 'nyc', 'oaklettes', 'native americans', 'french', 'united', 'eric allen bell', 'kate bush', 'rashad coulter', 'george soros', 'trumpland', 'vienna', 'rodriguez', 'janesville', 'sessions', 'daniel', 'austrian', 'affordable care act', 'harry cantando', 'romani', 'moon', 'camila', 'spain', 'baywatch', 'dustin poirier', 'jessica andrade', 'british isles', 'bon appetit', 'jorge', 'carter page', 'edgar holloway', 'mayonnaise', 'meia', 'cormier', 'kendrick lamar', '2016 presidential campaign', 'dear white people', 'jason chaffetz', 'united states', 'ron', 'russia', 'ariel g', 'hillary clinton', 'carl', 'wcs', 'le pen', 'united states of  east america', 'oleks skichko', 'darrell issa', 'chucklevision', 'armenia', 'hosue republicans', 'wrestlemania', 'mcgregor', 'libs', 'house of representatives', 'democratic party', 'katy', 'wp', 'sean', 'trumpanzees', 'mt. everest', 'mgk', 'scottish', 'brasil', 'disney', 'united states foreign intelligence surveillance court', 'linkin park', 'islamic', 'bosnia', 'bulgarian', 'tb', 'jews', 'ruslana', 'trumpcare', 'western cape', 'pro-birth', 'avocadies', 'paul manafort', 'nkorea', 'sherman', 'royce', 'nial', 'meu deus', 'gameb', 'nina', 'session', 'rep swalwelll', 'urals', 'svoboda', 'pa', 'louis', 'malibu', 'vodka', 'marine', 'sc', 'ivanka', 'ryan gosling', 'un secgen', 'harry', 'tang', 'obama', 'buzzfeed germany', 'hispanics', 'gamebred', 'ukraine', 'cantando', 'democrat party', 'holloway', 'russian', 'gigi hadid', 'universal health care', 'lbj', 'bbc news', 'medicaid', 'miah', 'salvador', 'flynn', 'republican fbi director comey', 'cuban', 'trump/russia', 'reconstruction south', 'britain', 'scared to be lonely', 'bananies', 'sam clovis', 'ibs', 'president trump', 'syria', 'lynch/obama/rice', 'congressional gop', 'libery university', 'zedd', 'europe', 'gitmo', 'matt brown', 'theresa may', 'luisa', 'ahca', 'bush', 'irish', 'rio', 'canonization', 'masvidal', 'lp', 'james comey', 'american people', 'sax', 'hold tight', 'whitehouse', 'breitbarters', 'joanna jedrzejczyk', 'reps', 'holocaust', 'dolly parton', 'gong show', 'tom clancy', 'ireland', 'sophie turner', 'msnbc', 'progressives', 'mayo', 'jorge masvidal', 'shadow boxing', 'slovenia', 'niall horan', 'nate', 'phil', 'dr.king', 'pro socialism', 'canelo', 'messer', 'spring', 'hof', 'el joven diario', 'kylo ren', 'odessa', 'moscow', 'blair', 'nazi', 'nsa', 'pagie', 'june', 'putin', 'bradford', 'lincoln', 'gauteng', 'authoritarian kleptocracy', 'sherdog forums', 'ukranian', 'clintons', 'mississippi mean', 'u.s', 'risacea', 'democrat', 'american health care act', 'don', 'the sleep inducing backpack', 'pastorjerry', 'house democrats', 'lego', 'mary', 'niall', 'france', 'stipe', 'diane abbott', 'european', 'obama admin', 'jim clyburn', 'poland', 'cat', 'diaz', 'trump-putin', 'djt', 'christians', 'sbs', 'gamebredfighter', 'savior', 'crips', 'collabs', 'prince', 'rep schiff', 'colonialism', 'macron', 'oil deals', 'muslims', 'north america', 'bev hills', 'jim crow', 'judy', 'canada', 'woodley', 'us', 'euro', 'puppet master', 'hitler', 'aca', 'mercedes', 'wapo', 'lord vader', 'tomahawk', 'alessia cara', 'gop health bill', 'tmt', 'lgbt', 'cnn', 'grand ole putin party', 'justin', 'tories', 'chester bennington', 'spanish', 'may 13', 'bryce harper', 'dos santos', 'liberty university', 'usa', 'joanna', 'christmas', 'haille steinfeld', 'italy', 'kabuki', 'holy father', 'california', 'amar pelos dois', 'kayla', 'susan rice', 'damian maia', 'frankie ed', 'bsb', 'salvador sobral', 'twitter', 'shashlik rap', 'vova ostapchuk', 'brazillians', 'antisemitism', 'north cork', 'page', 'hocaust centers', 'cbo', 'it security', 'cub swanson', 'las vegas', 'demian', 'hilary', 'republican party', 'nathan trent', 'martian', 'liberalism', 'chris hayes', 'watergate', 'mlk', 'lee lin', 'horan', 'dt', 'washington post', 'uk', 'paul ryan', 'nonsense karaoke', 'greg leppert', 'sorosian', 'des', 'buzina', 'martin luther king jr', 'computer networking', 'barry', 'gopher', 'twood', 'congress', 'aussie', 'sia', 'gentrification', 'real madrid', 'eurovision song contest', 'zionist', "donna d'erri", 'haim', 'u.k', 'south africa', 'mendes', 'comrade comey', 'rocky balboa', 'israel', 'aaron', 'dc', 'pantera', 'mike flynn', 'female genital mutilation', 'nov 2018', 'liam', 'obama ck', 'iraw war', 'adam levine', 'holocaust denier', 'comey', 'ny', 'gop republicans', 'jurassic world', 'breitbart', 'spicer', 'la', 'russian government', 'acha', 'adm levine', 'hallelujah', 'we cant stop', 'lincoln park', 'healthcare', 'campaign manager manaford', 'abortion', 'big homie stipe', 'wh', 'african americans', 'dq', 'cold war', 'geli', 'alessia', 'stoke', 'ostriches', 'democratic', 'jesus', 'bill nye', 'ukrainian', 'alexander rybak', 'ufc light heavyweight champion daniel cormier', 'vlad putin', 'house passage', 'twilight', 'mcdonalds', 'unicorns', 'chase sherman', 'jacare souza', 'joe-jitsu', 'electoral college', 'ca', 'uranium', 'dm', 'ne', 'sims city', 'limousine', 'rick story', 'clinton', 'the arc', 'wee bum', 'aussies', 'oakley', 'republican', 'american government', 'jorge "gamebread" masvidal', 'sheila jackson']

print(len(my_list2))
# print(my_list2)
# one_1_c=len(intersect(my_list,my_list2))


# print("len brooo",len(my_list),len(my_list2))
flat_tweets=""
for index, row in tweets.iterrows():
    tweetText=str(row['TweetText'])
    flat_tweets=flat_tweets+" "+str(index)+tweetText 

filtered=[]
for candidate in my_list2:

    possibilitiess=makeupperdynamic(candidate)
    matched= [val2 for val2 in possibilitiess if val2 in flat_tweets]
    # print(matched)
    #match found
    # if(index!=1):
        # print(candidate)
    if(len(matched)>0):
        # print(candidate)
        filtered.append(candidate)
        # print(flat_tweets[index])
        # correct_form=string.capwords(candidate)
        # if(correct_form not in flat_tweets):
        #     print(correct_form,candidate)



print(list(set(my_list2) - set(filtered)))
# print(filtered,len(filtered))

# print(one_1_c)

# print(list(set(my_list) - set(filtered)))
# print(flat_tweets)