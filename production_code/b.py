from collections import Iterable, OrderedDict

d={'a': 5, 'b': 2, 'c': 10, 'd':1, 'e':3}
l=d.keys()
print (l)
d_ordered= OrderedDict(sorted(d.items(), key=lambda x: x[1]))
l=d_ordered.keys()
print (l)
ranked_list=[ d_ordered[key] for key in d_ordered.keys()]
print(ranked_list)