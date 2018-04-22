names=["aa","ab","ac","ad","ae","af","ag","ah","ai","aj","ak","al","am","an"]

entities=[]
for i in names:
	file_name="segment_"+str(i)+"output"
	file = open(file_name,"r") 
	for line in file: 
		# print (line)
		line_new=line.strip("\n")
		line_new_new=line_new.lower()
		line_new_new_new=line_new_new.strip()
		entities.append(line_new_new_new)
		# entities.append(line)

# print(entities)
dedup=list(set(entities))
# print(dedup)



file = open("ritter_at_least_all_of_them", "w")

for i in dedup:
	file.write(i+"\n")

print(len(dedup))
file.close() 


# import json
# print (json.dumps(dedup))
# print(len(dedup))

# ', 'Michelle Obama\n', 'Saudi Arab…', 'Saudi Arab…\n', 'Trump', 'Trump\n', '', '\n', 'Carlos', 'Carlos\n', '', '\n', '', '\n', '', '\n']