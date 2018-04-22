print(whole_level)

for idx,level in enumerate(whole_level):

    accuracy=level[1]
    tweets_been_processed_list=level[2]

    plt.plot( tweets_been_processed_list,accuracy ,marker='o' , label=batch_size_recorder[idx],alpha=0.5)


    plt.xlabel('# of Tweets Seen')
    plt.ylabel('Execution Time')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig("Execution-Time-vs-Batch-Size.png")

plt.show()