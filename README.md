# TwiCSv2
TwiCS is a system which is a implementation of ideas discussed in research paper called "TwiCS: Microblog Entity Mention Detection with Multi-pass Lightweight Computations"
## 1.Usage of the code
This repository contains source code for both experiments that represented in the paper and production code that can be applied for different settings.

### 1.1 Production Code
Production code is great way to navigate the system. Simply run the simulator which is called simulation_cross_validation with your python 3 interpreter.
```
python3 simulation_cross_validation.py
```
### 1.2 Experiments
Experiments are way to assess our performance and effectiveness in order to compare to other systems.
* Usually each experiment has same driver naming for simplicity.
* Some experiments have multiple drivers for different setting.
* In order to find all the drivers for the experiment simply look for the naming pattern of simulation_{VARIATION}	 
```
python3 simulation_cross_validation.py
```

## 2.Structure of the System
We will examine the different phases of system in order to fully understand the system's output. 
### Terminology:
Before we dive into details here are some special meaning words and their corresponding meanings.
* Candidates: Possible named entities which our system detects. 
* TweetBase: A table data structure where we keep tweets and their corresponding candidates.
* IncompleteTweetBase: Modified version of TweetBase where we keep only tweets which have candidates where we are not confident about candidates being named entities.
* CandidateBase: A table data structure where we keep our candidates and their corresponding occurences in the TweetBase. 
### 2.1 PHASE 1
* Input: Tweets (from in file system)
* Output: TweetBase, CandidateBase
* Phase 1 is where we extract name entities from incoming tweets based on usage of capital letters.
* There is no statistical modelling in order to determine named entities from Tweets in this phase.
* Phase 1 is modelled under the name of SatadishaModule_final_trie.py in experiments and production code.

### 2.2 PHASE 2
* Input: TweetBase, CandidateBase, 
* Output: IncompleteTweetBase, CandidateBase
* PHASE 2 rescans the TweetBase in order to find new occurences of existing candidates inside CandidateBase which do not have proper capitalization.
* CandidateBase will be updated accordingly based on new occurences found by PHASE 2.
