###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids:desaish-aramali-anipatil-
   
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import numpy as np


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    
    def __init__(self):
        self.minProb = 1e-10
        self.unique_words = {}
        self.unique_pos = {}
        self.totalWordCount = 0
        self.stmEnd={}
        self.word_count={}
        self.pos_count = {}
        self.wordToPos_count = {}
        self.posToPos_count = {}
        self.pos_prob = {}
        self.emission_prob = {}
        self.transition_prob = {}
        self.transition_first_last_count={}
        self.transition_first_last_prob={}
        
    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            posterior = 0
            for i in range(len(sentence)):
                if sentence[i] in self.emission_prob and label[i] in self.emission_prob[sentence[i]]:
                     posterior += math.log(self.emission_prob[sentence[i]][label[i]])+math.log(self.pos_prob[label[i]])
            return posterior
        elif model == "Complex":
            posterior_complex = 0
            for l in range(len(sentence)):
                word = sentence[l]
                pos = label[l]
                if word not in self.emission_prob:
                    x = self.minProb
                else:
                    x = self.emission_prob[word][pos]
                posterior_complex += math.log(x)
                if l==0:
                    posterior_complex += math.log(self.pos_prob[pos])
                elif l==len(sentence):
                    pos_start = label[0]
                    pos_prev2 = label[l-1]
                    posterior_complex += math.log( self.transition_first_last_prob[pos][pos_start][pos_prev2])+math.log(self.pos_prob[pos_start])+math.log(self.pos_prob[pos_prev2])
                else:
                    pos_prev1 = label[l-1]
                    posterior_complex += math.log(self.transition_prob[pos][pos_prev1])+math.log(self.pos_prob[pos_prev1])
                # print(prob)
            return posterior_complex
        elif model == "HMM":
            posterior = 0
            for i in range(len(sentence)):
                if i == 0:
                    if sentence[i] in self.emission_prob:
                        if label[i] in self.emission_prob[sentence[i]]:
                            if label[i] in self.transition_prob[" "]:
                                posterior += math.log(self.transition_prob[" "][label[i]])+math.log(self.emission_prob[sentence[i]][label[i]])
                            else: 
                                posterior += math.log(self.minProb)
                        else: 
                                posterior += math.log(self.minProb) 
                    else: 
                                posterior += math.log(self.minProb) 
                else:
                    posterior += math.log(self.transition_prob[label[i]][label[i-1]])
                if sentence[i] in self.emission_prob and label[i] in self.emission_prob[sentence[i]]:
                    posterior += math.log(self.emission_prob[sentence[i]][label[i]])+math.log(self.pos_prob[label[i]])
               
            return posterior
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
         for sent,posArray in data:
            self.totalWordCount += len(sent)
            for i in range(len(sent)):
                #each word count
                if sent[i] in self.word_count:
                    self.word_count[sent[i]]+=1
                else:
                    self.word_count[sent[i]]=1
                #each pos count    
                if posArray[i] in self.pos_count:
                    self.pos_count[posArray[i]] += 1
                else:
                    self.pos_count[posArray[i]] = 1 
                #unique set of words    
                if sent[i] not in self.unique_words:
                    self.unique_words[sent[i]] = True
                #unique set of pos      
                if posArray[i] not in self.unique_pos:
                    self.unique_pos[posArray[i]] = True
                
                # word at last position in sentence
                if posArray[len(posArray)-1] not in self.stmEnd:
                    self.stmEnd[posArray[len(posArray)-1]]=0
                    self.stmEnd[posArray[len(posArray)-1]]+=1
            
                if sent[i] in self.wordToPos_count:
                    if posArray[i] in self.wordToPos_count[sent[i]]:
                        self.wordToPos_count[sent[i]][posArray[i]] += 1
                    else:
                        self.wordToPos_count[sent[i]][posArray[i]] = 1
                else:
                    self.wordToPos_count[sent[i]] = {}
                    self.wordToPos_count[sent[i]][posArray[i]] = 1
                    
                if i==len(sent):
                    if posArray[i] in  self.transition_first_last_count:
                        if [posArray[0],posArray[i-1]] in  self.transition_first_last_count[posArray[i]]:
                             self.transition_first_last_count[posArray[i]][posArray[0],posArray[i-1]] += 1
                        else:
                             self.transition_first_last_count[posArray[i]][posArray[0],posArray[i-1]] = 1
                    else:
                         self.transition_first_last_count[posArray[i]] = {}
                         self.transition_first_last_count[posArray[i]][posArray[0],posArray[i-1]] = {}
                        
                if i == 0:
                    if " " in self.posToPos_count:
                        if posArray[i] in self.posToPos_count[" "]:
                            self.posToPos_count[" "][posArray[i]] += 1
                        else:
                            self.posToPos_count[" "][posArray[i]] = 1
                    else:
                        self.posToPos_count[" "] = {}
                        self.posToPos_count[" "][posArray[i]] = 1

                if posArray[i] in self.posToPos_count:
                        if posArray[i-1] in self.posToPos_count[posArray[i]]:
                            self.posToPos_count[posArray[i]][posArray[i-1]] += 1
                        else:
                            self.posToPos_count[posArray[i]][posArray[i-1]] = 1
                else:
                        self.posToPos_count[posArray[i]] = {}
                        self.posToPos_count[posArray[i]][posArray[i-1]] = 1
                        
        # pos probability
         for pos in self.unique_pos.keys():
            self.pos_prob[pos] = float(self.pos_count[pos]) / float(self.totalWordCount)
        # emission probability
         for word in self.unique_words.keys():
            for pos in self.unique_pos.keys():
                if word not in self.emission_prob:
                    self.emission_prob[word] = {}
                if pos in self.wordToPos_count[word]:
                    self.emission_prob[word][pos] = float(self.wordToPos_count[word][pos]) / float(self.pos_count[pos])
                else:
                    self.emission_prob[word][pos] = self.minProb
         # transition probability
         for pos in self.unique_pos.keys():
            if pos not in self.transition_prob:
                self.transition_prob[pos] = {}
            for prev_pos in self.unique_pos.keys():
                if pos in self.posToPos_count[prev_pos]:
                    self.transition_prob[pos][prev_pos] = float(self.posToPos_count[prev_pos][pos]) / float(self.pos_count[prev_pos])
                else:
                    self.transition_prob[pos][prev_pos] = self.minProb

       # transition probability for first word in sentence
         for pos in self.unique_pos.keys():
            if pos in self.posToPos_count[" "]:
                self.transition_prob[" "]={}
                if pos in self.transition_prob[" "]:
                    self.transition_prob[" "][pos] = float(self.posToPos_count[" "][pos]) / float(self.pos_count[pos])
                else:
                    self.transition_prob[" "][pos] = self.minProb
            else:
                 self.posToPos_count[" "][pos]={}
                
         for pos in self.unique_pos.keys():
            if pos not in  self.transition_first_last_prob:
                 self.transition_first_last_prob[pos] = {}
            if pos not in  self.transition_first_last_count:
                 self.transition_first_last_count[pos] = {}
            for start_pos in self.unique_pos.keys(): 
                for prev_pos in self.unique_pos.keys():
                    if (start_pos,prev_pos) in  self.transition_first_last_count[pos]:
                        if (start_pos,prev_pos) not in  self.transition_first_last_prob[pos]:
                             self.transition_first_last_prob[pos][start_pos,prev_pos] = {}
                        else:
                             self.transition_first_last_prob[pos][start_pos,prev_pos]=float( self.transition_first_last_count[pos][start_pos,prev_pos]) / float(self.pos_count[pos]+self.pos_count[start_pos]+self.pos_count[prev_pos])
                    else:
                         self.transition_first_last_prob[pos][start_pos,prev_pos]=self.minProb
            

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        simple_posArray = []
        for word in sentence:
            prob_max = -1
            pos_max = "pos"
            for pos in self.unique_pos.keys():
                if word not in self.emission_prob:
                    x = self.minProb
                else:
                    x = self.emission_prob[word][pos]
                if x > prob_max:
                    prob_max = x
                    pos_max = pos
            simple_posArray.append(pos_max)

        return simple_posArray

    def complex_mcmc(self, sentence):
        pos_predicted = self.simplified(sentence)
        init_predicted = pos_predicted
        k = 0
        length=0
        columns={}
        samples = []
        for k in range(300):
            sample_predicted = self.gibbs_sample(sentence, init_predicted)
            init_predicted=sample_predicted
            if(k>100):
                samples.append(sample_predicted)
                length=len(sample_predicted)
            
            
        
        for j in range(length):
            for sample in samples:
                if sample[j] in columns:
                    columns[sample[j]]+=1
                else:
                    columns[sample[j]]=1  
            
            pos_predicted.append([max(columns, key=columns.get)])
            columns={}       
        pos_predicted = pos_predicted[:len(pos_predicted)//2]
        return pos_predicted
    
    def gibbs_sample(self, sentence, init_predicted):
        pos_sample = list(init_predicted)
        for i in range(len(sentence)):
            pos = []
            pos_prob = []
            for curr_pos in self.unique_pos.keys():
                pos.append(curr_pos)
                pos_sample[i] = curr_pos
                curr_pos_prob = self.get_probability(sentence, pos_sample)
                pos_prob.append(math.exp(curr_pos_prob))
                
            max_val=pos_prob.index(max(pos_prob))
            max_pos=pos[max_val]
            pos_sample[i]=max_pos

        return pos_sample
    
    def get_probability(self, sentence, pos_predicted):
        prob = 0
        for l in range(len(sentence)):
            word = sentence[l]
            pos = pos_predicted[l]
            if word not in self.emission_prob:
                x = self.minProb
            else:
                x = self.emission_prob[word][pos]
            prob += math.log(x)
            if l==0:
                prob += math.log(self.pos_prob[pos])
            elif l==len(sentence):
                pos_start = pos_predicted[0]
                pos_prev2 = pos_predicted[l-1]
                prob += math.log( self.transition_first_last_prob[pos][pos_start][pos_prev2])+math.log(self.pos_prob[pos_start])+math.log(self.pos_prob[pos_prev2])
            else:
                pos_prev1 = pos_predicted[l-1]
                prob += math.log(self.transition_prob[pos][pos_prev1])+math.log(self.pos_prob[pos_prev1])
           
        return prob


    def hmm_viterbi(self, sentence):
       viterbi={}
       viterbi_posArray=[]
       count=0
       pos_max = "pos"
       for count in range(0,len(sentence)):
          word=sentence[count]
          viterbi[count]={}
          if count == 0:
            for pos in self.unique_pos.keys():
               if word not in self.unique_words.keys():
                   e = self.minProb
               else:
                   e = self.emission_prob[sentence[0]][pos]  #emission
               if pos not in self.transition_prob:
                   t=self.minProb
               else:
                   if pos not in self.transition_prob[" "]:
                       t=self.minProb
                   else:
                       t=self.transition_prob[" "][pos]   #tranission
               viterbi[count][pos]=(float(e*t)," ")
           
          else:
            prev_word=count-1
            for pos in self.unique_pos.keys():
               maxval=-9999999
               if word not in self.unique_words.keys():
                      e=self.minProb
               else:
                       e=self.emission_prob[word][pos]  #emission
               for pos_1 in self.unique_pos.keys():
                      if pos_1 not in self.transition_prob[pos]:  ######Tcount
                          t=self.minProb
                      else:
                          t=self.transition_prob[pos_1][pos]  # transission   #####ordring
                      if pos not in self.pos_count:
                          prob=self.minProb
                      else:
                          prob=self.pos_count[pos_1]
                          
                      if pos_1 in viterbi[prev_word]:
                          prob_max=float(viterbi[prev_word][pos_1][0])*(float(e*t*prob))
                          if prob_max>maxval:
                              maxval=prob_max
                              pos_max=pos_1
                              viterbi[count][pos]=(maxval,pos_max)    
                                
               
               
      # print(viterbi)    
       word=count+1
       viterbi[word]={}
       for pos in self.stmEnd:
           if pos in viterbi[count]:
              if pos not in self.stmEnd:
                  prob=self.minProb
              else:
                  prob=float(self.stmEnd[pos]/float(sum(self.stmEnd.values())))
              viterbi[word][pos]=(float(viterbi[count][pos][0]*prob),pos)

       start=max(viterbi[len(sentence) - 1],key=viterbi[len(sentence)].get)
       for count in range(len(sentence),-1,-1):
            if count==len(sentence):
               pos=viterbi[count][start][1]
                   
            else:
               pos=viterbi[count][pos][1]
               
            viterbi_posArray.append(pos)
       viterbi_posArray.reverse()
       viterbi_posArray.remove(" ")
        
       return viterbi_posArray


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
