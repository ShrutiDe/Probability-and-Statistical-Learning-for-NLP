## B551 Assignment 3: Probability and Statistical Learning for NLP


### Part 1: Part-of-speech tagging

Initial Process was to train the given data on the bases of required counts to generate probabilities.

#### a.	Simple Model:
For Simple Model, the probability of each word was calculated simply by considering the part of speech related to that word. Here the emission probability was calculated for each word with corresponding pos and max of these probabilities was selected amongst the other pos and was assigned to that word.

Emission probability for word[i] and pos[i]= count of word[i] and pos[i] together / count(pos[i])

Snippet of emission calulation:
```python
for word in self.unique_words.keys():
            for pos in self.unique_pos.keys():
                if word not in self.emission_prob:
                    self.emission_prob[word] = {}
                if pos in self.wordToPos_count[word]:
                    self.emission_prob[word][pos] = float(self.wordToPos_count[word][pos]) / float(self.pos_count[pos])
                else:
                    self.emission_prob[word][pos] = self.minProb

```
For posterior probability: 
Posterior P(S|W) = Likelihood (P(S) * P(W|S))

**We achieved a word accuracy up to 90% and sentence accuracy up to 40% for bc. test file.**

#### b.	Hidden Markov Model:
For Hidden Markov Model, according to the Bayes net model, we had dependency from one pos to previous pos. We had to take into account the emission probability as done in simple model along with transition probability.

Here we created a dictionary named Viterbi to store the max probability for that word with reference to that pos along with value of pos which gave max probability of tranisition.

Emission probability for word[i] and pos[i]= count of word[i] and pos[i] tagged together / count(pos[i])

Transition probability[pos][previous pos]= count of [previous_pos][pos] tagged together / count(previous pos)

Max probability = probability of viterbi[prev_word][pos_1])*emission [word][pos] *transition[pos][previous pos]*prob of pos)) 

For backtracking: 

Considered the start point as the max probability from the Viterbi dictionary for last word in the sentence.
Now backtracking from this start point, taking into considering the previous pos stored at this position with max probability was now considered the next pos tag (start point) and this process was repeated till we reach the first word in sentence, backtracking on the max probability values and corresponding previous pos.

Code snippet:

```python
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
```
               
**We achieved a word accuracy up to 95% and sentence accuracy up to 50% for bc. test file.**


#### c.	Gibbs Sampling Model:
For Gibbs Sampling Model, according to the Bayes net model, there was additional dependency for last pos in the sentence. It had transition probabilities for both previous pos and first pos in the sentence.
P(posArray[length of sentence] | posArray[1],posArray[length of sentence-1]

For this model, the transition probabilities were calculated same like HMM model, except for last pos in the sentence. 

Transition Probability for pos on last position in sentence.

Transition prob[pos][start_pos,prev_pos]= count of [pos][start_pos,prev_pos]) / count of [prev_pos][pos] + count of [" "][pos])
                    
In gibbs sampling, the sampling is done by generating 500 samples, out of which first 100 are excluded.

As initial pos predicted array, we have given the output from simplified model.
Each new sample is created by taking the previous sample generated as the initial sample. By looping on each word for all 12 pos having the other pos tags constant. For each of 12 pos, probability of current word being current pos is calculated and maximum value of all is considered and saved for that word.

Post this process, all the samples are stored in a list along with each sample as list of pos. 
For calculating the final pos for word, below process is followed.
-For word i, every samples ith pos is taken and stored in a dictionary from where the pos is selected with maximum count and considered as the final pos for that word.
	
**We achieved a word accuracy up to 92% and sentence accuracy up to 50% for bc. test file.**

### Part 2: Code breaking

The problem requires us to decrypt a given input file using 2 methods. Replacement, where every letter in the alphabet is replaced with another. Rearrangement, where the first 4 indices are shuffled in a random sequence.

**Proposed Solution:**

Initially to solve the problem, it was required to calculate the probabilities of occurrence of a letter, given the previous letter. 
For this calculation, we created a Frequency dictionary, that contains all the frequencies of a letter occurring, given a previous letter as key. Example, 

```json
{
  "a": {
    "a": 120,
    "b": 230,
      …
  }
}
```

This design decision was made to implement faster fetches of frequencies with order O (1). The probability of a letter is calculated by dividing the frequency against the total count of the characters of the input.

**Code Snippet**
```python
def get_probability_table(corpus):
    letters = list(map(chr, range(97, 123))) + [' ']
    frequency_map = {letters[i]: {letters[j]: 0 for j in range(len(letters))} for i in range(len(letters))}
    words = corpus.split()
    for i in range(len(words)):
        for j in range(len(words[i])):
            if j == 0:
                frequency_map[" "][words[i][j]] += 1
            else:
                frequency_map[words[i][j]][words[i][j - 1]] += 1
    return frequency_map, len(corpus)
```

**Metropolis-Hastings algorithm**

For finding the best possible encryption tables we use the algorithm. For implementing the algorithm, we followed an iterative approach and generated replacement and rearrangement tables during runtime. The initial tables are a random shuffle over the possible values.

A significant design decision that was taken was minimizing the sum over the negative logs of the probabilities. This decision was made to avoid underflow, when the probabilities of the letters became extremely small. 

Another design decision was while generating the next decryption tables, we gave a higher probability to the replacement table over the rearrangement table. This was because there are more possibilities of change in the replacement table than the rearrange.

Finally, we maintain the minimum probability and return the result as a decryption with the tables used to attain the minimum. 



### Part 3: Spam Classification
1. We use the Naive Bayes classifier for this task.
2. The vocabulary is based on the words extracted from the given files in the train directory.
3. We clean text based on the following:
	- Remove HTML tags using regex
	- Lowecase
	- 2 < length(word) < 20
	- word.strip(special_characters)
```python
cleaned_text = []  
text = text.lower()  
text = re.sub('<.*?>', '', text)  
text = text.split()  
for i in text:  
    if 2 < len(i) < 20:  
        cleaned_text.append(i.strip("\'?!():;\""))  
return cleaned_text
```
4. We use a Dirchlet prior with m=0.1 as the smoothing parameter.
5. The posterior thus becomes proportional to P(w)P(t|w) for both t=spam and t=notspam.
6. Our prediction chooses the label with the greatest posterior for a maximum aposteriori prediction as per the following code snippet
``` python
positive_vocab_count = len(V[1])  
negative_vocab_count = len(V[0])  
num_positive_count = sum(V[1].values())  
num_negative_count = sum(V[0].values())  
m = 0.1  
prior_positive = 0.5  
prior_negative = 1 - prior_positive  
pos_denominator = num_positive_count + m * (positive_vocab_count + negative_vocab_count)  
neg_denominator = num_negative_count + m * (positive_vocab_count + negative_vocab_count)  
predictions = []  
for datapoint in data:  
    prob_positive = np.log(prior_positive)  
    prob_negative = np.log(prior_negative)  
    file_name = datapoint[0]  
    words = datapoint[1]  
    for word in words:  
        if word in V[1]:  
            prob_positive += np.log((V[1][word] + m) / pos_denominator)  
        else:  
            if m > 0:  
                prob_positive += np.log(m / pos_denominator)  
            else:  
                prob_positive -= np.log(pos_denominator)  
  
        if word in V[0]:  
            prob_negative += np.log((V[0][word] + m) / neg_denominator)  
        else:  
            if m > 0:  
                prob_negative += np.log(m / neg_denominator)  
            else:  
                prob_negative -= np.log(neg_denominator)  
  
    if prob_positive > prob_negative:  
        predictions.append([file_name, 'spam'])  
    else:  
        predictions.append([file_name, 'notspam'])
```
7. We arrived at the choice of m=0.1 using 10-fold cross-validation over the search space of m={0,0.1,...,0.9,1,2,...,10}
**We achieved an accuracy of 98.15% with m=0.1 and 97.84% with no smoothing on the given data**
