import numpy as np 
from chromosome import Chromosome


'''ch_i:first  choromosome'''
'''ch_j:second  choromosome'''
'''fitness:[choromosom, probability]'''
class Genetic:
    def __init__(self,x,y,target):
        self.x = x
        self.y = y
        self.target = target

    def genetic_algorithm(self,minority_set,iterations,fit_function,size):
        chromosomes=[]
        for i in minority_set:
            chromosome=Chromosome(self.x[i],fit_function.predict_proba(self.x[i].reshape(1,-1))[:,self.target][0])
            chromosomes.append(chromosome)

        
        chromosomes=np.array(chromosomes) 
        probability=np.array([item.prob for item in chromosomes])
        # print(probability)
        probability=probability/probability.sum()

        for _ in range(iterations):
            ch_i=np.random.choice(chromosomes,1,p=probability)[0]
            ch_j=np.random.choice(chromosomes,1,p=probability)[0]

            new_gene=self.crossover(ch_i,ch_j,size)
            prob=fit_function.predict_proba(new_gene.reshape(1,-1))[:,self.target][0]
            chromosome=Chromosome(new_gene,prob)
            chromosomes=np.append(chromosomes,chromosome)

            probability=np.array([item.prob for item in chromosomes])
            probability=probability/probability.sum()

        

        best_ch=np.where(probability==probability.max())[0]
        # print(chromosomes[best_ch[0]].prob)
        return chromosomes[best_ch[0]].gene 



    def crossover(self,ch_i,ch_j,size,mutation_rate=0.1):
        p=[ch_i.prob,ch_j.prob]
        p=[item/sum(p) for item in p]

        '''bitmask is an array that shows every feature must inherit from which chromosomes'''
        bitmask=np.random.choice([0,1],size,p=p)

        new_gene=[ch_i.gene[i] if bitmask[i]==0 else ch_j.gene[i] for i in range(size)]

        '''if random numbe is greater than mutation_rate the value of '''
        if(np.random.random()<mutation_rate):
            max_cols=np.amax(self.x,axis=0)
            index=np.random.randint(0,size)
            print(max_cols)
            new_gene[index] = max_cols[index]-new_gene[index]

        return np.array(new_gene)


