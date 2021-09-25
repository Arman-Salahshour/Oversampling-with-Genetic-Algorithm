from genetic import *
from mice import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

class Synthesizing(Genetic):
    def __init__(self,x,y,feature_num,target,n_neighbors=5):
        self.x = x
        self.y = y
        self.feature_num=feature_num
        self.target = target
        self.n_neighbors = n_neighbors
        self.ref_model()

        # '''Inherit all functions from Genetic class'''
        # super.__init__(x,y,target)
    
    def ref_model(self):
        self.knn=KNeighborsClassifier()
        self.knn.fit(self.x,self.y)

    def synthesize(self,desire=10):
        '''find target sets'''
        self.ts=np.where(self.y==self.target)[0]
        self.ts_count=len(self.ts)

        '''use logestic regression for fitness function'''
        logR=LogisticRegression()
        logR.fit(self.x,self.y)

        scalable_list=np.random.choice(self.ts,desire)

        for item in scalable_list:
            neighbors=self.find_neighbors(item)
            temp_x=self.genetic_algorithm(neighbors,len(neighbors),logR,self.feature_num)
            temp_y=self.knn.predict(temp_x.reshape(1,-1))
            # temp_y=logR.predict(temp_x.reshape(1,-1))
            self.x=np.append(self.x,temp_x.reshape(1,-1),axis=0)
            self.y=np.append(self.y,temp_y)

        return self.x,self.y



    def find_neighbors(self,i):
        neighbors=self.knn.kneighbors(self.x[i].reshape(1,-1),n_neighbors=self.n_neighbors, return_distance=False)[0]
        return neighbors



if __name__ == "__main__":
    hd=pd.read_csv('data/processed.cleveland.data')
    columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

    hd.columns=columns
    '''Merging labels which are grater than 0 into 1'''
    hd.target=hd.target.apply(lambda x: 1 if x> 0 else x )

    '''change ? into -1'''
    hd['ca']=hd.ca.apply(lambda x: float(x) if x!='?' else -1)
    hd['thal']=hd.thal.apply(lambda x: float(x) if x!='?' else -1)

    '''features without target column'''
    cols=columns[:]
    cols.remove('target')

    '''handle missing values in ['ca','thal'] features'''
    hd=mice(data_frame=hd,miss_cols=['ca','thal'],list=columns,step=10)

    x=hd[cols].to_numpy()
    y=hd['target'].to_numpy()


    '''Synthesize new data for minority samples'''
    synthesizing=Synthesizing(x,y,13,1,5)
    new_x,new_y=synthesizing.synthesize(desire=20)

    new_hd=pd.DataFrame(new_x,columns=cols)
    new_hd['target']=new_y
    print(new_hd)
    
