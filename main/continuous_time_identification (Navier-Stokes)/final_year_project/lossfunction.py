import tensorflow as tf 
from tensorflow import keras 
import pandas as pd 
import matplotlib.pyplot as plt 
model=keras.Sequential([keras.layers.Dense(100,activation='relu'),
                       keras.layers.Dense(100,activation='relu'),
                       keras.layers.Dense(10,activation='relu'),
                       keras.layers.Dense(4,activation='relu'),
                       keras.layers.Dense(4,activation='relu')
                       ])
a=pd.read_excel('C:/Users/srbeh/Desktop/Final Year Project/git/v1/PINNs/main/continuous_time_identification (Navier-Stokes)/final_year_project/final year project.xlsx')
print(a)

def loss_fun(self,hw,hs,pw,ps,beta,lam):
    lambda_1=self.lambda_1 
    lambda_2=self.lambda_2 
    phims=(ps-pmin)/(pmax-pmin)
    phimw=(pw-pmin)/(pmax-pmin)
    phihs=(hs-hmin)/(hmax-hmin)
    phihw=(hw-hmin)/(hmax-hmin)
    X=x/z 
    phimwprime=tf.gradients(phimw,X)[0]
    phimsprime=tf.gradients(phims,X)[0]
    phihwprime=tf.gradients(phihw,X)[0]
    phihsprime=tf.gradients(phihs,X)[0]
    f1=(lambda_1*(phims-phimw))-phimwprime
    f2=(lambda_1*(phimw-phims)*beta)-phimsprime
    f3=(lambda_2*(phihs-phihw))+(lambda_1*beta*lam/(hs-hw))-phihwprime
    f4=(lambda_2*(phihw-phihs)*beta)-(lambda_1*beta*lam/(hs-hw))-phihsprime
    dataloss=tf.square(hw-phw)+tf.square(hs-phs)+tf.square(pw-ppw)+tf.square(ps-pps)
    physicsloss=f1+f2+f3+f4
    loss=dataloss+physicsloss 
    return loss 

twi=a['twi'].to_frame()
tsi=a['tsi'].to_frame()
gwi=a['gwi'].to_frame()
gsi=a['gsi'].to_frame()
pwi=a['pwi'].to_frame()
psi=a['psi'].to_frame()
lam=a['lambda'].to_frame()
beta=a['beta'].to_frame()
hw=a['hw'].to_frame()
hs=a['hs'].to_frame()
pw=a['pw'].to_frame()
ps=a['ps'].to_frame()
p=pw.append(ps)
h=hw.append(hs)
pmax=p.max()
pmin=p.min()
hmax=h.max()
hmin=h.min()
f=model.output
phw=f[0]
phs=f[1]
ppw=f[2]
pps=f[3]
z=0.3
x=[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5]

model.compile(optimizer='adam',loss=loss_fun)
model.fit([twi,tsi,gwi,gsi,pwi,psi,lam,beta],[hw,hs,pw,ps])



