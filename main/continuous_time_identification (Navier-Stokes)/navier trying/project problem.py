import numpy as np
import tensorflow as tf
import pandas as pd
class pinn:
    def xavier_init(self, size):
        in_dim = size[0]        # 2 elements passed with size
        out_dim = size[1]        # one is inner dimension other is outer dimension
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))   # my guess is that it is returning it as a normal distributable vairable
        return tf.Variable(tf.compat.v1.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)  # returns a tensor with size as normally distributed and randomly intialized values normally
    def intialize_NN(self,layers):
        weights=[]          # empty list initialized
        biases=[]           # empty list initialized
        num_layers=len(layers)  # okay got the number of layers in the neural network
        for l in range(0,num_layers-1):   # now run through each layer baically we are turning the dataframe into a neural network with dynamica operations
            W=self.xavier_init(size=[layers[l],layers[l+1]])   # basically gote weights ra tensor return kare with randome normal initialization
            b=tf.Variable(tf.zeros([1,layers[l+1]],dtype=tf.float32),dtype=tf.float32) # basically biases re tensor return kare with initialization with 0
            weights.append(W)                               # point to note is only sei layer ra weiht tensor
            biases.append(b)                                  #only sei layer ra biases tensor
        return weights, biases                      # at the end of iterations weights and biases tensors for each layer is defined. The collectiv data frames weights and baises contain an aggregation of all of them and so we have initilaized a neural network's weights and biases
    def neural_net(self,X,weights,biases):
        num_layers=len(weights)+1  # number of layers length of weights + 1 for outer layer
        H=2.0*(X-self.lb)/(self.ub-self.lb)-1.0     #lb up amra upper and lower bound of X X hela input to neural network here 3 variables x, y ,t H re intially gote intialization deidela tapre seihire amra acivation fun apply kaal pare jou vlaue ase taku rakhidela
        for l in range(0,num_layers-2):
            W=weights[l]
            b=biases[l]
            H=tf.tanh(tf.add(tf.matmul(H,W),b))       # each layer pain H value calculated okay , just look at how the value of H gets out for each layer noice
        W=weights[-1]                                 # basically last weight mane just purba of output layer
        b=biases[-1]                                   # just befor last ra purba biase ta nia heichi
        Y=tf.add(tf.matmul(H,W),b)                     # beauty ethi just output just calculate heichi using purba layer ra H value jouta sethire already achi   output re bhi sei weights multiplied with H value of previous layer plus ammra biase
        return Y                        # beauty output ta return heichi ehti just beauty so at this point we have the output dataframe with us lets beauty notice that number of nodes re starting re 3 tarpre around 20 achi and last re 2 last re 2 ta kemti jana nahin laguchi resize karideichanti
    def callback(self,loss,lambda_1,lambda_2):
        print('Loss %.3e, l1: %.3f, l2:%.5f' % (loss, lambda_1,lambda_2))
    def net_NS(self,twi,tsi,gwi,gsi,pwi,psi,lam,beta):
        lambda_1=self.lambda_1
        lambda_2=self.lambda_2
        two_h_two_p=self.neural_net(tf.concat([twi,tsi,gwi,gsi,pwi,psi,lam,beta],1),self.weights,self.biases)
        hw=two_h_two_p[:,0:1]
        hs=two_h_two_p[:,1:2]
        pw=two_h_two_p[:,2:3]
        ps=two_h_two_p[:,3,4]
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
        return hw,hs,pw,ps,f1,f2,f3,f4 
    def train(self,nIter):
        tf_dict={self.twi_tf:self.twi,
                 self.tsi_tf:self.tsi,
                 self.gwi_tf:self.gwi,
                 self.gsi_tf:self.gsi,
                 self.pwi_tf:self.pwi,
                 self.psi_tf:self.psi,
                 self.lam_tf:self.lam,
                 self.beta_tf:self.beta,
                 self.hw_tf:self.hw,
                 self.hs_tf:self.hs,
                 self.pw_tf:self.pw,
                 self.ps_tf:self.ps,
                 }
        start_time=time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam,tf_dict)
            if it % 10 ==0:
                elapsed= time.time() - start_time
                loss_value=self.sess.run(self.loss,tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)
                print('It: %d, Loss: %.3e, l1: %.3f , l2: %.5f, Time: %.2f' %(it,loss_value,lambda_1_value,lambda_2_value,elapsed))
                start_time=time.time()
        self.optimizer.minimize(self.sess,feed_dict=tf_dict,
                                fetches=[self.loss ,self.lambda_1,self.lambda_2],
                                loss_callback=self.callback)
    def predict(self,twi_star,tsi_star,gwi_star,gsi_star,pwi_star,psi_star,lam_star,beta_star):
        tf_dict={
                self.twi_tf:self.twi_star,
                 self.tsi_tf:self.tsi_star,
                 self.gwi_tf:self.gwi_star,
                 self.gsi_tf:self.gsi_star,
                 self.pwi_tf:self.pwi_star,
                 self.psi_tf:self.psi_star,
                 self.lam_tf:self.lam_star,
                 self.beta_tf:self.beta_star
        }
        ps_star=self.sess.run(self.ps_pred,tf_dict)
        pw_star=self.sess.run(self.pw_pred,tf_dict)
        hs_star=self.sess.run(self.hs_pred,tf_dict)
        hw_star=self.sess.run(self.hw_pred,tf_dict)
        return ps_star, pw_star,hs_star,hw_star 
    def __init__(self,twi,tsi,gwi,gsi,pwi,psi,lam,beta,ps,pw,hs,hw,layers) -> None:
        X=np.concatenate([twi,tsi,gwi,gsi,pwi,psi,lam,beta],1)
        self.lb=X.min(0)
        self.ub=X.max(0)
        self.X=X 
        self.twi=X[:,0:1]
        self.tsi=X[:,1:2]
        self.gwi=X[:,2:3]
        self.gsi=X[:,3:4]
        self.pwi=X[:,4:5]
        self.psi=X[:,5:6]
        self.lam=X[:,6:7]
        self.beta=X[:,7:8]
        self.ps=ps 
        self.pw=pw 
        self.hs=hs 
        self.hw=hw 
        self.layers=layers
        self.weights, self.biases= self.intialize_NN(layers)
        self.lambda_1=tf.Variable([0.0],dtype=tf.float32)
        self.lambda_2=tf.Variable([0.0],dtype=tf.float32)
        self.sess=tf.compat.v1.Session()
        self.twi_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.twi.shape[1]])
        self.tsi_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.tsi.shape[1]])
        self.gwi_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.gwi.shape[1]])
        self.gsi_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.gsi.shape[1]])
        self.pwi_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.pwi.shape[1]])
        self.psi_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.psi.shape[1]])
        self.lam_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.lam.shape[1]])
        self.beta_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.beta.shape[1]])
        self.ps_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.ps.shape[1]])
        self.pw_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.pw.shape[1]])
        self.hs_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.hs.shape[1]])
        self.hw_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.hw.shape[1]])
        self.hw_pred,self.hs_pred,self.pw_pred,self.ps_pred,self.f1_pred,self.f2_pred,self.f3_pred,self.f4_pred=self.net_NS(self.twi_tf,self.tsi_tf,self.gwi_tf,self.gsi_tf,self.pwi_tf,self.psi_tf,self.lam_tf,self.beta_tf)
        self.loss=tf.reduce_sum(tf.square(self.ps_tf-self.ps_pred))+tf.reduce_sum(tf.square(self.pw_tf-self.pw_pred))+tf.reduce_sum(tf.square(self.hs_tf-self.hs_pred))+tf.reduce_sum(tf.square(self.hw_tf-self.hw_pred))+tf.reduce_sum(tf.square(self.f1_pred))+tf.reduce_sum(tf.square(self.f2_pred))+tf.reduce_sum(tf.square(self.f3_pred))+tf.reduce_sum(tf.square(self.f4_pred))
        self.optimizer=tf.compat.v1.train.Optimizer(self.loss,method='L-BFGS-B',
                                                    options={'maxiter':50000,
                                                             'maxfun':50000,
                                                             'maxcor':50,
                                                             'maxls':50,
                                                             })
        self.optimizer_Adam=tf.train.AdamOptimizer()
        self.train_op_Adam=self.optimizer_Adam.minimize(self.loss)
        init=tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        pass
if __name__=="__main__":
    N_train=5000
    layers=[8,24,24,24,24,24,24,24,24,24,24,24,24,4]
    data=pd.read_excel('C:/Users/srbeh/Desktop/Final Year Project/git/v1/PINNs/main/continuous_time_identification (Navier-Stokes)/navier trying/unique.xlsx')
    
    
    
    
    
    
    
    
    
    
    