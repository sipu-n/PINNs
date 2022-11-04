from mimetypes import init
import sys
sys.path.insert(0,'..')
from tkinter.messagebox import NO
import matplotlib.pyplot as plt
from requests import options
sys.path.insert(0,'C:/Users/srbeh/Desktop/Final Year Project/git/v1/PINNs/Utilities')
import tensorflow as tf
import numpy as np
import scipy as sc
import scipy.io as sio
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
np.random.seed(1234)
tf.random.set_seed(1234)
class pinn:
    def __init__(self,x,y,t,u,v,layers) -> None:
        X=np.concatenate([x,y,t],1)
        self.lb=X.min(0)
        self.ub=X.max(0)
        self.X=X 
        self.x=X[:,0:1]     # 1st column
        self.y=X[:,1:2]     # 2nd column    
        self.t=X[:,2:3]     # 3rd column
        self.u=u 
        self.v=v 
        self.layers = layers        # layers are passed in main
        self.weights, self.biases= self.initialize_NN(layers)       # implemented initatialize the neural network
        self.lambda_1= tf.Variable([0.0],dtype=tf.float32)          # initialized lambda 1 and lambda 2
        self.lambda_2=tf.Variable([0.0],dtype=tf.float32)
        tf.compat.v1.disable_eager_execution()
        self.sess=tf.compat.v1.Session()
        self.x_tf=tf.compat.v1.placeholder(tf.float32,shape=[None,self.x.shape[1]])
        self.y_tf=tf.compat.v1.placeholder(tf.float32,shape=[None, self.y.shape[1]])
        self.t_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        self.u_pred,self.v_pred,self.p_pred,self.f_u_pred,self.f_v_pred=self.net_NS(self.x_tf,self.y_tf,self.t_tf)
        self.loss=tf.reduce_sum(tf.square(self.u_tf-self.u_pred))+\
            tf.reduce_sum(tf.square(self.v_tf-self.v_pred))+\
                tf.reduce_sum(tf.square(self.f_u_pred))+\
                    tf.reduce_sum(tf.square(self.f_v_pred))
        self.optimizer= tf.compat.v1.train.Optimizer(self.loss,
                                                     method='L-BFGS-B',
                                                     options={'maxiter':50000,
                                                              'maxfun':50000,
                                                              'maxcor':50,
                                                              'maxls':50,
                                                              'ftol':1.0*np.np.finfo(float).eps
                                                              }
                                                        )
        self.optimizer_Adam=tf.train.AdamOptimizer()
        self.train_op_Adam= self.optimizer_Adam.minimize(self.loss)
        init= tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        pass
    def initialize_NN(self,layers):
        weights=[]          # empty list initialized
        biases=[]           # empty list initialized
        num_layers=len(layers)  # okay got the number of layers in the neural network
        for l in range(0,num_layers-1):   # now run through each layer baically we are turning the dataframe into a neural network with dynamica operations
            W=self.xavier_init(size=[layers[l],layers[l+1]])   # basically gote weights ra tensor return kare with randome normal initialization
            b=tf.Variable(tf.zeros([1,layers[l+1]],dtype=tf.float32),dtype=tf.float32) # basically biases re tensor return kare with initialization with 0
            weights.append(W)                               # point to note is only sei layer ra weiht tensor
            biases.append(b)                                  #only sei layer ra biases tensor
        return weights, biases                      # at the end of iterations weights and biases tensors for each layer is defined. The collectiv data frames weights and baises contain an aggregation of all of them and so we have initilaized a neural network's weights and biases
    def xavier_init(self, size):
        in_dim = size[0]        # 2 elements passed with size
        out_dim = size[1]        # one is inner dimension other is outer dimension
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))   # my guess is that it is returning it as a normal distributable vairable
        return tf.Variable(tf.compat.v1.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)  # returns a tensor with size as normally distributed and randomly intialized values normally
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
    def net_NS(self,x,y,t):
        lambda_1=self.lambda_1                               # se jou 2 ta unknown constants define karinathila sei gudaka define heichi first
        lambda_2=self.lambda_2 
        psi_and_p = self.neural_net(tf.concat([x,y,t],1),self.weights,self.biases)
        """
        ta basically psi_and_p variable re amra pura solution ta rakha heichi
        tapre taku 2 ta variable p and psi re split heichi 
        solution value gote list ethi achi taku split heichi using indexing gote list with step 1 au gote with step2
        se comma operator bisayre tike dekhide in []
        comma basically gote trick pura gote column ru value utheibara naki rows jouta ki ame usually karu
        ta ,0 mane 1st column ta
        ,1 mane 2nd column ta
        column range slicing heichi au kichi nahin
        0:1 mane 0 index column ta select heichi
        1:2 mane 1 index column ta select heichi
        :,0:1 ethi jou 1st : nahi seta emti lekhile pura data frame ta nei hue
        ta basically psi re 1st column ta jauchi and p re 2nd column
        just last doubt ki sabue tensor gudaka ra size kemti manipulate hauchi and
        last re [[,],[,].....] ei pattern re Y ra value return hauchi ki maniki chal ebe pain
        layers re 2 jete bele pass karu eita adjust heijae bodhe
        ta penultimate tensor re jetiki achi elements segudda sabu resize heijae  and last re ama tensor deidie"""
        psi=psi_and_p[:,0:1]      
        p=psi_and_p[:,1:2]          
        u=tf.gradients(psi,y)[0]      # gradients 2 ta value return kare [,] amku 1st ta darkar     
        v=-tf.gradients(psi,x)[0]     # gradients of a tensor dekhde kam sarla
        u_t=tf.gradients(u,t)[0]        # gradients amku 2 ta element return kare amku 1st ta darkar
        u_x=tf.gradients(u,x)[0]        # ei sabu gote got data frame i guess
        u_y=tf.gradients(u,y)[0]
        u_xx=tf.gradients(u_x,x)[0]
        u_yy=tf.gradients(u_y,y)[0]
        v_t=tf.gradients(v,t)[0]
        v_x=tf.gradients(v,x)[0]
        v_y=tf.gradients(v,y)[0]
        v_xx=tf.gradients(v_x,x)[0]
        v_yy=tf.gradients(v_y,y)[0]
        p_x=tf.gradients(p,x)[0]
        p_y=tf.gradients(p,y)[0]
        f_u=u_t + lambda_1 * (u*u_x+v*u_y) + p_x - lambda_2*(u_xx+u_yy)
        f_v=v_t + lambda_1*(u*v_x + v*v_y) + p_y- lambda_2*(v_xx+ v_yy)
        return u,v,p,f_u,f_v                # seita last p and psi ku use kariki u,v,p,f_u,f_v value gudaka die neural network will do this feel the stuff jou jagare jaha lekha heich variable neural network nijaku adjust karidaba crazy and beautiful
    def callback(self,loss,lambda_1,lambda_2):
        print('Loss %.3e, l1: %.3f, l2:%.5f' % (loss, lambda_1,lambda_2))
    def train(self,nIter):
        tf_dict={self.x_tf:self.x, self.y_tf:self.y, self.t_tf:self.t,
                 self.u_tf:self.u , self.v_tf:self.v}
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
    def predict(self,x_star,y_star,t_star):
        tf_dict= {self.x_tf:x_star, self.y_tf:y_star, self.t_tf:t_star}
        u_star=self.sess.run(self.u_pred, tf_dict)
        v_star=self.sess.run(self.v_pred, tf_dict)
        p_star=self.sess.run(self.p_pred,tf_dict)
        return u_star, v_star ,p_star 
    def plot_solution(self,X_star,u_star,index):
        lb=X_star.min(0)
        ub=X_star.max(0)
        nn=200
        x=np.linspace(lb[0],ub[0],nn)
        y=np.linspace(lb[1],ub[1],nn)
        X,Y= np.meshgrid(x,y)
        U_star=griddata(X_star,u_star.flatten(),(X,Y),method='cubic')
        plt.figure(index)
        plt.pcolor(X,Y,U_star,cmap='jet')
        plt.colorbar()
    def axisEqual3D(self,ax):
        extents= np.array([getattr(ax,'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz=extents[:,1]-extents[:,0]
        centers=np.mean(extents,axis=1)
        maxsize=max(abs(sz))
        r=maxsize/4
        for ctr,dim in zip(centers,'xyz'):
            getattr(ax,'set_{}lim'.format(dim))(ctr-r,ctr+r)
if __name__ == "__main__": 
    N_train = 5000
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    
    # Load Data
    data = sio.loadmat('C:/Users/srbeh/Desktop/Final Year Project/git/v1/PINNs/main/Data/cylinder_nektar_wake.mat')         
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    # Training
    model = pinn(x_train,y_train,t_train,u_train,v_train,layers)
    model.train(N_train)
    
    # Test Data
    snap = np.array([100])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    
    u_star = U_star[:,0,snap]
    v_star = U_star[:,1,snap]
    p_star = P_star[:,snap]
    
    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)
    
    # Error
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
    
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))                  
    
    # Plot Results
#    plot_solution(X_star, u_pred, 1)
#    plot_solution(X_star, v_pred, 2)
#    plot_solution(X_star, p_pred, 3)    
#    plot_solution(X_star, p_star, 4)
#    plot_solution(X_star, p_star - p_pred, 5)
    
    # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')
    plt.savefig('./figures/NavierStokes_prediction')
    
    