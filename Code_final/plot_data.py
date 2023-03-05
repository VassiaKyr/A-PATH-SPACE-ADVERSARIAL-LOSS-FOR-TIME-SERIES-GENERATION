import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import os
import csv
import sys

epochs = int(sys.argv[1])  # input from command line
iteration = int(sys.argv[2])  # input from command line
beta = float(sys.argv[3])
gamma = float(sys.argv[4])

sgm=1.0
actf="elu"
layers="10-10"# _16_16"#_only_g"

toy_data = genfromtxt('V2_data/double_well/double_well_well.csv', delimiter=',', dtype='float32')
toy_data_g = genfromtxt('V2_results/double_well/csv/pathgan_beta_'+str(beta)+'_'+str(iteration)+'_nonpar.csv', delimiter=',', dtype='float32')


x=toy_data[:,0]
y=toy_data[:,1]


x_g=toy_data_g[:,0]
y_g=toy_data_g[:,1]

if not os.path.exists('plot_data/'):
    os.makedirs('plot_data/')
   
def plot_data_scatter(x,y,x_g,y_g):
    plt.subplot(2, 1, 1)
    plt.grid()
    plt.scatter(x, y,c="blue",s=2)
    plt.legend(["data"])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2.5])

    
    plt.subplot(2, 1, 2)
    plt.grid()
    plt.scatter(x_g, y_g,c="red",s=2)
    plt.legend(["data_g"])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2.5])

    plt.suptitle("Data and Gener. Data("+actf+layers+") ("+str(beta)+","+str(gamma)+")")
    plt.savefig("plot_data/plot_data_news_V2_"+actf+'_'+str(epochs)+"_"+layers+"_sgm="+str(sgm)+"("+str(beta)+","+str(gamma)+").png")
    plt.close()
    

    '''   
    plt.grid()
    plt.scatter(x, y,c="blue",s=2)
    plt.scatter(x, y,c="red",s=2)
    plt.legend(["data" , "data_g"])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("plot_data/plot_data_V2_sgm="+str(sgm)+".png")
    plt.close()
'''
def plot_data_transition(x,y,x_g,y_g):
    plt.figure(figsize =(12, 12))
    plt.subplot(2, 2, 1)
    N=np.arange(1,len(x)+1)
    plt.plot(N, x,c="blue",label="x") 
    plt.xlabel('Number of data')
    plt.ylabel('x')
    plt.ylim([-2, 2])    
    
    plt.subplot(2, 2, 2)    
    plt.plot(N, y,c="red",label="y")
    plt.xlabel('Number of data')
    plt.ylabel('y')
    plt.ylim([-1.5, 2])
    
    N2=np.arange(1,len(x_g)+1)
    plt.subplot(2, 2, 3)
    plt.plot(N2, x_g,c="blue",label="x") 
    plt.xlabel('Number of data')
    plt.ylabel('x_g')    
    plt.ylim([-2, 2])
    
    plt.subplot(2, 2, 4)    
    plt.plot(N2, y_g,c="red",label="y")
    plt.xlabel('Number of data')
    plt.ylabel('y_g')
    plt.ylim([-1.5, 2])

    plt.suptitle("Transition of Data and Gener. Data("+actf+" "+layers+") ("+str(beta)+","+str(gamma)+")")
    plt.savefig("plot_data/plot_data_news_trans_V2_"+actf+'_'+str(epochs)+"_"+layers+"_sgm="+str(sgm)+"("+str(beta)+","+str(gamma)+").png")
    plt.close()


plot_data_transition(x,y,x_g,y_g)
plot_data_scatter(x,y,x_g,y_g)