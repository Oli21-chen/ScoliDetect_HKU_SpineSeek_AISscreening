'''
# use reference point to adjust translational movement of coordinates system
# Get mean&std at the end to analysis acc and prec

1. compare 2 different signal- Normalization of signal from unknown camera pose
可以证明上下两个角度的视频信息可以标准化
    
'''
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import pdb
from scipy.stats import zscore
from scipy import stats
from utili_Getpose_v4 import GetAllFeatures

#%%  Task 1

filepath_ref1=r'C:\Users\Olive\Desktop\table\sz_622_step_1.csv' # cam high
filepath_ref2=r'C:\Users\Olive\Desktop\table\sz_624_step_1.csv' # cam low

coors = [['x1','y1'],['x2','y2'],['x3','y3'],['x4','y4'],['x5','y5'],
         ['x6','y6'],['x7','y7'],['x8','y8'],['x9','y9'],['x10','y10'],
         ['x11','y11'],['x12','y12'],['x13','y13'],['x14','y14'],['x15','y15'],
         ['x16','y16'],['x17','y17']]
nums = len(coors)

def read_data_for_comparing_the_normalization_results(
        filepath_ref1,
        filepath_ref2,
        index ):
    'add zscore here is for easily comparing the normalization results'
    ref1 = pd.read_csv(filepath_ref1,header=0).values[:,:-1].astype('float32')#
    ref1_x = zscore(ref1[:,index])
    ref1_y = zscore(ref1[:,index+1])
    
    ref1_norm = np.squeeze(GetAllFeatures(ref1))
    ref1_xnorm = zscore(ref1_norm[:,index])
    ref1_ynorm = zscore(ref1_norm[:,index+1])
    
    
    ref2 = pd.read_csv(filepath_ref2,header=0).values[:,:-1].astype('float32')#
    ref2_x = zscore(ref2[:,index])
    ref2_y = zscore(ref2[:,index+1])
    ref2_norm = np.squeeze(GetAllFeatures(ref2))
    ref2_xnorm = zscore(ref2_norm[:,index])
    ref2_ynorm = zscore(ref2_norm[:,index+1])
    
    return ref1_x,ref2_x,ref1_xnorm,ref2_xnorm,ref1_y,ref2_y,ref1_ynorm,ref2_ynorm

def plt_GetAllFeatures_zscore(
        index,
        ref1_x,
        ref2_x,
        ref1_xnorm,
        ref2_xnorm,
        ref1_y,
        ref2_y,
        ref1_ynorm,
        ref2_ynorm):
    # plot results
    fig, axs = plt.subplots(2,2,figsize=(10,6))
    axs[0,0].plot(ref1_x, label='X-ref1')
    axs[0,0].plot(ref2_x, label='X-ref2')
    axs[0,0].set_title('X-Coordinates')
    axs[0,0].legend()
    
    axs[0,1].plot(ref1_xnorm,label='Xnorm-ref1')
    axs[0,1].plot(ref2_xnorm,label='Xnorm-ref2')
    axs[0,1].set_title('Xnorm-Coordinates')
    axs[0,1].legend()
    
    
    axs[1,0].plot(ref1_y,label='Y-ref1')
    axs[1,0].plot(ref2_y,label='Y-ref2')
    axs[1,0].set_title('Y-Coordinates')
    axs[1,0].legend()
    
    axs[1,1].plot(ref1_ynorm,label='Ynorm-ref1')
    axs[1,1].plot(ref2_ynorm,label='Ynorm-ref2')
    axs[1,1].set_title('Ynorm-Coordinates')
    axs[1,1].legend()
    
    fig.suptitle("Comparison between high cam and low cam \n index:{} ".format(index+1))
    # Display the plots
    plt.tight_layout()
    plt.show()
    
'''
for index in range(nums):
    ref1_x,ref2_x,ref1_xnorm,ref2_xnorm,ref1_y,ref2_y,ref1_ynorm,ref2_ynorm = read_data_for_comparing_the_normalization_results(
            filepath_ref1,
            filepath_ref2,
            index )    
    plt_GetAllFeatures_zscore(
        index, 
        ref1_x,
        ref2_x,
        ref1_xnorm,
        ref2_xnorm,
        ref1_y,
        ref2_y,
        ref1_ynorm,
        ref2_ynorm)
'''

#%% Task 2 Draw 2D Gaussian and plot scatter
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
''' 
for index in range(nums):
    ref1_x,ref2_x,ref1_xnorm,ref2_xnorm,ref1_y,ref2_y,ref1_ynorm,ref2_ynorm = read_data_for_comparing_the_normalization_results(
            filepath_ref1,
            filepath_ref2,
            index )    
    fig, axs = plt.subplots(2,2,figsize=(10,6))  
    
    x,y = ref1_x,ref1_y 
    u_x,u_y=np.mean(x),np.mean(y)
    std_x,std_y=np.std(x),np.std(y)
    mse=((np.subtract(np.transpose([x,y]),[u_x,u_y])**2).sum(axis=1)).mean(axis=None)  
    axs[0,0].scatter(x,y, s=0.5)
    axs[0,0].axvline(x=u_x,c='grey', lw=1)
    axs[0,0].axhline(y=u_y,c='grey', lw=1)
    confidence_ellipse(x, y, axs[0,0],n_std=1, 
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, axs[0,0], n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, axs[0,0], n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':')
    axs[0,0].scatter(u_x, u_y, c='red', s=3,label='ref1')
    axs[0,0].set_title('Ref1 \n MSE: {:.2g}\n 2D mean: {:.2g}, {:.2g}\n 2D std: {:.2g}, {:.2g}'
                  .format(mse,u_x,u_y,std_x,std_y))
    axs[0,0].set_xlabel('Pixel wise pisition')
    axs[0,0].set_ylabel('Pixel wise pisition')
    axs[0,0].legend()
    
    x,y = ref1_xnorm,ref1_ynorm 
    u_x,u_y=np.mean(x),np.mean(y)
    std_x,std_y=np.std(x),np.std(y)
    mse=((np.subtract(np.transpose([x,y]),[u_x,u_y])**2).sum(axis=1)).mean(axis=None)
    axs[0,1].scatter(x,y, s=0.5)
    axs[0,1].axvline(x=u_x,c='grey', lw=1)
    axs[0,1].axhline(y=u_y,c='grey', lw=1)
    confidence_ellipse(x, y, axs[0,1],n_std=1, 
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, axs[0,1], n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, axs[0,1], n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':')
    axs[0,1].scatter(u_x, u_y, c='red', s=3)
    axs[0,1].set_title('Ref1_norm \n MSE: {:.2g}\n 2D mean: {:.2g}, {:.2g}\n 2D std: {:.2g}, {:.2g}'
                  .format(mse,u_x,u_y,std_x,std_y))
    axs[0,1].set_xlabel('Pixel wise pisition')
    axs[0,1].set_ylabel('Pixel wise pisition')
    axs[0,1].legend()
    
    x,y = ref2_x,ref2_y
    u_x,u_y=np.mean(x),np.mean(y)
    std_x,std_y=np.std(x),np.std(y)
    mse=((np.subtract(np.transpose([x,y]),[u_x,u_y])**2).sum(axis=1)).mean(axis=None)
    axs[1,0].scatter(x,y, s=0.5)
    axs[1,0].axvline(x=u_x,c='grey', lw=1)
    axs[1,0].axhline(y=u_y,c='grey', lw=1)
    confidence_ellipse(x, y, axs[1,0],n_std=1, 
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, axs[1,0], n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, axs[1,0], n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':')
    axs[1,0].scatter(u_x, u_y, c='red', s=3)
    axs[1,0].set_title('Ref2 \n MSE: {:.2g}\n 2D mean: {:.2g}, {:.2g}\n 2D std: {:.2g}, {:.2g}'
                  .format(mse,u_x,u_y,std_x,std_y))
    axs[1,0].set_xlabel('Pixel wise pisition')
    axs[1,0].set_ylabel('Pixel wise pisition')
    axs[1,0].legend()
    
    x,y = ref2_xnorm,ref2_ynorm 
    u_x,u_y=np.mean(x),np.mean(y)
    std_x,std_y=np.std(x),np.std(y)
    mse=((np.subtract(np.transpose([x,y]),[u_x,u_y])**2).sum(axis=1)).mean(axis=None)
    axs[1,1].scatter(x,y, s=0.5)
    axs[1,1].axvline(x=u_x,c='grey', lw=1)
    axs[1,1].axhline(y=u_y,c='grey', lw=1)
    confidence_ellipse(x, y, axs[1,1],n_std=1, 
                       label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(x, y, axs[1,1], n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(x, y, axs[1,1], n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':')
    axs[1,1].scatter(u_x, u_y, c='red', s=3)
    axs[1,1].set_title('Ref2_norm \n MSE: {:.2g}\n 2D mean: {:.2g}, {:.2g}\n 2D std: {:.2g}, {:.2g}'
                  .format(mse,u_x,u_y,std_x,std_y))
    axs[1,1].set_xlabel('Pixel wise pisition')
    axs[1,1].set_ylabel('Pixel wise pisition')
    axs[1,1].legend()
    
     
    fig.suptitle("Comparison index:{} ".format(index+1))
    plt.tight_layout()
    plt.show()
'''        
#%%
# =============================================================================
# # Draw 3D figures
# from scipy.stats import multivariate_normal
# from mpl_toolkits.mplot3d import Axes3D
# X, Y = np.meshgrid(sample_x,sample_y)
# # Multivariate Normal
# mu_x = np.mean(sample_x)
# sigma_x = np.std(sample_x)
# mu_y = np.mean(sample_y)
# sigma_y = np.std(sample_y)
# rv = multivariate_normal([mu_x, mu_y], [[sigma_x, 0], [0, sigma_y]])
# # Probability Density
# pos = np.empty(X.shape + (2,))
# pos[:, :, 0] = X
# pos[:, :, 1] = Y
# pd = rv.pdf(pos)
# # Plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, pd, cmap='viridis', linewidth=0)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Probability Density')
# plt.title("Multivariate Normal Distribution")
# plt.show()
# =============================================================================
