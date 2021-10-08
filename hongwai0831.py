from ctypes import *
from matplotlib import pyplot as plt
import time
import numpy as np
from scipy import ndimage
import subprocess
import cv2

mlx90640 = cdll.LoadLibrary('./libmlx90640.so')

cap=cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

def base(cmd):
    if subprocess.call(cmd, shell=True):
        raise Exception("{} failed!".format(cmd))
 
# mlx90640 will output 32*24 temperature array with chess mode
frame=(c_float*768)()

mlx_shape = (24,32) # mlx90640 shape

mlx_interp_val = 10 # interpolate # on each dimension
mlx_interp_shape = (mlx_shape[0]*mlx_interp_val,
                    mlx_shape[1]*mlx_interp_val) # new shape

fig = plt.figure(figsize=(8,6)) # start figure
ax = fig.add_subplot(111) # add subplot
fig.subplots_adjust(0.1,0.05,0.95,0.95) # get rid of unnecessary padding
therm1 = ax.imshow(np.zeros(mlx_interp_shape),interpolation='none',
                   cmap=plt.cm.bwr,vmin=25,vmax=45) # preemptive image

cbar = fig.colorbar(therm1) # setup colorbar
cbar.set_label('Temperature [$^{\circ}$C]',fontsize=14) # colorbar label

fig.canvas.draw() # draw figure to copy background
ax_background = fig.canvas.copy_from_bbox(ax.bbox) # copy background
fig.show() # show the figure before blitting

def plot_update():
    fig.canvas.restore_region(ax_background) # restore background
    mlx90640.get_mlxFrame(frame) # read mlx90640
    print("frame=%.2f"%(np.max(np.array(frame))))

    data_array = np.fliplr(np.reshape(frame,mlx_shape)) # reshape, flip data

    data_array = ndimage.zoom(data_array,mlx_interp_val) # interpolate
    print("maxTemp=%.2f"%(np.max(data_array)))

    therm1.set_array(data_array) # set data,更新图像数据而不生成新图

    therm1.set_clim(vmin=np.min(data_array),vmax=np.max(data_array)) # set bounds
    cbar.on_mappable_changed(therm1) # update colorbar range

    ax.draw_artist(therm1) # draw new thermal image  
    
    fig.canvas.blit(ax.bbox) # draw background
    fig.canvas.flush_events() # show the new image

    if np.max(data_array) > 45.0:
        plt.savefig('temp.jpg')

    return

t_array = []       
while True:
    t1 = time.monotonic() # for determining frame rate
    ret,frameImg=cap.read()
    
    try:
        plot_update() # update plot
        cv2.imshow("frameImg",frameImg)
        cv2.waitKey(10)
    except:
        break

    # approximating frame rate
    t_array.append(time.monotonic()-t1)

    if len(t_array)>10:
        t_array = t_array[1:] # recent times for frame rate approx

    print('Frame Rate: {0:2.1f}fps'.format(len(t_array)/np.sum(t_array)))    

cap.release()#释放摄像头
cv2.destroyAllWindows()#销毁窗口

    

