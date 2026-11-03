import numpy as np

def moving_rectangle_sequence(num_frames=45, height=140, width=220):
    frames = np.zeros((num_frames,height,width,3), dtype=np.uint8)
    for i in range(num_frames):
        x,y = 20+2*i, 35+i
        frames[i,y:min(y+18,height),x:min(x+32,width),:] = 230
    return frames
