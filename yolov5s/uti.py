import torch

def make_yolo_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()  ### [1,1,ny,nx,2)] --->> 1  1 for batch + anchor



if __name__ == '__main__':
    grid = make_yolo_grid()
    print(grid[0,0,2,3,:])  # [3, 2]