import torch
import numpy as np
from os import path
import os
from PIL import Image
from preprocessing import generate_lidar,kitti_util

device = 'cpu'
N, D_in, H1,H2, D_out = 1242*375, 1, 5, 5 , 1

# x = torch.randn(N, D_in, device=device)
# y = torch.randn(N, D_out, device=device)

GWC_PATH = '/home/maxc303/aer1515/GwcNet/gwc_2012_all'
LIDAR_DISP_PATH = '/home/maxc303/link_aer1515/pseudo_lidar/KITTI/object/training/disp_pl'
MODEL_PATH = './tuning_model_1d.ckpt'

SAVE_PL_PATH = '../result/psmnet_sedge_val_pl/'
CALIB_PATH = '../KITTI/object/training/calib/'

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.ReLU(),
    torch.nn.Linear(H1,H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, D_out),
)
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, D_out),
# )
file1 = open('train.txt', 'r')
lines = file1.read().splitlines()
file1.close()
print(lines)

if(path.exists(MODEL_PATH)):
    model = torch.load(MODEL_PATH)
    model.eval()
if torch.cuda.is_available():
    model = model.cuda()

def load_data(idx, testing = False):
    gwc_img = Image.open(GWC_PATH+'/'+str(idx)+'.png')
    gwc_data = np.asarray(gwc_img) / 256
    img_height = gwc_data.shape[0]
    img_width = gwc_data.shape[1]
    true_img = Image.open(LIDAR_DISP_PATH+'/'+str(idx)+'.png')
    true_data = np.asarray(true_img) / 256
    if(testing):
        mask = np.ones((img_height,img_width))
    else:
        mask = 1 * (true_data != 0)
        mask2 =  1 * (true_data <50)
        mask = np.logical_and(mask,mask2)

    mask = np.reshape(mask,(-1,1))

    x = np.reshape(gwc_data, (-1, 1))
    y = np.reshape(true_data, (-1, 1))

    # print(x.shape)
    # print(y.shape)

    x = np.reshape(x[(mask != 0)],(-1,1)).astype(np.float32)
    y = np.reshape(y[(mask != 0)],(-1,1)).astype(np.float32)
    w = y/192
    # print(x.shape)
    # print(y.shape)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    w = torch.from_numpy(w)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        w = w.cuda()

    return x , y ,img_height, img_width, w



def train(model):
    if torch.cuda.is_available():
        model = model.cuda()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    count = 0
    for epoch in range(20):
        for index, line in enumerate(lines):
            # print(index)
            if (index % 10 == 0):
                x, y, _, _ ,w = load_data(line)
            else:
                x_1, y_1, _, _, w_1 = load_data(line)
                x = torch.cat([x, x_1], dim=0)
                y = torch.cat([y, y_1], dim=0)
                w = torch.cat([w, w_1], dim=0)


            if (index % 10 == 9 or index == len(lines) - 1):
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                    w = w.cuda()
                    # print("Train image idx: "+ line)
                # print(x.shape)
                # print(y.shape)
                # for t in range(10000):
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = model(x)

                # Compute and print loss.
                loss = loss_fn(y_pred, y)
                count += 1
                if (count % 10 == 0):
                    print(count, loss.item())
                    # print("Saved model")
                    # torch.save(model, MODEL_PATH)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
        print("Saved model after epoch# ", epoch)
        torch.save(model, MODEL_PATH)

def test():
    file1 = open('train.txt', 'r')
    lines = file1.read().splitlines()
    file1.close()
    #print(lines)


    # for param in model.parameters():
    #     print(param.data)
    for line in lines:
        print("Img = ",line)
        x,y,h,w,_ = load_data(line,testing = True)
        y_pred = model(x)
        loss_fn = torch.nn.MSELoss(reduction='mean')

        loss = loss_fn(y_pred, y)
        print("loss = ",loss)
        # print(x)
        # print(y_pred)
        y_pred = y_pred.cpu()
        pred_img = np.reshape(y_pred.detach().numpy(),(h,w))
        pred_img = (256*pred_img).astype('uint16')
        # print(pred_img)
        # print(np.max(pred_img))
        im = Image.fromarray(pred_img)
        directory = './disp_1d_w/'
        if not path.exists(directory):
            os.makedirs(directory)
        im.save(directory+line+'.png')

def testone():

# for param in model.parameters():
#     print(param.data)
    line = '000016'
    print("Img = ",line)
    x,y,h,w,_ = load_data(line,testing = True)
    y_pred = model(x)
    loss_fn = torch.nn.MSELoss(reduction='mean')

    loss = loss_fn(w*y_pred, w*y)
    print("loss = ",loss)
    # print(x)
    # print(y_pred)
    y_pred = y_pred.cpu()
    pred_img = np.reshape(y_pred.detach().numpy(),(h,w))
    disp = pred_img
    pred_img = (256*pred_img).astype('uint16')
    # print(pred_img)
    # print(np.max(pred_img))
    im = Image.fromarray(pred_img)
    directory = './disp_1d_w/'
    if not path.exists(directory):
        os.makedirs(directory)
    im.save(directory+line+'.png')

    calib_file = CALIB_PATH + line + '.txt'
    calib = kitti_util.Calibration(calib_file)
    max_high = 1
    lidar = generate_lidar.project_disp_to_points(calib, disp, max_high)
    lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
    lidar = lidar.astype(np.float32)
    lidar.tofile('{}/{}.bin'.format('./pl/', line))


if __name__ == '__main__':
    #train(model)
    test()
    #testone()
    #load_data('000017', testing=False)
    print("main")
