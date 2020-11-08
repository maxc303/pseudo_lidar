import torch
import numpy as np
from os import path
from PIL import Image
device = 'cpu'
N, D_in, H1,H2, D_out = 1242*375, 4, 10, 4, 1

# x = torch.randn(N, D_in, device=device)
# y = torch.randn(N, D_out, device=device)

GWC_PATH = '/home/maxc303/aer1515/GwcNet/gwc_2012_all'
LIDAR_DISP_PATH = '/home/maxc303/link_aer1515/pseudo_lidar/KITTI/object/training/disp_pl'
MODEL_PATH = './tuning_model_4d_inv.ckpt'


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
    row_num = np.arange(img_height)
    col_num = np.arange(img_width)
    row_num = np.tile(row_num,(img_width,1)).transpose()
    col_num = np.tile(col_num,(img_height,1))
    # print(row_num)
    # print(col_num)
    # print(row_num.shape)
    # print(col_num.shape)

    if(testing):
        mask = np.ones((img_height,img_width))
    else:
        mask = 1 * (true_data != 0)

    mask = np.reshape(mask,(-1,1))

    x = np.reshape(gwc_data, (-1, 1))
    y = np.reshape(true_data, (-1, 1))
    row_num = np.reshape(row_num,(-1, 1))/img_height
    col_num = np.reshape(col_num,(-1, 1))/img_width
    #x= (x-np.mean(x))/np.std(x)

    # row_num = np.reshape(row_num[(mask != 0)],(-1,1)).astype(np.float32)
    # col_num = np.reshape(col_num[(mask != 0)],(-1,1)).astype(np.float32)
    row_num = row_num[(mask != 0)].astype(np.float32)
    col_num = col_num[(mask != 0)].astype(np.float32)
    x = x[(mask != 0)].astype(np.float32)
    #x = np.reshape(x[(mask != 0)],(-1,1)).astype(np.float32)
    x_d = 1/x
    x = np.stack((x,x_d,row_num,col_num),axis=-1)
    #print(x.shape)
    y = np.reshape(y[(mask != 0)],(-1,1)).astype(np.float32)
    #print(y.shape)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    return x , y ,img_height, img_width



def train(model):
    if torch.cuda.is_available():
        model = model.cuda()
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    count = 0;
    for epoch in range(100):
        for index,line in enumerate(lines):
            #print(index)
            if(index%20==0):
                x,y,_,_ = load_data(line)
            else:
                x_1,y_1,_,_ = load_data(line)
                x = torch.cat([x,x_1], dim=0)
                y = torch.cat([y,y_1], dim=0)

            if (index % 10 == 9 or index == len(lines)-1):
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                    # print("Train image idx: "+ line)
                # print(x.shape)
                # print(y.shape)
                #for t in range(10000):
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = model(x)

                # Compute and print loss.
                loss = loss_fn(y_pred, y)
                count += 1
                if(count%10==0):
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
        x,y,h,w = load_data(line,testing = True)
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
        im.save('./disp_4d/'+line+'.png')



if __name__ == '__main__':
   # train(model)
    test()
   # load_data('000017', testing=False)
    print("main")
