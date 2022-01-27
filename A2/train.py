import torch
from utils import gain_coef_v2, normalization,process_coef,shape_tune

def train(net, xfm, ifm, device, trainloader, ratio_calculator, epoch, losst, optimizer, len_list):
    #global losst
    rloss = 0  # 记录每一epoch的loss,而running_loss每20批平均后置为0
    running_loss = 0.0  # 按批求每个样本的平均loss,每批后置位0
    net.train()
    ratio_calculator.train()
    criterion = torch.nn.MSELoss(reduction='none')
    # criterion2 = torch.nn.MSELoss(reduction='none')  # 也要保存测试集的error vector,便于预测anomaly score
    # optimizer = optim.Adam(net.parameters(), lr=0.002)
    # with open(inp+indicator+'loader.pkl','rb') as fin:
    #     trainloader=pickle.load(fin)
    for i, data in enumerate(trainloader):#data[1]时 seq label(b,1)

        inputs = data[0][:,:1,:]
        targets = inputs #先尝试data[1]＝data[0]，data[1]保存原ts,label位于最后一个feature
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        # 每次喂入数据前，都需要将梯度清零
        coef=gain_coef_v2(inputs,xfm)
        normalized_coef=normalization(coef)
        trec_inputs = net(inputs)  # 批训练，则输出size为(batch_size,#channel,timestep) input为（batch_size,#feature,timestep）
        new_coef=process_coef(normalized_coef,len_list) #(b,1,t)
        rec_coef=shape_tune(new_coef,len_list)  #将新的系数整理成ifm可用的形式(tuple)
        frec_inputs=ifm(rec_coef)

        ratio = ratio_calculator(trec_inputs, frec_inputs)
        # ratio = torch.mean(ratio)
        loss_t= criterion(trec_inputs, targets)
        loss_f=criterion(frec_inputs, targets)
        ratio=torch.unsqueeze(ratio,dim=1)
        # print(ratio)

        loss=(1-ratio)*loss_t+ratio*loss_f
        loss=torch.sum(loss)
        # loss = criterion(trec_inputs, targets) + ratio * criterion(frec_inputs, targets)  # 得到的是GPU上的floattensor
        # print(loss_t.item(),loss_f.item(),ratio.item())

        # 计算loss
        loss.backward()
        # 传回反向梯度
        optimizer.step()

        # 梯度传回，利用优化器将参数更新
        rloss += loss.item()

        running_loss += loss.item()
        if i % 20 == 19:  # print every  20 steps
            print('[%d, %5d] train loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
    losst.append(rloss)  # 记录每轮的loss
    torch.cuda.empty_cache()
    return losst


def val(net,xfm,ifm,window,ratio_calculator,device,val_loader,epoch,ep,lossv,vloss,counter,patience,threshold,batchsize,len_list,outp,optimizer):
    num=len(val_loader.dataset)
    err_array = torch.empty(size=(num, 1, window), device=device)
    label_array = torch.empty(size=(num, window), device=device)
    real_seq = torch.empty(size=(num, 1, window), device=device)
    rec_seq = torch.empty(size=(num, 1, window), device=device)
    ratio_array=torch.empty(size=(num, 1), device=device)
    # total_ratio=0
    net.eval()
    ratio_calculator.eval()
    val_loss = 0
    vtotal_l = 0
    # correct = 0
    # total = 0

    criterion = torch.nn.MSELoss(reduction='none')
    criterion2 = torch.nn.MSELoss(reduction='none')  # 也要保存测试集的error vector,便于预测anomaly score
    # optimizer = optim.Adam(net.parameters(), lr=0.002)
    # batchnumber=0  #给出的threshold为每个bathch的threshold,故需记录batch数
    with torch.no_grad():
        # 因为是测试，因此禁止梯度
        for batch_idx, data in enumerate(val_loader):
            start_posi = batch_idx * batchsize
            end_posi = min(num,start_posi+batchsize)
            #取dataset,gain_coef_v2,送入网络。
            # batchnumber+=1
            # label = targets[:, :,-1, :].squeeze().tolist()
            # label=data[0][:,1,:].to(device)  #(b,t)形式
            label=data[0][:,1,:].to(device)
            inputs = data[0][:, :1, :]
            targets = inputs  # 先尝试data[1]＝data[0]，data[1]保存原ts,label位于最后一个feature
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 每次喂入数据前，都需要将梯度清零
            coef = gain_coef_v2(inputs, xfm)
            normalized_coef = normalization(coef)
            trec_inputs = net(inputs)  # 批训练，则输出size为(batch_size,#channel,timestep) input为（batch_size,#feature,timestep）
            new_coef = process_coef(normalized_coef, len_list)
            rec_coef = shape_tune(new_coef, len_list)  # 将新的系数整理成ifm可用的形式(tuple)
            frec_inputs = ifm(rec_coef)
            ratio = ratio_calculator(trec_inputs, frec_inputs)
            # ratio_tensor=torch.mean(ratio,dim=0)
            # ratio = torch.mean(ratio)
            # total_ratio+=ratio
            ratio = torch.unsqueeze(ratio, dim=1)

            loss = (1-ratio)*criterion(trec_inputs, targets) + ratio * criterion(frec_inputs, targets)  # 得到的是GPU上的floattensor

            loss = torch.sum(loss)

            err=criterion2(trec_inputs,targets)
            # err=err.detach().cpu().numpy()
            # err=err.squeeze(2)
            # print(err)
            # print(err.shape)        #（128，28，1）
            # exit()
            # errlist.append(err)
            val_loss += loss.item()  # val_loss每5step清零。vtotal_l每轮清零，和全局变量比较。
            vtotal_l += loss.item()
            label_array[start_posi:end_posi, :] = label
            err_array[start_posi:end_posi, :, :] = err
            real_seq[start_posi:end_posi, :, :] = inputs
            rec_seq[start_posi:end_posi, :, :] = trec_inputs
            ratio_array[start_posi:end_posi,:]=ratio
            if batch_idx % 20 == 19:
                print('[%d, %5d] val loss: %.3f' %
                      (epoch + 1, batch_idx + 1, val_loss / 20))  # 每20步（step)打印每步平均loss
                val_loss = 0
    # mean_ratio=total_ratio/len(val_loader)

    # acc = 100. * correct / total
    # print("Now acc is {}".format(acc),'Now val loss is {}'.format(vtotal_l))  #打印每个epoch的acc和val loss
    pred_error=err_array.detach().cpu().numpy()
    real_label=label_array.detach().cpu().numpy()
    real_seq = real_seq.detach().cpu().numpy()
    rec_seq = rec_seq.detach().cpu().numpy()
    ratio_array=ratio_array.detach().cpu().numpy()
    print('%d val loss: %.3f' %(epoch + 1, vtotal_l))

    lossv.append(vtotal_l)  # 记录每轮val loss.不能放在函数末尾，避免early stopping退出时未记录本轮vtotal_l

    if vtotal_l > vloss - threshold:
        counter += 1
    else:      #只要出现比vloss小，就保存，
                              # 避免出现vtotal_l不满足earlydropping条件而没有保存模型的情况，
                              # 通过文件名称覆盖得到一个model,来源有两种，或earlystopping,或最小vloss。
        counter = 0
        vloss = vtotal_l
        state = {
            'net': net.state_dict(),
            'net_ratio': ratio_calculator.state_dict(),
            'val_loss': vtotal_l,
            # 'acc':acc,
            'epoch': epoch + 1,  # 存储的轮数和显示的轮数统一！！
            'optimizer': optimizer.state_dict(),
            "valerr_vectors": pred_error,
            'val_label':real_label,
            'real_seqs': real_seq,
            'rec_seqs': rec_seq,
            'ratio_array':ratio_array
            # 'mean_ratio':mean_ratio

        }
        ep = epoch
        # torch.save(state, outp + 'epoch'+str(epoch)+'model.pth')  #val loss普遍成减小趋势，epoch已在state中保存，此处只需同名覆盖
        torch.save(state, outp + 'model.pth')

    print(counter)
    torch.cuda.empty_cache()
    return lossv,vloss,ep,counter