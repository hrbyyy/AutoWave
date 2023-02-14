import torch
from utils import gain_coef_v2, normalization,process_coef,shape_tune

def train(net,xfm,ifm,device,trainloader,ratio_calculator,epoch,losst,optimizer,len_list):
  
    rloss = 0  
    running_loss = 0.0  
    net.train()
    criterion = torch.nn.MSELoss(reduction='none')
   
    for i, data in enumerate(trainloader):

        inputs = data[0][:,:1,:]
        targets = inputs 
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
      
        coef=gain_coef_v2(inputs,xfm)
        normalized_coef = normalization(coef)
        trec_inputs = net(inputs) 
        new_coef = process_coef(normalized_coef, len_list)  
        rec_coef=shape_tune(new_coef,len_list) 
        frec_inputs=ifm(rec_coef)

        ratio = ratio_calculator(trec_inputs, frec_inputs)
        ratio=torch.unsqueeze(ratio,dim=1)
       
        loss =(1-ratio)* criterion(trec_inputs, targets) + ratio * criterion(frec_inputs, targets)  
        loss=torch.sum(loss)
        
        loss.backward()
        
        optimizer.step()
       
        rloss += loss.item()

        running_loss += loss.item()
        if i % 20 == 19:  # print every  20 steps
            print('[%d, %5d] train loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0
    losst.append(rloss)  
    torch.cuda.empty_cache()
    return losst


def val(net,xfm,ifm,window,ratio_calculator,device,val_loader,epoch,ep,lossv,vloss,counter,patience,threshold,batchsize,len_list,outp,optimizer):
    num=len(val_loader.dataset)
    err_array = torch.empty(size=(num, 1, window), device=device)
    label_array = torch.empty(size=(num, 1), device=device)
    real_seq = torch.empty(size=(num, 1, window), device=device)
    rec_seq = torch.empty(size=(num, 1, window), device=device)
    ratio_array = torch.empty(size=(num, 1), device=device)
 
    net.eval()
    val_loss = 0
    vtotal_l = 0
 
    criterion = torch.nn.MSELoss(reduction='none')
    criterion2 = torch.nn.MSELoss(reduction='none') 
   
    with torch.no_grad():
       
        for batch_idx, data in enumerate(val_loader):
            start_posi = batch_idx * batchsize
            end_posi = min(num,start_posi+batchsize)
           
            label=data[1].to(device)
            inputs = data[0][:, :1, :]
            targets = inputs  
            inputs = inputs.to(device)
            targets = targets.to(device)

           
            coef = gain_coef_v2(inputs, xfm)
            normalized_coef = normalization(coef)
            trec_inputs = net(inputs) 
            new_coef = process_coef(normalized_coef, len_list)  
            rec_coef = shape_tune(new_coef, len_list) 
            frec_inputs = ifm(rec_coef)
            ratio = ratio_calculator(trec_inputs, frec_inputs)
           
            ratio = torch.unsqueeze(ratio, dim=1)
           
            loss = (1 - ratio) * criterion(trec_inputs, targets) + ratio * criterion(frec_inputs,targets) 
            loss = torch.sum(loss)
            err=criterion2(trec_inputs,targets)
           
            val_loss += loss.item() 
            vtotal_l += loss.item()
            label_array[start_posi:end_posi, :] = label
            err_array[start_posi:end_posi, :, :] = err
            real_seq[start_posi:end_posi, :, :] = inputs
            rec_seq[start_posi:end_posi, :, :] = trec_inputs
            ratio_array[start_posi:end_posi, :] = ratio
            if batch_idx % 20 == 19:
                print('[%d, %5d] val loss: %.3f' %
                      (epoch + 1, batch_idx + 1, val_loss / 20)) 
                val_loss = 0
   
    pred_error=err_array.detach().cpu().numpy()
    real_label=label_array.detach().cpu().numpy()
    real_seq = real_seq.detach().cpu().numpy()
    rec_seq = rec_seq.detach().cpu().numpy()
    ratio_array = ratio_array.detach().cpu().numpy()
    print('%d val loss: %.3f' %(epoch + 1, vtotal_l))

    lossv.append(vtotal_l)  

    if vtotal_l > vloss - threshold:
        counter += 1
    else:      
        counter = 0
        vloss = vtotal_l
        state = {
            'net': net.state_dict(),
            'ratio_calculator':ratio_calculator.state_dict(),
            'val_loss': vtotal_l,
         
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            "valerr_vectors": pred_error,
            'val_label':real_label,
            'real_seqs': real_seq,
            'rec_seqs': rec_seq,
            'ratio_array': ratio_array
          
        }
        ep = epoch
   
        torch.save(state, outp + 'model.pth')

    print(counter)
    torch.cuda.empty_cache()
    return lossv,vloss,ep,counter
