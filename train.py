import os
import sys
from pathlib import Path

from tqdm import tqdm
import torch
import torch_geometric as tg
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter

import network.MIPDataset as MIPDataset
import network.utils as utils
from network.gnn_policy import GNNPolicy_ranking, GNNPolicy_cl
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import DotProductSimilarity

working_directory = Path(".").resolve()
# probably need to be saving + reading normalization constants
def pad_by_batch(tensor, x_vars_batch, value=0):
    """
    Organizes and pads the tensor based on batch indices given in x_vars_batch.

    Parameters:
    - tensor: The input tensor containing values.
    - x_vars_batch: A tensor containing batch indices for each value in tensor.
    - value: The value used for padding.

    Returns:
    - A 2D tensor with each row being a padded sequence.
    """
    unique_batches = torch.unique(x_vars_batch)
    grouped_tensors = [tensor[x_vars_batch == batch_idx] for batch_idx in unique_batches]

    max_length = max([len(t) for t in grouped_tensors])

    padded_tensors = []
    for t in grouped_tensors:
        padding_size = max_length - len(t)
        pad = torch.full((padding_size,), value, device=tensor.device, dtype=tensor.dtype)
        padded_tensor = torch.cat([t, pad])
        padded_tensors.append(padded_tensor)
        
    return torch.stack(padded_tensors), max_length

def fit_distribution(
    mip_distribution="facilities_train_c100_f100",
    train_list="generated/instances/cauctions/train_100_500/",
    valid_list="generated/instances/cauctions/valid_100_500/",
    method="cl",
    batch_size=32,
    num_epochs=100,
    learning_rate=0.001,
    weight_decay=0.00005,
    activation="relu",
    seed=0,
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    ):
    """
    # this is from bad filenaming, ideally sample_backdoors will be fixed now so future samplings will be correct
    python -m scripts.train_model1_rank_estimator fit_distribution --mip_distribution=gisp_train_n175_p0.3_c1_w100_a0.25  --mip_distribution_dir=/data/aaron/mip_instances/generated/instances/gisp/gisp_train_n175_p0.3_c1_w100_a0.25 --batch_size=64 --num_epochs=200 --learning_rate=0.001 --weight_decay=0.001 --activation=relu --seed=457611 --pct_backdoor=0.03
    mip_distribution_dir should be relative to mip_distribution
    """

    torch.manual_seed(seed)

    if method == "ranking":
        # dataset loading
        train_data = MIPDataset.BackdoorDatasetRanking(
            root=os.path.join(working_directory, "backdoor_dataset/train"),
            instance_list = train_list,
            mip_distribution_name=mip_distribution,
        )
    
        valid_data = MIPDataset.BackdoorDatasetRanking(
            root=os.path.join(working_directory, "backdoor_dataset/valid"),
            instance_list = valid_list,
            mip_distribution_name=mip_distribution,
        )
    
        train_dataset = MIPDataset.RankingBackdoorDataset(train_data)
        valid_dataset = MIPDataset.RankingBackdoorDataset(valid_data)
    
        writer = SummaryWriter('log/ranking')
        
        train_loader = tg.loader.DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x_vars', 'x_cons', 'candidate_backdoor'], shuffle=True)
        val_loader = tg.loader.DataLoader(valid_dataset, batch_size=batch_size, follow_batch=['x_vars', 'x_cons', 'candidate_backdoor'], shuffle=False)
        
        value_estimator = GNNPolicy_ranking().to(device)
        value_optimizer = torch.optim.Adam(value_estimator.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optimizer, 5, eta_min=learning_rate/10, verbose=False)
    
        # get data loader with all data call it value_data_loader
        # train value estimator
        value_estimator.train()
        ranking_loss_fn = torch.nn.MarginRankingLoss(margin=0.1)
        best_valid_loss = float("inf")
        last_improved = 0
        for epoch in tqdm(range(num_epochs), desc="training"):
            train_loss = 0
            # for each mip instance maintain runtime of fastest solver
            for ranking_data in tqdm(train_loader, desc="training"):
                value_optimizer.zero_grad()
                pair_outputs = []
                for data in ranking_data:
    
                    data = data.to(device)
    
                    # need float here for some reason, the predictive model should also output float
                    data.candidate_backdoor = data.candidate_backdoor.float()
    
                    out = value_estimator(data.x_cons,
                                          data.edge_index_cons_to_vars,
                                          data.edge_attr,
                                          torch.hstack([data.x_vars, data.candidate_backdoor.unsqueeze(-1)]),
                                          data.x_vars_batch)

                    pair_outputs.append(out)
           
                predictions = [out["output"] for out in pair_outputs]
                first_higher = ranking_data[0].solve_time > ranking_data[1].solve_time
                second_higher = ranking_data[0].solve_time <= ranking_data[1].solve_time
     
                target_values = (1 * first_higher) + (-1 * second_higher)
                loss = ranking_loss_fn(predictions[0].squeeze(), predictions[1].squeeze(), target_values)
    
                loss.backward()
                value_optimizer.step()
                train_loss += utils.to_numpy(loss.detach())
         
            # print('Training Loss: ' + str(train_loss))
            writer.add_scalar('Training Loss', train_loss, epoch)
    
            value_estimator.eval()
            valid_loss = 0
            with torch.no_grad():
                for ranking_data in tqdm(val_loader, desc="validating"):
                    pair_outputs = []
                    for data in ranking_data:
    
                        data = data.to(device)
                        data.candidate_backdoor = data.candidate_backdoor.float()
    
                        out = value_estimator(data.x_cons,
                                            data.edge_index_cons_to_vars,
                                            data.edge_attr,
                                            torch.hstack([data.x_vars, data.candidate_backdoor.unsqueeze(-1)]),
                                            data.x_vars_batch)
                        pair_outputs.append(out)
            
                    predictions = [out["output"] for out in pair_outputs]
                    first_higher = ranking_data[0].solve_time > ranking_data[1].solve_time
                    second_higher = ranking_data[0].solve_time <= ranking_data[1].solve_time
        
                    target_values = (1 * first_higher) + (-1 * second_higher)
                    loss = ranking_loss_fn(predictions[0].squeeze(), predictions[1].squeeze(), target_values)
                    valid_loss += utils.to_numpy(loss.detach())
            # print(f'Validation Loss: {valid_loss}')
            writer.add_scalar('Validation Loss', valid_loss, epoch)
            value_estimator.train()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                last_improved = epoch
                torch.save(value_estimator.state_dict(), os.path.join(working_directory, "model_ranking.pt"))
            elif epoch - last_improved > 20:
                learning_rate /= 2
                print(f"Adjusting the learning rate to {learning_rate}")
                optimizer = torch.optim.Adam(value_estimator.parameters(), lr=learning_rate, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=learning_rate/10, verbose=False)
                last_improved = epoch
    
        writer.close()
        # torch.save(value_estimator.state_dict(), os.path.join(working_directory, "model_ranking.pt"))
    elif method == "cl":
        # dataset loading
        train_dataset = MIPDataset.BackdoorDatasetCL(
            root=os.path.join(working_directory, "backdoor_dataset/train"),
            instance_list = train_list,
            mip_distribution_name=mip_distribution,
        )
    
        valid_dataset = MIPDataset.BackdoorDatasetCL(
            root=os.path.join(working_directory, "backdoor_dataset/valid"),
            instance_list = valid_list,
            mip_distribution_name=mip_distribution,
        )
    
        config_str = f'bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay}'
        writer = SummaryWriter(f'log/cl')
        
        train_loader = tg.loader.DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x_vars', 'x_cons', 'candidate_backdoor'], shuffle=True)
        val_loader = tg.loader.DataLoader(valid_dataset, batch_size=batch_size, follow_batch=['x_vars', 'x_cons', 'candidate_backdoor'], shuffle=False)
       
        value_estimator = GNNPolicy_cl().to(device)
        value_optimizer = torch.optim.Adam(value_estimator.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optimizer, 5, eta_min=learning_rate/10, verbose=False)
    
        value_estimator.train()
        infoNCE_loss_function = losses.NTXentLoss(temperature=0.07).to(device)
        best_valid_loss = float("inf")
        last_improved = 0
        for epoch in tqdm(range(num_epochs), desc="training"):
            train_loss = 0
            # for each mip instance maintain runtime of fastest solver
            for data in tqdm(train_loader, desc="training"):
                value_optimizer.zero_grad()
                data = data.to(device)
    
                # need float here for some reason, the predictive model should also output float
                data.candidate_backdoor = data.candidate_backdoor.float()
    
                # out = value_estimator(data.x_cons,
                #                     data.edge_index_cons_to_vars,
                #                     data.edge_attr,
                #                     torch.hstack([data.x_vars, data.candidate_backdoor.unsqueeze(-1)]),
                #                     data.x_vars_batch)
    
                out = value_estimator(data.x_cons,
                                    data.edge_index_cons_to_vars,
                                    data.edge_attr,
                                    data.x_vars,
                                    data.x_vars_batch)
                
                pred, max_length = pad_by_batch(out["output"], data.x_vars_batch)
                embeddings = torch.sigmoid(pred)
                anchor_positive = []
                anchor_negative = []
                positive_idx = []
                negative_idx = []
                total_sample = len(data)
    
                for i in range(len(data)):
                    for j in range(len(data.pos_sample[i])):
                        anchor_positive.append(i)
                        positive_idx.append(total_sample)
                        tensor = torch.tensor(data.pos_sample[i][j])
                        padding_size = max_length - len(tensor)
                        pad = torch.full((padding_size,), 0, device=tensor.device, dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, pad]).unsqueeze(0)
                        embeddings = torch.cat([embeddings, padded_tensor.to(device)])
                        total_sample += 1
                    for j in range(len(data.neg_sample[i])):
                        anchor_negative.append(i)
                        negative_idx.append(total_sample)
                        tensor = torch.tensor(data.neg_sample[i][j])
                        padding_size = max_length - len(tensor)
                        pad = torch.full((padding_size,), 0, device=tensor.device, dtype=tensor.dtype)
                        padded_tensor = torch.cat([tensor, pad]).unsqueeze(0)
                        embeddings = torch.cat([embeddings, padded_tensor.to(device)])
                        total_sample += 1
    
                triplets = (torch.tensor(anchor_positive).to(device), torch.tensor(positive_idx).to(device), torch.tensor(anchor_negative).to(device), torch.tensor(negative_idx).to(device))
                loss = infoNCE_loss_function(embeddings, indices_tuple = triplets)
                # print(loss)
    
                loss.backward()
                value_optimizer.step()
                train_loss += utils.to_numpy(loss.detach())
         
            # print('Training Loss: ' + str(train_loss))
            writer.add_scalar('Training Loss', train_loss, epoch)
    
            value_estimator.eval()
            valid_loss = 0
            with torch.no_grad():
                for data in tqdm(val_loader, desc="validating"):
                    data = data.to(device)
                    data.candidate_backdoor = data.candidate_backdoor.float()
    
                    out = value_estimator(data.x_cons,
                                    data.edge_index_cons_to_vars,
                                    data.edge_attr,
                                    data.x_vars,
                                    data.x_vars_batch)
                    pred, max_length = pad_by_batch(out["output"], data.x_vars_batch)
                    embeddings = torch.sigmoid(pred)
                    anchor_positive = []
                    anchor_negative = []
                    positive_idx = []
                    negative_idx = []
                    total_sample = len(data)
            
                    for i in range(len(data)):
                        for j in range(len(data.pos_sample[i])):
                            anchor_positive.append(i)
                            positive_idx.append(total_sample)
                            tensor = torch.tensor(data.pos_sample[i][j])
                            padding_size = max_length - len(tensor)
                            pad = torch.full((padding_size,), 0, device=tensor.device, dtype=tensor.dtype)
                            padded_tensor = torch.cat([tensor, pad]).unsqueeze(0)
                            embeddings = torch.cat([embeddings, padded_tensor.to(device)])
                            total_sample += 1
                        for j in range(len(data.neg_sample[i])):
                            anchor_negative.append(i)
                            negative_idx.append(total_sample)
                            tensor = torch.tensor(data.neg_sample[i][j])
                            padding_size = max_length - len(tensor)
                            pad = torch.full((padding_size,), 0, device=tensor.device, dtype=tensor.dtype)
                            padded_tensor = torch.cat([tensor, pad]).unsqueeze(0)
                            embeddings = torch.cat([embeddings, padded_tensor.to(device)])
                            total_sample += 1
    
                    triplets = (torch.tensor(anchor_positive).to(device), torch.tensor(positive_idx).to(device), torch.tensor(anchor_negative).to(device), torch.tensor(negative_idx).to(device))
                    loss = infoNCE_loss_function(embeddings, indices_tuple = triplets)
                    # print(loss)
                    valid_loss += utils.to_numpy(loss.detach())
            # print(f'Validation Loss: {valid_loss}')
            writer.add_scalar('Validation Loss', valid_loss, epoch)
            value_estimator.train()
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                last_improved = epoch
                torch.save(value_estimator.state_dict(), os.path.join(working_directory, "model_cl.pt".format(epoch)))
            elif epoch - last_improved > 50:
                learning_rate /= 2
                print(f"Adjusting the learning rate to {learning_rate}")
                optimizer = torch.optim.Adam(value_estimator.parameters(), lr=learning_rate, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=learning_rate/10, verbose=False)
                last_improved = epoch
    
        writer.close()
    
        # torch.save(value_estimator.state_dict(), os.path.join(working_directory, "model_cl.pt"))
    elif method == "classifier":
        # dataset loading
        train_dataset = MIPDataset.BackdoorDatasetClassifier(
            root=os.path.join(working_directory, "backdoor_dataset/train"),
            instance_list = train_list,
            mip_distribution_name=mip_distribution,
        )
    
        valid_dataset = MIPDataset.BackdoorDatasetClassifier(
            root=os.path.join(working_directory, "backdoor_dataset/valid"),
            instance_list = valid_list,
            mip_distribution_name=mip_distribution,
        )
    
        # writer = SummaryWriter('log/classifier')
        
        train_loader = tg.loader.DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x_vars', 'x_cons', 'candidate_backdoor'], shuffle=True)
        val_loader = tg.loader.DataLoader(valid_dataset, batch_size=batch_size, follow_batch=['x_vars', 'x_cons', 'candidate_backdoor'], shuffle=False)
        
        value_estimator = GNNPolicy_ranking().to(device)
        value_optimizer = torch.optim.Adam(value_estimator.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(value_optimizer, 5, eta_min=learning_rate/10, verbose=False)
    
        # get data loader with all data call it value_data_loader
        # train value estimator
        value_estimator.train()
        best_valid_loss = float("inf")
        last_improved = 0
        for epoch in tqdm(range(num_epochs), desc="training"):
            train_loss = 0
            # for each mip instance maintain runtime of fastest solver
            for data in tqdm(train_loader, desc="training"):
                value_optimizer.zero_grad()
                data = data.to(device)
    
                # need float here for some reason, the predictive model should also output float
                data.candidate_backdoor = data.candidate_backdoor.float()
                # print(data.candidate_backdoor.shape)
    
                out = value_estimator(data.x_cons,
                                            data.edge_index_cons_to_vars,
                                            data.edge_attr,
                                            torch.hstack([data.x_vars, data.candidate_backdoor.unsqueeze(-1)]),
                                            data.x_vars_batch)

                # print(output)
                loss = F.binary_cross_entropy_with_logits(out["output"].squeeze(), data.use.float())
    
                loss.backward()
                value_optimizer.step()
                train_loss += utils.to_numpy(loss.detach())
         
            print('Training Loss: ' + str(train_loss))
            # writer.add_scalar('Training Loss', train_loss, epoch)
    
            value_estimator.eval()
            valid_loss = 0
            with torch.no_grad():
                for data in tqdm(val_loader, desc="validating"):
                    value_optimizer.zero_grad()
                    data = data.to(device)
        
                    # need float here for some reason, the predictive model should also output float
                    data.candidate_backdoor = data.candidate_backdoor.float()
    
                    out = value_estimator(data.x_cons,
                                            data.edge_index_cons_to_vars,
                                            data.edge_attr,
                                            torch.hstack([data.x_vars, data.candidate_backdoor.unsqueeze(-1)]),
                                            data.x_vars_batch)

                # print(output)
                    loss = F.binary_cross_entropy_with_logits(out["output"].squeeze(), data.use.float())
                    valid_loss += utils.to_numpy(loss.detach())
            # print(f'Validation Loss: {valid_loss}')
            # writer.add_scalar('Validation Loss', valid_loss, epoch)
            value_estimator.train()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                last_improved = epoch
                torch.save(value_estimator.state_dict(), os.path.join(working_directory, "model_classifier.pt"))
            elif epoch - last_improved > 100:
                learning_rate /= 2
                print(f"Adjusting the learning rate to {learning_rate}")
                optimizer = torch.optim.Adam(value_estimator.parameters(), lr=learning_rate, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=learning_rate/10, verbose=False)
                last_improved = epoch
    
        # writer.close()
        # torch.save(value_estimator.state_dict(), os.path.join(working_directory, "model_classifier.pt"))

if __name__ == "__main__":
    methods = ["ranking"]
    batch_sizes = [128]
    learning_rates = [0.0005]
    weight_decays = [0]

    # fit_distribution(
    #                 mip_distribution="is", 
    #             train_list="/home/azureuser/workspace/train.txt",
    #                 valid_list="/home/azureuser/workspace/valid.txt",
    #                 num_epochs=100,
    #                 method="cl",
    #                 batch_size=32,
    #                 learning_rate=0.00005,
    #                 weight_decay=0.01
    #             )
    
    # for method in methods:
    #     for batch_size in batch_sizes:
    #         for learning_rate in learning_rates:
    #             for weight_decay in weight_decays:
    #                 fit_distribution(
    #                     mip_distribution="sc", 
    #                 train_list="/home/azureuser/workspace/train.txt",
    #                     valid_list="/home/azureuser/workspace/valid.txt",
    #                     num_epochs=50,
    #                     method=method,
    #                     batch_size=batch_size,
    #                     learning_rate=learning_rate,
    #                     weight_decay=weight_decay
    #                 )


    # fit_distribution(
    #                 mip_distribution="sc", 
    #                 train_list="/home/azureuser/workspace/train.txt",
    #                     valid_list="/home/azureuser/workspace/valid.txt",
    #                     num_epochs=100,
    #                     method="classifier",
    #                     batch_size=128,
    #                     learning_rate=0.005,
    #                     weight_decay=0
    #                 )
