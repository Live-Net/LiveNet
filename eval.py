import os
import torch
import config
import matplotlib.pyplot as plt
from model_utils import ModelDefinition
from models import BarrierNet
from data_logger import DataGenerator, Dataset
from sklearn.model_selection import train_test_split


def test(dataloader, model, loss_fn, losses):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(config.device), y.to(config.device)
            pred = model(X, 1)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
    test_loss /= num_batches
    losses.append(test_loss)
    print(f"Test avg loss: {test_loss:>8f} \n")
    return losses

if __name__ == "__main__":
    params = {'batch_size': config.train_batch_size,
            'shuffle': True,
            'num_workers': 4}
    
    model_definition_filepath = 'weights/model2_smg_w_lims_opp_pen_dgoal_obs_l_s_1_bn_definition.json'
    eval_data_paths = [
        'obs_doorway_with_offsets/l_0_faster_off0.json',
        'obs_doorway_with_offsets/l_0_faster_edge_cases.json',
    ]

    model_definition = ModelDefinition.from_json(model_definition_filepath)
    generator = DataGenerator(eval_data_paths, model_definition.x_is_d_goal)

    model = BarrierNet(model_definition, generator.get_obstacles(), generator.data_streams[0]["iterations"][0]["goals"][config.agents_to_train_on[0]]).to(config.device)
    model.load_state_dict(torch.load(model_definition.weights_path))
    model.eval()

    norm_inputs, input_mean, input_std = generator.get_inputs(agent_idx=config.agents_to_train_on, normalize=True)
    norm_outputs, output_mean, output_std = generator.get_outputs(agent_idx=config.agents_to_train_on, normalize=True)

    dataset = Dataset(norm_inputs, norm_outputs)
    dataloader = torch.utils.data.DataLoader(dataset, **params)

    loss_fn = torch.nn.MSELoss()
    calc_loss = test(dataloader, model, loss_fn, [])[0]
    print("Average loss:", calc_loss)

    tr = []
    ctrl1, ctrl2, ctrl1_real, ctrl2_real = [], [], [], []
    t0 = 0

    with torch.no_grad():
        for X, y in zip(norm_inputs, norm_outputs):
            x = torch.autograd.Variable(torch.from_numpy(X), requires_grad=False)
            x = torch.reshape(x, (1, len(X)))
            x = x.to(config.device)
            ctrl = model(x, 0)
            
            unnorm_ctrl = ctrl * output_std + output_mean
            if config.use_barriernet:
                ctrl1.append(unnorm_ctrl[0])
                ctrl2.append(unnorm_ctrl[1])
            else:
                ctrl1.append(unnorm_ctrl[0,0].item())
                ctrl2.append(unnorm_ctrl[0,1].item())
            unnorm_y = y * output_std + output_mean
            ctrl1_real.append(unnorm_y[0])
            ctrl2_real.append(unnorm_y[1])
            tr.append(t0)
            t0 = t0 + 0.2

    print("Test done!")
    print("Average loss:", calc_loss)


    plt.figure(1)
    plt.plot(tr, ctrl1_real, color = 'red', label = 'actual(optimal)')
    plt.plot(tr, ctrl1, color = 'blue', label = 'implemented')
    plt.legend()
    plt.ylabel('Angular speed (control)')
    plt.xlabel('time')

    plt.figure(2)
    plt.plot(tr, ctrl2_real, color = 'red', label = 'actual(optimal)')
    plt.plot(tr, ctrl2, color = 'blue', label = 'implemented')
    plt.legend()
    plt.ylabel('Acceleration (control)')
    plt.xlabel('time')

    plt.show()
