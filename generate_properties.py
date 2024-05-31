import pandas as pd
import numpy as np
import argparse
import os


def generate_vnnlib_files(datasets, h_names):
    instances_dict =  {
                'Network': [],
                'Property': [],
                'Timeout': []
                }
    
    for dataset in datasets:
        hyperrectangles = []
        for h_name in h_names:
            if hyperrectangles == []:
                hyperrectangles = np.load(f'data/{dataset}/{h_name}.npy')
            else:
                hyperrectangles = np.concatenate((hyperrectangles, np.load(f'data/{dataset}/{h_name}.npy')), axis=0)

        print(f'{dataset} -|- {hyperrectangles.shape}')
        
        properties_directory = f'vnnlib/{dataset}'
        if not os.path.exists(properties_directory):
            os.makedirs(properties_directory)

        for i, hyperrectangle in enumerate(hyperrectangles):
            with open(f'{properties_directory}/hyperrectangle_{i}.vnnlib', 'w') as property_file:

                property_file.write(f'; {dataset} perturbations property.\n\n')
                for j,d in enumerate(hyperrectangle):
                    property_file.write(f'(declare-const X_{j} Real)\n')
                property_file.write(f'\n(declare-const Y_0 Real)\n')
                property_file.write(f'(declare-const Y_1 Real)\n\n')

                property_file.write('; Input constraints:\n')
                for j,d in enumerate(hyperrectangle):
                    property_file.write(f'(assert (>= X_{j} {d[0]}))\n')
                    property_file.write(f'(assert (<= X_{j} {d[1]}))\n\n')
                property_file.write('; Output constraints:\n')
                property_file.write('(assert (<= Y_0 Y_1))')

                instances_dict['Network'].append(f'onnx/{dataset}/perturbations_0.onnx')
                instances_dict['Property'].append(f'{properties_directory}/hyperrectangle_{i}.vnnlib')
                instances_dict['Timeout'].append(20)

    return pd.DataFrame(instances_dict)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, default=42, help='random seed.')
    args = parser.parse_args()

    datasets = ['ruarobot', 'medical']
    hyperrectangles = ['character', 'word', 'vicuna']

    instances_df = generate_vnnlib_files(datasets, hyperrectangles)

    instances_df.sample(n=1080, random_state=args.seed).to_csv(f'instances.csv', index=False, header=False)