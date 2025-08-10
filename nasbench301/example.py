import os
from collections import namedtuple

from ConfigSpace.read_and_write import json as cs_json

import nasbench301 as nb
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=pathlib.Path, default=pathlib.Path(__file__).parent)
parser.add_argument("--config_space_dir", type=pathlib.Path, default=pathlib.Path(__file__).parent)

args = parser.parse_args()

# Default dirs for models
# Note: Uses 0.9 as the default models, switch to 1.0 to use 1.0 models

# models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
# model_paths_0_9 = {
#     model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
#     for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
# }



model_paths_1_0 = {
    model_name: (args.model_dir / "{}_v1.0".format(model_name)) 
        for model_name in ["xgb", "lgb_runtime"]
}

model_paths = model_paths_1_0


# If the models are not available at the paths, automatically download
# the models
# Note: If you would like to provide your own model locations, comment this out
# if not all(os.path.exists(model) for model in model_paths.values()):
#     nb.download_models(version=version, delete_zip=True,
#                        download_dir=current_dir)
#
# Load the performance surrogate model
#NOTE: Loading the ensemble will set the seed to the same as used during training (logged in the model_configs.json)
#NOTE: Defaults to using the default model download path
print("==> Loading performance surrogate model...")
ensemble_dir_performance = model_paths['xgb']
print(ensemble_dir_performance)
performance_model = nb.load_ensemble(ensemble_dir_performance)

# Load the runtime surrogate model
#NOTE: Defaults to using the default model download path
print("==> Loading runtime surrogate model...")
ensemble_dir_runtime = model_paths['lgb_runtime']
runtime_model = nb.load_ensemble(ensemble_dir_runtime)

# Option 1: Create a DARTS genotype
print("==> Creating test configs...")
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
genotype_config = Genotype(
        normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)],
        normal_concat=[2, 3, 4, 5],
        reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)],
        reduce_concat=[2, 3, 4, 5]
        )

print(f"Genotype {genotype_config}")
# Option 2: Sample from a ConfigSpace

configspace_path = args.config_space_dir  / "configspace.json"
#configspace_path = os.path.join(current_dir, 'configspace.json')
with open(configspace_path, "r") as f:
    json_string = f.read()
    configspace = cs_json.read(json_string)
configspace_config = configspace.sample_configuration()
print(f"config_space {configspace_config}")

# Predict
print("==> Predict runtime and performance...")
prediction_genotype = performance_model.predict(config=genotype_config, representation="genotype", with_noise=True)
prediction_configspace = performance_model.predict(config=configspace_config, representation="configspace", with_noise=True)

runtime_genotype = runtime_model.predict(config=genotype_config, representation="genotype")
runtime_configspace = runtime_model.predict(config=configspace_config, representation="configspace")

print("Genotype architecture performance: %f, runtime %f" %(prediction_genotype, runtime_genotype))
print("Configspace architecture performance: %f, runtime %f" %(prediction_configspace, runtime_configspace))
