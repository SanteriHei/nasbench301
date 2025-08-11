import pathlib
import pickle
import argparse
import xgboost


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=pathlib.Path, default=None)

args = parser.parse_args()

assert args.model_dir.is_dir()

model_paths = [fp for fp in args.model_dir.glob("*/xgb/*/surrogate_model.json")]
assert len(model_paths) == 10, f"Got {len(model_paths)} instead of 10 (expected)!"

model = xgboost.Booster()
for i, fp in enumerate(model_paths):
    model.load_model(fp)
    # with fp.open('rb') as f:
    #     model = pickle.load(f)
    model_num = fp.parts[-4]
    print(f"Converting model {model_num}")
    ubj_path = fp.with_suffix(".ubj")
    model.save_model(ubj_path)
    print(f"Saved model to {ubj_path!s}")
    json_path = fp.with_suffix(".json")
    model.save_model(json_path)
    print(f"Saved model to {json_path!s}")
    print(f"Model {model_num} ({(i+1)}/{len(model_paths)}) ready")
