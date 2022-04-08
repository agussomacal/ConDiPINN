from pathlib import Path

project_root = Path(__file__).parent.parent
data_path = Path.joinpath(project_root, 'Data')
results_path = Path.joinpath(project_root, 'Results')

data_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)

experiments_path = Path.joinpath(project_root, 'src')
experiments_path = Path.joinpath(experiments_path, 'experiments')
