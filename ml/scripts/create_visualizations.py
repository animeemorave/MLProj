import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "project"))
from ml.utils.visualizations import create_all_visualizations  # noqa: E402


def main():
    results_dir = project_root / "results"
    models_dir = project_root / "models"
    data_dir = project_root / "data" / "processed"
    label_mapping_path = (
        project_root / "data" / "processed" / "banking49_label_mapping.json"
    )
    output_dir = project_root / "presentations" / "visualizations"
    create_all_visualizations(
        results_dir=results_dir,
        models_dir=models_dir,
        data_dir=data_dir,
        label_mapping_path=label_mapping_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
