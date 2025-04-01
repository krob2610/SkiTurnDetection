from datetime import timedelta
from pathlib import Path

from .device_namespace import DATA_PATH, FINAL_DF_SENSORS


class Labeler:
    def __init__(self, date):
        self.folders_dict = self._get_folders_files_paths(date)

    def _get_folders_files_paths(self, date: str) -> dict[list]:
        """It will provide path to folders in the selected date and store it in the dictionary with lists of files for each device

        Returns
        -------
        list[str]
            _description_
        """
        path = Path(DATA_PATH, date)
        folders = [file for file in path.iterdir() if file.is_dir()]
        folders_files = {}
        for folder in folders:
            folders_files[folder.name] = [
                file for file in folder.iterdir() if file.suffix == ".csv"
            ]
        return folders_files

    def prepare_date(self) -> None:
        """Prepare date for labeling"""
        for folder in self.folders_dict:
            self._generate_final_df(self.folders_dict[folder])
            for file in self.folders_dict[folder]:
                self._prepare_file(file)

    def _generate_final_df(self, paths: list[Path]) -> None:
        """Generate final dataframe with all data from selected sensors per device

        Parameters
        ----------
        paths : list[Path]
            list with paths to files

        Returns
        -------
        pd.DataFrame
            final dataframe
        """
        df = pd.DataFrame()
        for path in paths:
            df = pd.concat([df, pd.read_csv(path)])
        return df

    def _check_for_final_df_file(self, path: Path) -> bool:
        pass

    def _create_skier_dataframe(
        self, time_start, time_stop, skier, skier_skill
    ) -> None:
        """Create dataframe with skier data

        Parameters
        ----------
        time_start : timedelta
            start time of the skier
        time_stop : timedelta
            stop time of the skier
        skier : str
            skier name
        skier_skill : str
            skier skill
        """

    def _apply_labels(
        self,
        time_start: timedelta,
        time_stop: timedelta,
        style: str,
        skier: str,
        turn_direction: str,
    ) -> None:
        """Apply labels to files

        Parameters
        ----------
        labels : list[str]
            list with labels
        """
        for folder in self.folders_dict:
            for file in self.folders_dict[folder]:
                with open(file, "a") as f:
                    f.write(f"\n{labels[int(folder)]}")
                    f.write(f"\n{labels[int(folder)]}")
