from pathlib import Path
import pandas as pd
import intake


RELEASES = 'ag3.0', 'ag3.1'


class Base:

    def _sample_sets(self, release: str):
        raise NotImplementedError

    def _in_release(self, sample_set):
        for release in RELEASES:
            df = self._sample_sets(release)
            if sample_set in df['sample_set'].values:
                return release
        raise ValueError

    def sample_sets(self, release):
        return self._sample_sets(release)

    def _samples(self, sample_set):
        raise NotImplementedError

    def samples(self, sample_sets):

        if sample_sets in RELEASES:
            # convenience to allow loading all sample sets in a release
            release = sample_sets
            sample_sets = self.sample_sets(release)['sample_set'].values.tolist()

        if isinstance(sample_sets, str):
            df = self._samples(sample_sets)

        elif isinstance(sample_sets, (list, tuple)):
            dfs = [self._samples(s) for s in sample_sets]
            df = (
                pd.concat(dfs, axis=0, sort=False)
                .reset_index(drop=True)
            )

        else:
            raise TypeError

        return df


class Local(Base):

    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path).expanduser()
        self.path = path

    def _release_path(self, release):
        # assume local paths mirror GCS paths under the vo_agam_release bucket
        if release == RELEASES[0]:
            return self.path / 'v3'
        elif release in RELEASES:
            return self.path / 'v' + release[2:]
        else:
            raise ValueError

    def _sample_sets(self, release):
        release_path = self._release_path(release)
        path = release_path / "manifest.tsv"
        df = pd.read_csv(path, sep="\t")
        return df

    def _samples(self, sample_set: str):

        # find which release the sample set is in
        release = self._in_release(sample_set)

        # find base path for release
        release_path = self._release_path(release)

        # build file path
        path = release_path / f"metadata/general/{sample_set}/samples.meta.csv"

        # read dataframe
        df = pd.read_csv(path)

        # add sample set identifier for convenience
        df["sample_set"] = sample_set

        return df


class Cloud(Base):

    def __init__(self, catalog="https://malariagen.github.io/intake/gcs.yml"):
        self.cat = intake.open_catalog(catalog)

    def _release_cat(self, release):
        if release == RELEASES[0]:
            return self.cat['ag3']
        elif release in RELEASES:
            return self.cat[release]
        else:
            raise ValueError

    def _sample_sets(self, release):
        release_cat = self._release_cat(release)
        df = release_cat["sample_sets"].read()
        return df

    def _samples(self, sample_set):

        # find which release the sample set is in
        release = self._in_
        df = self.cat["ag3"]["samples"](sample_set=sample_set).read()
        df["sample_set"] = sample_set
        return df
