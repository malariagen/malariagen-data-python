import pandas as pd
from fsspec.core import url_to_fs


class Ag3:

    def __init__(self, url, **kwargs):

        # special case Google Cloud Storage, use anonymous access, avoids a delay
        if url.startswith('gs://') or url.startswith('gcs://'):
            kwargs.setdefault('token', 'anon')

        # process the url using fsspec
        fs, path = url_to_fs(url, **kwargs)
        self.fs = fs
        self.path = path

        # check which releases are available
        sub_dirs = [p.split('/')[-1] for p in self.fs.ls(self.path)]
        releases = [d for d in sub_dirs if d.startswith('v')]
        if len(releases) == 0:
            raise ValueError(f'No releases found at location {url!r}')
        self.releases = releases

        # setup caches
        self._cache_sample_sets = dict()
        self._cache_sample_metadata = dict()

    def sample_sets(self, release='v3'):
        try:
            return self._cache_sample_sets[release]
        except KeyError:
            path = f"{self.path}/{release}/manifest.tsv"
            with self.fs.open(path) as f:
                df = pd.read_csv(f, sep='\t')
            df['release'] = release
            self._cache_sample_sets[release] = df
            return df

    def _get_release(self, sample_set):
        # find which release this sample set was included in
        for release in self.releases:
            df_sample_sets = self.sample_sets(release)
            if sample_set in df_sample_sets['sample_set'].values:
                return release
        raise ValueError(f'No release for sample set {sample_set!r}')

    def _read_sample_metadata(self, sample_set, release=None):
        """Read metadata for a single sample set."""
        try:
            return self._cache_sample_metadata[sample_set]
        except KeyError:
            if release is None:
                release = self._get_release(sample_set)
            path = f"{self.path}/{release}/metadata/general/{sample_set}/samples.meta.csv"
            with self.fs.open(path) as f:
                df = pd.read_csv(f)
            df['sample_set'] = sample_set
            df['release'] = release
            self._cache_sample_metadata[sample_set] = df
            return df

    def sample_metadata(self, sample_sets):
        """Read sample metadata for one or more sample sets.

        @@TODO parameters and returns
        
        """

        if isinstance(sample_sets, str) and sample_sets.startswith('v'):
            # convenience, can use a release identifier to denote all sample sets
            # in a release
            release = sample_sets
            sample_sets = self.sample_sets(release=release)['sample_set'].tolist()

        if isinstance(sample_sets, str):
            # single sample set given
            df = self._read_sample_metadata(sample_set=sample_sets)
            return df

        elif isinstance(sample_sets, (list, tuple)):
            # multiple sample sets given
            dfs = [self.sample_metadata(sample_sets=s)
                   for s in sample_sets]
            df = pd.concat(dfs, axis=0, sort=False).reset_index(drop=True)
            return df

        else:
            raise TypeError
