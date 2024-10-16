import argparse
import joblib
import matplotlib.pyplot as plt
import xarray


def plot_samples(netcdf_file, output_dir, n_samples):
    """Plot multiple samples"""
    root = xarray.open_dataset(netcdf_file)
    ds = xarray.open_dataset(netcdf_file, group='prediction').merge(root
        ).set_coords(['lat', 'lon'])
    truth = xarray.open_dataset(netcdf_file, group='truth').merge(root
        ).set_coords(['lat', 'lon'])
    os.makedirs(output_dir, exist_ok=True)
    truth_expanded = truth.assign_coords(ensemble='truth').expand_dims(
        'ensemble')
    ens_mean = ds.mean('ensemble').assign_coords(ensemble='ensemble_mean'
        ).expand_dims('ensemble')
    ds['ensemble'] = [str(i) for i in range(ds.sizes['ensemble'])]
    merged = xarray.concat([truth_expanded, ens_mean, ds], dim='ensemble')

    def plot(v):
        print(v)
        merged[v][:n_samples + 2, :].plot(row='time', col='ensemble')
        plt.savefig(f'{output_dir}/{v}.png')
    joblib.Parallel(n_jobs=8)(joblib.delayed(plot)(v) for v in merged)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_file', help='Path to the NetCDF file')
    parser.add_argument('--output_dir', help='Path to the output directory')
    parser.add_argument('--n-samples', help='Number of samples', default=5,
        type=int)
    args = parser.parse_args()
    main(args.netcdf_file, args.output_dir, args.n_samples)
