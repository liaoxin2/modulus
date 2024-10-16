import os
"""Score the generated samples


Saves a netCDF of crps and other scores. Depends on time, but space and ensemble have been reduced::

    netcdf scores {
dimensions:
        metric = 4 ;
        time = 205 ;
variables:
        double eastward_wind_10m(metric, time) ;
                eastward_wind_10m:_FillValue = NaN ;
        double maximum_radar_reflectivity(metric, time) ;
                maximum_radar_reflectivity:_FillValue = NaN ;
        double northward_wind_10m(metric, time) ;
                northward_wind_10m:_FillValue = NaN ;
        double temperature_2m(metric, time) ;
                temperature_2m:_FillValue = NaN ;
        int64 time(time) ;
                time:units = "hours since 1990-01-01" ;
                time:calendar = "standard" ;
        string metric(metric) ;
}


"""
import dask.diagnostics
import xarray as xr
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import xskillscore


def open_samples(f):
    """
    Open prediction and truth samples from a dataset file.

    Parameters:
        f: Path to the dataset file.

    Returns:
        tuple: A tuple containing truth, prediction, and root datasets.
    """
    print(f'Opening {f}')
    root = xr.open_dataset(f)
    pred = xr.open_dataset(f, group='prediction')
    truth = xr.open_dataset(f, group='truth')
    pred = pred.merge(root)
    truth = truth.merge(root)
    truth = truth.set_coords(['lon', 'lat'])
    pred = pred.set_coords(['lon', 'lat'])
    return truth, pred, root


def compute_and_save_metrics(netcdf_file, output_dir):
    truth, pred, root = open_samples(netcdf_file)
    print('pred',pred['maximum_radar_reflectivity'].values.shape)
    exit()
    pred = pred.chunk(time=1)
    truth = truth.chunk(time=1)
    dim = ['x', 'y']
    a = xskillscore.rmse(truth, pred.mean('ensemble'), dim=dim)
    b = xskillscore.crps_ensemble(truth, pred, member_dim='ensemble', dim=dim)
    c = pred.std('ensemble').mean(dim)
    crps_mean = xskillscore.crps_ensemble(truth, pred.mean('ensemble').
        expand_dims('ensemble'), member_dim='ensemble', dim=dim)
    metrics = xr.concat([a, b, c, crps_mean], dim='metric').assign_coords(
        metric=['rmse', 'crps', 'std_dev', 'mae'])
    os.makedirs(output_dir, exist_ok=True)
    for var_name in a.data_vars:
        # 将 dask array 转为 NumPy array
        value = b[var_name].values
        
        # 打印变量名和它的值
        print(f"{var_name}: {value}")
    # for var_name in rmse.data_vars:
    #     print('rmse',a.data_vars)
    #     print('crps',b.data_vars)
    #     print('std_dev',c.data_vars)
    #     print('mae',crps_mean.data_vars)
    exit()
    print(f'Saving metrics to {output_dir}/scores.nc')
    with dask.diagnostics.ProgressBar():
        metrics.to_netcdf(os.path.join(output_dir, 'scores.nc'), mode='w')


def write_metrics_to_txt(netcdf_file, output_dir):
    """Read and save metrics from a NetCDF file to a text file, CSV file, and XLS file in transposed table format."""
    ds = xr.open_dataset(netcdf_file)
    metrics = ds.coords['metric'].values
    variables = list(ds.data_vars)
    avg_values = {variable: [] for variable in variables}
    for variable in variables:
        data = ds[variable]
        for i, metric in enumerate(metrics):
            metric_values = data[i, :].values
            average_value = metric_values.mean()
            avg_values[variable].append(average_value)
    df = pd.DataFrame(avg_values, index=metrics)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        header = 'Variable'.ljust(30) + ''.join([metric.ljust(15) for
            metric in metrics])
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')
        for variable in df.columns:
            row = variable.ljust(30) + ''.join([f'{value:.3f}'.ljust(15) for
                value in df[variable]])
            f.write(row + '\n')
    df.to_csv(os.path.join(output_dir, 'metrics.csv'), float_format='%.3f')
    df.to_excel(os.path.join(output_dir, 'metrics.xls'), float_format='%.3f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--netcdf_file', help='Path to the NetCDF file')
    parser.add_argument('--output_dir', help='Path to the output directory')
    args = parser.parse_args()
    compute_and_save_metrics(args.netcdf_file, args.output_dir)
    write_metrics_to_txt(os.path.join(args.output_dir, 'scores.nc'), args.
        output_dir)
