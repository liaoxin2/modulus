import os
import paddle
from dataclasses import dataclass
import cftime
from datetime import datetime, timedelta
import numpy as np
import xarray as xr


def norm_minus1_to_plus1_transform(x, x_min, x_max):
    results = (x - x_min) / (x_max - x_min)
    results = results * 2 - 1
    return results


def norm_transform(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


>>>>>>class DataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.training_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size

    def setup(self, stage: str=None):
        if stage == 'fit' or stage is None:
            self.train = CycleDataset('train', self.config)
            self.valid = CycleDataset('valid', self.config)
        if stage == 'test':
            self.test = CycleDataset('test', self.config)

    def train_dataloader(self):
        return paddle.io.DataLoader(dataset=self.train, batch_size=self.
            training_batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return paddle.io.DataLoader(dataset=self.valid, batch_size=self.
            test_batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return paddle.io.DataLoader(dataset=self.test, batch_size=self.
            test_batch_size, shuffle=False, num_workers=0)


class GFDLDataset(paddle.io.Dataset):

    def __init__(self, stage, config):
        """ 
            stage: train, valid, test
        """
        self.transforms = config.transforms
        self.epsilon = config.epsilon
        self.config = config
        self.splits = {'train': [str(config.train_start), str(config.
            train_end)], 'valid': [str(config.valid_start), str(config.
            valid_end)], 'test': [str(config.test_start), str(config.test_end)]
            }
        self.stage = stage
        self.date_list = self.get_timeList(int(self.splits[stage][0]), int(
            self.splits[stage][1]))
        self.num_samples = len(self.date_list)
        print(f'{stage} samples: {self.num_samples}')

    def get_timeList(self, startyear, endyear):
        start_date = datetime(startyear, 1, 1)
        end_date = datetime(endyear, 12, 31)
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            if not (current_date.month == 2 and current_date.day == 29 and
                (current_date.year % 4 == 0 and (current_date.year % 100 !=
                0 or current_date.year % 400 == 0))):
                date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        return date_list

    def load_minmax_reference(self, model_name):
        maindir = getattr(self.config, f'{model_name}_path')
        start_year, end_year = self.splits['train'][0], self.splits['train'][1]
        vars_data_min = []
        vars_data_max = []
        varnames_filename = getattr(self.config,
            f'vars_used_{model_name}_filename')
        varnames_filecontent = getattr(self.config,
            f'vars_used_{model_name}_filecontent')
        for varname_filename, varname_filecontent in zip(varnames_filename,
            varnames_filecontent):
            subdir = maindir
            filelist = os.listdir(subdir)
            data_min = []
            data_max = []
            for f in filelist:
                fpath = os.path.join(subdir, f)
                file_data = xr.open_dataset(fpath).isel(lat=slice(self.
                    config.lat_startidx, self.config.lat_endidx))
                data_min.append(file_data[varname_filecontent + '_min'])
                data_max.append(file_data[varname_filecontent + '_max'])
            data_min = xr.concat(data_min, dim='time').values
            data_min = np.nanmin(data_min, keepdims=True)
            data_max = xr.concat(data_max, dim='time').values
            data_max = np.nanmax(data_max, keepdims=True)
            vars_data_min.append(data_min)
            vars_data_max.append(data_max)
        vars_data_min = np.concatenate(vars_data_min, axis=0)
        vars_data_max = np.concatenate(vars_data_max, axis=0)
        return vars_data_min.astype('float32'), vars_data_max.astype('float32')

    def apply_transforms(self, data, data_min, data_max):
        if 'normalize' == self.transforms:
            data = norm_transform(data, data_min, data_max)
        if 'normalize_minus1_to_plus1' == self.transforms:
            data = norm_minus1_to_plus1_transform(data, data_min, data_max)
        return data

    def __getitem__(self, index):
        gfdl_data = []
        for varname_filename, varname_filecontent in zip(self.config.
            vars_used_gfdl_filename, self.config.vars_used_gfdl_filecontent):
            if varname_filename.startswith('zg'):
                middle_text = '_Eday_GFDL-ESM4_historical_r1i1p1f1_gr1_'
            elif varname_filename == 'tos':
                middle_text = '_Oday_GFDL-ESM4_historical_r1i1p1f1_gr_'
            else:
                middle_text = '_day_GFDL-ESM4_historical_r1i1p1f1_gr1_'
            subdir = os.path.join(self.config.gfdl_path, varname_filename +
                '_day')
            fname = varname_filename + middle_text + self.date_list[index
                ] + '.nc'
            fpath = os.path.join(subdir, fname)
            data = xr.open_dataset(fpath).isel(lat=slice(self.config.
                lat_startidx, self.config.lat_endidx))[varname_filecontent
                ].values
            gfdl_data.append(data)
        gfdl_data = np.stack(gfdl_data, 0)
        era5_data = []
        prefix = 'daily_mean_'
        for varname_filename, varname_filecontent in zip(self.config.
            vars_used_era5_filename, self.config.vars_used_era5_filecontent):
            subdir = os.path.join(self.config.era5_path, varname_filename +
                '_day')
            fname = prefix + varname_filename + '_' + self.date_list[index
                ] + '.nc'
            fpath = os.path.join(subdir, fname)
            data = xr.open_dataset(fpath).isel(lat=slice(self.config.
                lat_startidx, self.config.lat_endidx))[varname_filecontent
                ].values
            era5_data.append(data)
        era5_data = np.stack(era5_data, 0)
        era5_data = self.apply_transforms(era5_data, self.
            era5_min_reference, self.era5_max_reference)
        gfdl_data = self.apply_transforms(gfdl_data, self.
            climate_model_min_reference, self.climate_model_max_reference)
        gfdl_data[np.isnan(gfdl_data)] = 0
        era5_data[np.isnan(era5_data)] = 0
        x = paddle.to_tensor(data=gfdl_data).astype(dtype='float32')
        y = paddle.to_tensor(data=era5_data).astype(dtype='float32')
        sample = {'A': x, 'B': y}
        return sample

    def __len__(self):
        return self.num_samples
