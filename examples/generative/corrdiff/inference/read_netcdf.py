import netCDF4 as nc
file_path = 'image_outdir_0_score.nc'
dataset = nc.Dataset(file_path, 'r')
print('Variables:')
for var_name, var in dataset.variables.items():
    print(f'{var_name}: {var[:]}')
print("""
Global attributes:""")
for attr_name in dataset.ncattrs():
    print(f'{attr_name}: {getattr(dataset, attr_name)}')
dataset.close()
