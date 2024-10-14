import sys
sys.path.append('../..')

from nullingexplorer.io import FitResult

hdf5_file = '../results/Job_20240917_153553/result_earth.hdf5'
fit_result = FitResult.load(hdf5_file)

fit_result.draw_scan_result(position_name=['earth.r_polar', 'earth.r_angular'], show=True, polar=True)