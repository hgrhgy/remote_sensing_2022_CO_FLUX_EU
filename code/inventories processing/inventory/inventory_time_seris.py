from glob import glob

import pandas as pd
import zipfile
import os

os.chdir(r'H:\code\wrf\co_eu_eval')


if __name__ == '__main__':
    dir = r'data/inventory/unfccc'
    code = 'EUA'
    files = glob(os.path.join(dir, '%s_2021_*.xlsx' % code))
    files = sorted(files)

    all_data = []

    for xls in files:
        with open(xls, 'rb') as f:
            fn = xls.replace(dir + '\\', '')
            fn_data = fn.split('_')
            code = fn_data[0]
            ver = fn_data[1]
            year = fn_data[2]
            inv_xls = pd.read_excel(f, engine='openpyxl')
            sheet_names = inv_xls.values.squeeze()
            sheet_data = pd.read_excel(f, sheet_name='Summary1.As1', engine='openpyxl')

            co2_eng = sheet_data.values[5][1]
            co_eng = sheet_data.values[5][10]
            nox_eng = sheet_data.values[5][9]

            # co2_ind = sheet_data.values[18][1]
            # co_ind = sheet_data.values[18][10]
            # nox_ind = sheet_data.values[19][9]
            all_data.append((code, fn_data[2], co2_eng, co_eng, nox_eng))

    for k in all_data:
        print('%s\t%s\t%s\t%s\t%s' % k)