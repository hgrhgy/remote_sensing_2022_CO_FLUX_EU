from glob import glob

import pandas as pd
import zipfile
import os

os.chdir(r'H:\code\wrf\co_eu_eval')

if __name__ == '__main__':
    # inv_zip = zipfile.ZipFile(r'data/inventory/unfccc/deu-2021-crf-14apr21.zip', mode='r')
    #
    # fname = inv_zip.filelist[-1].filename
    # sheet_names = []
    # with inv_zip.open(fname, mode='r') as zf:
    #     inv_xls = pd.read_excel(zf)
    #     sheet_names.extend(inv_xls.values.squeeze())
    #
    #     for sn in sheet_names:
    #         sheet_data = pd.read_excel(zf, sheet_name=sn, engine='openpyxl')
    #         print(sheet_data)

    dir = r'data/inventory/unfccc'

    files = glob(os.path.join(dir, '*_2021_2015_*.xlsx'))
    files = sorted(files)

    all_data = dict()

    for xls in files:
        with open(xls, 'rb') as f:
            fn = xls.replace(dir + '\\', '')
            fn_data = fn.split('_')
            code = fn_data[0]
            ver = fn_data[1]
            year = fn_data[2]
            inv_xls = pd.read_excel(f, engine='openpyxl')
            sheet_names = inv_xls.values.squeeze()
            sheet_data = pd.read_excel(f, sheet_name='Summary1.As1', engine='openpyxl' )


            if not all_data.__contains__(code):
                all_data[code] = []

            all_data[code].append(sheet_data.values[5][10])


    for k in all_data.keys():
        print(k, all_data[k][0])