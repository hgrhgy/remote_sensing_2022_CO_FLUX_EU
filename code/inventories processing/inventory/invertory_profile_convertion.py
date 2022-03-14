from glob import glob

import pandas as pd
import zipfile
import os
import numpy as np

os.chdir(r'H:\code\wrf\co_eu_eval')

countries = ['AUT', 'BEL', 'CHE', 'CZE', 'DEU', 'DNK', 'ESP', 'FRA', 'IRL', 'ITA', 'LUX', 'NLD', 'PRT', 'SVN', 'GBR']
months_dict = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def is_number(s):
    try:
        f = float(s)
        if np.isnan(f):
            return False
        return True
    except ValueError:
        pass


class COEmissionSectoral:

    def __init__(self):
        pass

    def init_from_xls(self, xls):
        with open(xls, 'rb') as f:
            fn = xls.replace(dir + '\\', '')
            fn_data = fn.split('_')
            ver = fn_data[1]
            year = fn_data[2]
            inv_xls = pd.read_excel(f, engine='openpyxl')
            sheet_names = inv_xls.values.squeeze()
            summary1_as1 = pd.read_excel(f, sheet_name='Summary1.As1', engine='openpyxl')
            summary1_as2 = pd.read_excel(f, sheet_name='Summary1.As2', engine='openpyxl')
            summary1_as3 = pd.read_excel(f, sheet_name='Summary1.As3', engine='openpyxl')
            table2_i_s1 = pd.read_excel(f, sheet_name='Table2(I)s1', engine='openpyxl')
            table2_i_s2 = pd.read_excel(f, sheet_name='Table2(I)s2', engine='openpyxl')
            table1_Aa_s1 = pd.read_excel(f, sheet_name='Table1.A(a)s1', engine='openpyxl')
            table1_Aa_s2 = pd.read_excel(f, sheet_name='Table1.A(a)s2', engine='openpyxl')
            table1_Aa_s3 = pd.read_excel(f, sheet_name='Table1.A(a)s3', engine='openpyxl')
            table1_Aa_s4 = pd.read_excel(f, sheet_name='Table1.A(a)s4', engine='openpyxl')
            table1_s1 = pd.read_excel(f, sheet_name='Table1s1', engine='openpyxl')
            table1_s2 = pd.read_excel(f, sheet_name='Table1s2', engine='openpyxl')

            self.co_1a = summary1_as1.values[8, 10]
            self.ene_1a = table1_Aa_s1.values[8, 1]

            self.co_1a1 = summary1_as1.values[9, 10]
            self.ene_1a1_liquid = table1_Aa_s1.values[9, 1]
            self.ene_1a1_solid = table1_Aa_s1.values[10, 1]
            self.ene_1a1_gas = table1_Aa_s1.values[11, 1]
            self.ene_1a1_other = table1_Aa_s1.values[12, 1]
            self.ene_1a1_peats = table1_Aa_s1.values[13, 1] if is_number(table1_Aa_s1.values[13, 1]) else 0
            self.ene_1a1_biomass = table1_Aa_s1.values[14, 1]

            self.co_1a2 = summary1_as1.values[10, 10]
            self.co_1a2a = table1_s1.values[12, 5] if is_number(table1_s1.values[12, 5]) else 0
            self.co_1a2b = table1_s1.values[13, 5] if is_number(table1_s1.values[13, 5]) else 0
            self.co_1a2c = table1_s1.values[14, 5] if is_number(table1_s1.values[14, 5]) else 0
            self.co_1a2d = table1_s1.values[15, 5] if is_number(table1_s1.values[15, 5]) else 0
            self.co_1a2e = table1_s1.values[16, 5] if is_number(table1_s1.values[16, 5]) else 0
            self.co_1a2f = table1_s1.values[17, 5] if is_number(table1_s1.values[17, 5]) else 0
            self.co_1a2g = table1_s1.values[18, 5] if is_number(table1_s1.values[18, 5]) else 0

            self.co_1a3 = summary1_as1.values[11, 10]
            self.co_1a3a = table1_s1.values[20, 5] if is_number(table1_s1.values[20, 5]) else 0
            self.co_1a3b = table1_s1.values[21, 5] if is_number(table1_s1.values[21, 5]) else 0
            self.co_1a3c = table1_s1.values[22, 5] if is_number(table1_s1.values[22, 5]) else 0
            self.co_1a3d = table1_s1.values[23, 5] if is_number(table1_s1.values[23, 5]) else 0
            self.co_1a3e = table1_s1.values[24, 5] if is_number(table1_s1.values[24, 5]) else 0

            self.co_1a4 = summary1_as1.values[12, 10] if is_number(summary1_as1.values[12, 10]) else 0
            self.co_1a4 += summary1_as1.values[13, 10] if is_number(summary1_as1.values[13, 10]) else 0

            self.co_1b = summary1_as1.values[14, 10] if is_number(summary1_as1.values[14, 10]) else 0
            self.co_1c = summary1_as1.values[17, 10] if is_number(summary1_as1.values[17, 10]) else 0

            self.co_2a1 = table2_i_s1.values[7, 10] if is_number(table2_i_s1.values[7, 10]) else 0
            self.co_2a2 = table2_i_s1.values[8, 10] if is_number(table2_i_s1.values[8, 10]) else 0
            self.co_2a3 = table2_i_s1.values[9, 10] if is_number(table2_i_s1.values[9, 10]) else 0
            self.co_2a4 = table2_i_s1.values[10, 10] if is_number(table2_i_s1.values[10, 10]) else 0

            self.co_2b1 = table2_i_s1.values[12, 10] if is_number(table2_i_s1.values[12, 10]) else 0
            self.co_2b2 = table2_i_s1.values[13, 10] if is_number(table2_i_s1.values[13, 10]) else 0
            self.co_2b3 = table2_i_s1.values[14, 10] if is_number(table2_i_s1.values[14, 10]) else 0
            self.co_2b4 = table2_i_s1.values[15, 10] if is_number(table2_i_s1.values[15, 10]) else 0
            self.co_2b5 = table2_i_s1.values[16, 10] if is_number(table2_i_s1.values[16, 10]) else 0
            self.co_2b6 = table2_i_s1.values[17, 10] if is_number(table2_i_s1.values[17, 10]) else 0
            self.co_2b7 = table2_i_s1.values[18, 10] if is_number(table2_i_s1.values[18, 10]) else 0
            self.co_2b8 = table2_i_s1.values[19, 10] if is_number(table2_i_s1.values[19, 10]) else 0
            self.co_2b9 = table2_i_s1.values[20, 10] if is_number(table2_i_s1.values[20, 10]) else 0
            self.co_2b10 = table2_i_s1.values[21, 10] if is_number(table2_i_s1.values[21, 10]) else 0

            self.co_2c1 = table2_i_s1.values[23, 10] if is_number(table2_i_s1.values[23, 10]) else 0
            self.co_2c2 = table2_i_s1.values[24, 10] if is_number(table2_i_s1.values[24, 10]) else 0
            self.co_2c3 = table2_i_s1.values[25, 10] if is_number(table2_i_s1.values[25, 10]) else 0
            self.co_2c4 = table2_i_s1.values[26, 10] if is_number(table2_i_s1.values[26, 10]) else 0
            self.co_2c5 = table2_i_s1.values[27, 10] if is_number(table2_i_s1.values[27, 10]) else 0
            self.co_2c6 = table2_i_s1.values[28, 10] if is_number(table2_i_s1.values[28, 10]) else 0
            self.co_2c7 = table2_i_s1.values[29, 10] if is_number(table2_i_s1.values[29, 10]) else 0

            self.co_2d1 = table2_i_s2.values[6, 10] if is_number(table2_i_s2.values[6, 10]) else 0
            self.co_2d2 = table2_i_s2.values[7, 10] if is_number(table2_i_s2.values[7, 10]) else 0
            self.co_2d3 = table2_i_s2.values[8, 10] if is_number(table2_i_s2.values[8, 10]) else 0

            self.co_2e1 = table2_i_s2.values[10, 10] if is_number(table2_i_s2.values[10, 10]) else 0
            self.co_2e2 = table2_i_s2.values[11, 10] if is_number(table2_i_s2.values[11, 10]) else 0
            self.co_2e3 = table2_i_s2.values[12, 10] if is_number(table2_i_s2.values[12, 10]) else 0
            self.co_2e4 = table2_i_s2.values[13, 10] if is_number(table2_i_s2.values[13, 10]) else 0
            self.co_2e5 = table2_i_s2.values[14, 10] if is_number(table2_i_s2.values[14, 10]) else 0

            self.co_2f1 = table2_i_s2.values[16, 10] if is_number(table2_i_s2.values[16, 10]) else 0
            self.co_2f2 = table2_i_s2.values[17, 10] if is_number(table2_i_s2.values[17, 10]) else 0
            self.co_2f3 = table2_i_s2.values[18, 10] if is_number(table2_i_s2.values[18, 10]) else 0
            self.co_2f4 = table2_i_s2.values[19, 10] if is_number(table2_i_s2.values[19, 10]) else 0
            self.co_2f5 = table2_i_s2.values[20, 10] if is_number(table2_i_s2.values[20, 10]) else 0
            self.co_2f6 = table2_i_s2.values[21, 10] if is_number(table2_i_s2.values[21, 10]) else 0

            self.co_2g1 = table2_i_s2.values[23, 10] if is_number(table2_i_s2.values[24, 10]) else 0
            self.co_2g2 = table2_i_s2.values[24, 10] if is_number(table2_i_s2.values[24, 10]) else 0
            self.co_2g3 = table2_i_s2.values[25, 10] if is_number(table2_i_s2.values[25, 10]) else 0
            self.co_2g4 = table2_i_s2.values[26, 10] if is_number(table2_i_s2.values[26, 10]) else 0
            self.co_2h = table2_i_s2.values[27, 10] if is_number(table2_i_s2.values[27, 10]) else 0

            self.co_3 = summary1_as2.values[6, 10] if is_number(summary1_as2.values[6, 10]) else 0
            self.co_4 = summary1_as2.values[17, 10] if is_number(summary1_as2.values[17, 10]) else 0
            self.co_5 = summary1_as2.values[26, 10] if is_number(summary1_as2.values[26, 10]) else 0
            self.co_6 = summary1_as2.values[32, 10] if is_number(summary1_as2.values[32, 10]) else 0


class EnergyConsumption:

    def __init__(self):
        pass


class HighTemporalPofile:

    def __init__(self, xls='data/edgar/EDGAR_temporal_profiles_r1.xlsx'):
        with open(xls, 'rb') as f:
            self.profiles = pd.read_excel(f, sheet_name=1, engine='openpyxl', header=1)
            self.regions = pd.read_excel(f, sheet_name=2, engine='openpyxl', header=0)

    def get_code(self, iso_country_code):
        return int(
            self.regions[self.regions['Country ISO codes'] == iso_country_code]['World regions (as mapped in  Fig.1)'])

    # 1.A.1 Energy industries
    def get_energy_month_coffs(self, code_or_region, year, month):
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles['Year'] == year) & (self.profiles[
                                                                                                         'IPCC_2006_source_category'] == '1.A.1')]

        d = dict()
        coffs = data[months_dict[month]].array

        if len(coffs) < 5:
            code = self.get_code(code_or_region)
            data = self.profiles[
                (self.profiles['Region/country'] == code) & (self.profiles['Year'] == year) & (self.profiles[
                                                                                                   'IPCC_2006_source_category'] == '1.A.1')]
            coffs = data[months_dict[month]].array
        # classified by fuel type
        d['biofuels'] = float(coffs[0])
        d['coal'] = float(coffs[1])
        d['gas'] = float(coffs[2])
        d['oil'] = float(coffs[3])
        d['other'] = float(coffs[4])
        return d

    # 1.A.2 Manufacturing industries and construction
    def get_manufacturing_month_coffs(self, code_or_region, month):
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles[
                'IPCC_1996_source_category'].str.startswith(
                '1A2'))]

        d = dict()
        # 1A2a iron and steel
        d['1A2a'] = float(data[data['IPCC_1996_source_category'] == '1A2a'][months_dict[month]])
        # 1A2b Non-ferrous metals
        d['1A2b'] = float(data[(data['IPCC_1996_source_category'] == '1A2b') & (
            data['Activity sector description'].str.contains('Non-ferrous metals'))][months_dict[month]])
        # 1A2c Chemicals
        d['1A2c'] = float(data[data['IPCC_1996_source_category'] == '1A2c'][months_dict[month]])
        # 1A2d Pulp, paper and print
        d['1A2d'] = float(data[data['IPCC_1996_source_category'] == '1A2d'][months_dict[month]])
        # 1A2e Food processing, beverages and tobacco
        d['1A2e'] = float(data[(data['IPCC_1996_source_category'] == '1A2e')][months_dict[month]])
        # 1A2f Non-metallic minerals
        d['1A2f'] = float(data[(data['IPCC_1996_source_category'] == '1A2b') & (
            data['Activity sector description'].str.contains('Non-metallic minerals'))][months_dict[month]])
        # 1A2g Other
        d['1A2g'] = np.mean(data[(data['IPCC_1996_source_category'] == '1A2f')][months_dict[month]])
        return d

    # 1.A.3 Transport
    def get_transport_month_coffs(self, code_or_region, month):
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles[
                'IPCC_1996_source_category'].str.startswith(
                '1A3'))]

        d = dict()
        # 1A3a Domestic aviation
        d['1A3a'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('1A3a')][months_dict[month]])
        # 1A3b Road transportation
        d['1A3b'] = float(data[(data['IPCC_1996_source_category'] == '1A3b')][months_dict[month]])
        # 1A3c Railways
        d['1A3c'] = float(data[(data['IPCC_1996_source_category'] == '1A3c') & (
                data['Activity sector description'] == 'Rail transport')][months_dict[month]])
        # 1A3d Domestic navigation
        d['1A3d'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('1A3d')][months_dict[month]])
        # 1A3e Other transportation
        d['1A3e'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('1A3e')][months_dict[month]])

        return d

    # 1.A.4 Other
    def get_residential_and_other_sector(self, code_or_region, year, month):
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles['Year'] == year) & (self.profiles[
                                                                                                         'IPCC_2006_source_category'] == '1A4')]

        d = dict()
        # small combustion
        d['1A4'] = float(data[months_dict[month]])
        return d

    # 2.A Mineral industry
    def get_mineral_industry_coffs(self, code_or_region, month):
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles[
                'IPCC_1996_source_category'].str.startswith(
                '2A'))]

        d = dict()

        # 2A1 cement production
        d['2A1'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2A1')][months_dict[month]])
        # 2A2 Lime production
        d['2A2'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2A2')][months_dict[month]])
        # 2A3 Glass production
        d['2A3'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2A3')][months_dict[month]])
        # 2A4 other process uses of carbonates
        d['2A4'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2A4')][months_dict[month]])

        return d

    # 2B Chemical industry
    def get_chemical_industry_coffs(self, code_or_region, month):
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles[
                'IPCC_1996_source_category'].str.startswith(
                '2B'))]

        d = dict()

        # 2B1 Ammonia production
        d['2B1'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2B1')][months_dict[month]])
        # 2B2 Nitric acid production
        d['2B2'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2B2')][months_dict[month]])
        # 2B3 Adipic acid production
        d['2B3'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2B3')][months_dict[month]])
        # 2B4 Caprolactam, glyxal and glyoxylic acid production
        d['2B4'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2B4')][months_dict[month]])
        # 2B5 Carbide production
        d['2B5'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2B5')][months_dict[month]])
        # 2B6 Titanium dioxide production
        d['2B6'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2B6')][months_dict[month]])
        # 2B7 Soda ash production
        d['2B7'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2B7')][months_dict[month]])
        # 2B8 Petrochemical and carbon black production
        d['2B8'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2B8')][months_dict[month]])
        # 2B9 Fluorochemical production
        d['2B9'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2B9')][months_dict[month]])
        # 2B10 Other
        d['2B10'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2B10')][months_dict[month]])

        for k in d.keys():
            if np.isnan(d[k]):
                d[k] = 1.0 / 12.0

        return d

    # 2C Metal industry
    def get_metal_industry_coffs(self, code_or_region, month):
        type = '2C'
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles[
                                                                       'IPCC_1996_source_category'].str.startswith(type
                                                                                                                   ))]

        d = dict()

        # 2C1 Iron and steel production
        d['2C1'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2C1')][months_dict[month]])
        # 2C2 Ferroalloys production
        d['2C2'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2C2')][months_dict[month]])
        # 2C3 Aluminium production
        d['2C3'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2C3')][months_dict[month]])
        # 2C4 Magnesium production
        d['2C4'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2C4')][months_dict[month]])
        # 2C5 Lead production
        d['2C5'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2C5')][months_dict[month]])
        # 2C6 Zinc production
        d['2C6'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2C6')][months_dict[month]])
        # 2C7 Other
        d['2C7'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2C7')][months_dict[month]])

        for k in d.keys():
            if np.isnan(d[k]):
                d[k] = 1.0 / 12.0

        return d

    # 2D Non-energy products from fuels and solvent use
    def get_non_energy_product_coffs(self, code_or_region, month):
        type = '2D'
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles[
                'IPCC_1996_source_category'].str.startswith(
                type
            ))]

        d = dict()

        # 2D1 Lubricant use
        d['2D1'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2D1')][months_dict[month]])
        # 2D2 Paraffin wax use
        d['2D2'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2D2')][months_dict[month]])
        # 2D3 Other
        d['2D3'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2D3')][months_dict[month]])

        for k in d.keys():
            if np.isnan(d[k]):
                d[k] = 1.0 / 12.0

        return d

    # 2E Electronics industry
    def get_electronics_industry_coffs(self, code_or_region, month):
        type = '2E'
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles[
                'IPCC_1996_source_category'].str.startswith(
                type
            ))]

        d = dict()

        # 2E1 Intergrated circuit or semiconductor
        d['2E1'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2E1')][months_dict[month]])
        # 2E2 TFT flat panel display
        d['2E2'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2E2')][months_dict[month]])
        # 2E3 Photovoltaics
        d['2E3'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2E3')][months_dict[month]])
        # 2E4 Heat transfer fluid
        d['2E4'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2E4')][months_dict[month]])
        # 2E5 Other
        d['2E5'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2E5')][months_dict[month]])

        for k in d.keys():
            if np.isnan(d[k]):
                d[k] = 1.0 / 12.0

        return d

    # 2F Product uses as substitutes for ODS
    def get_substitutes_for_ods_coffs(self, code_or_region, month):
        type = '2F'
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles[
                'IPCC_1996_source_category'].str.startswith(
                type
            ))]

        d = dict()

        # 2F1 Refrigeration and air conditioning
        d['2F1'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2F1')][months_dict[month]])
        # 2F2 Foam blowing agents
        d['2F2'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2F2')][months_dict[month]])
        # 2F3 Fire protection
        d['2F3'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2F3')][months_dict[month]])
        # 2F4 Aerosols
        d['2F4'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2F4')][months_dict[month]])
        # 2F5 Solvents
        d['2F5'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2F5')][months_dict[month]])
        # 2F6 Other application
        d['2F6'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2F6')][months_dict[month]])

        for k in d.keys():
            if np.isnan(d[k]):
                d[k] = 1.0 / 12.0

        return d

    # 2G Other product manufacture and use
    def get_other_product_use_coffs(self, code_or_region, month):
        type = '2G'
        data = self.profiles[
            (self.profiles['Region/country'] == code_or_region) & (self.profiles[
                'IPCC_1996_source_category'].str.startswith(
                type
            ))]

        d = dict()

        # 2G1 Electrical equipment
        d['2G1'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2G1')][months_dict[month]])
        # 2G2 SF6 and PFCs from other product use
        d['2G2'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2G2')][months_dict[month]])
        # 2G3 N2O from product uses
        d['2G3'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2G3')][months_dict[month]])
        # 2G4 Other
        d['2G4'] = np.mean(data[data['IPCC_1996_source_category'].str.startswith('2G4')][months_dict[month]])

        for k in d.keys():
            if np.isnan(d[k]):
                d[k] = 1.0 / 12.0

        return d

    # 3 & 4 & 5 : Agriculture & Land use, land-use change and forestry & Waste
    def get_other_coffs(self, code_or_region, month):
        return 1.0 / 12.0


if __name__ == '__main__':
    dir = r'data/inventory/unfccc'
    year = 2019
    profiles = HighTemporalPofile()

    for country in countries:
        sum_co = 0
        for month in [12, 1, 2]:
            if month == 12:
                xls = glob(os.path.join(dir, '%s_2021_%d*.xlsx' % (country, year-1)))[0]
            else:
                xls = glob(os.path.join(dir, '%s_2021_%d*.xlsx' % (country, year)))[0]
            emissions = COEmissionSectoral()

            emissions.init_from_xls(xls)

            code = profiles.get_code(country)

            # 1A1
            if year == 2017 and month ==12:
                coff_1a1 = profiles.get_energy_month_coffs(country, year - 1, month)
            else:
                coff_1a1 = profiles.get_energy_month_coffs(country, 2017, month)

            sum_co += emissions.ene_1a1_gas / emissions.ene_1a * emissions.co_1a1 * coff_1a1['gas']
            sum_co += emissions.ene_1a1_biomass / emissions.ene_1a * emissions.co_1a1 * coff_1a1['biofuels']
            sum_co += emissions.ene_1a1_liquid / emissions.ene_1a * emissions.co_1a1 * coff_1a1['oil']
            sum_co += emissions.ene_1a1_solid / emissions.ene_1a * emissions.co_1a1 * coff_1a1['coal']
            sum_co += emissions.ene_1a1_other / emissions.ene_1a * emissions.co_1a1 * coff_1a1['other']

            # 1A2
            coff_1a2 = profiles.get_manufacturing_month_coffs(code, month)

            sum_co += coff_1a2['1A2a'] * emissions.co_1a2a
            sum_co += coff_1a2['1A2b'] * emissions.co_1a2b
            sum_co += coff_1a2['1A2c'] * emissions.co_1a2c
            sum_co += coff_1a2['1A2d'] * emissions.co_1a2d
            sum_co += coff_1a2['1A2f'] * emissions.co_1a2f
            sum_co += coff_1a2['1A2g'] * emissions.co_1a2g

            # 1A3
            coff_1a3 = profiles.get_transport_month_coffs(code, month)
            sum_co += coff_1a3['1A3a'] * emissions.co_1a3a
            sum_co += coff_1a3['1A3b'] * emissions.co_1a3b
            sum_co += coff_1a3['1A3c'] * emissions.co_1a3c
            sum_co += coff_1a3['1A3d'] * emissions.co_1a3d
            sum_co += coff_1a3['1A3e'] * emissions.co_1a3e

            # 1A4
            coff_1a4 = profiles.get_residential_and_other_sector(country, 2017, month)
            sum_co += coff_1a4['1A4'] * emissions.co_1a4
            sum_co += 1 / 12 * emissions.co_1b
            sum_co += 1 / 12 * emissions.co_1c

            # 2A
            coff_2a = profiles.get_mineral_industry_coffs(code, month)
            sum_co += coff_2a['2A1'] * emissions.co_2a1
            sum_co += coff_2a['2A2'] * emissions.co_2a2
            sum_co += coff_2a['2A3'] * emissions.co_2a3
            sum_co += coff_2a['2A4'] * emissions.co_2a4

            # 2B
            coff_2b = profiles.get_chemical_industry_coffs(code, month)
            sum_co += coff_2b['2B1'] * emissions.co_2b1
            sum_co += coff_2b['2B2'] * emissions.co_2b2
            sum_co += coff_2b['2B3'] * emissions.co_2b3
            sum_co += coff_2b['2B4'] * emissions.co_2b4
            sum_co += coff_2b['2B5'] * emissions.co_2b5
            sum_co += coff_2b['2B6'] * emissions.co_2b6
            sum_co += coff_2b['2B7'] * emissions.co_2b7
            sum_co += coff_2b['2B8'] * emissions.co_2b8
            sum_co += coff_2b['2B9'] * emissions.co_2b9
            sum_co += coff_2b['2B10'] * emissions.co_2b10

            # 2C
            coff_2c = profiles.get_metal_industry_coffs(code, month)
            sum_co += coff_2c['2C1'] * emissions.co_2c1
            sum_co += coff_2c['2C2'] * emissions.co_2c2
            sum_co += coff_2c['2C3'] * emissions.co_2c3
            sum_co += coff_2c['2C4'] * emissions.co_2c4
            sum_co += coff_2c['2C5'] * emissions.co_2c5
            sum_co += coff_2c['2C6'] * emissions.co_2c6
            sum_co += coff_2c['2C7'] * emissions.co_2c7

            # 2D
            coff_2d = profiles.get_non_energy_product_coffs(code, month)
            sum_co += coff_2d['2D1'] * emissions.co_2d1
            sum_co += coff_2d['2D2'] * emissions.co_2d2
            sum_co += coff_2d['2D3'] * emissions.co_2d3

            # 2E
            coff_2e = profiles.get_electronics_industry_coffs(code, month)
            sum_co += coff_2e['2E1'] * emissions.co_2e1
            sum_co += coff_2e['2E2'] * emissions.co_2e2
            sum_co += coff_2e['2E3'] * emissions.co_2e3
            sum_co += coff_2e['2E4'] * emissions.co_2e4
            sum_co += coff_2e['2E5'] * emissions.co_2e5

            # 2F
            coff_2f = profiles.get_substitutes_for_ods_coffs(code, month)
            sum_co += coff_2f['2F1'] * emissions.co_2f1
            sum_co += coff_2f['2F2'] * emissions.co_2f2
            sum_co += coff_2f['2F3'] * emissions.co_2f3
            sum_co += coff_2f['2F4'] * emissions.co_2f4
            sum_co += coff_2f['2F5'] * emissions.co_2f5
            sum_co += coff_2f['2F6'] * emissions.co_2f6

            # 2G
            coff_2g = profiles.get_other_product_use_coffs(code, month)
            sum_co += coff_2g['2G1'] * emissions.co_2g1
            sum_co += coff_2g['2G2'] * emissions.co_2g2
            sum_co += coff_2g['2G3'] * emissions.co_2g3
            sum_co += coff_2g['2G4'] * emissions.co_2g4

            # 2H
            sum_co += emissions.co_2h * profiles.get_other_coffs(code, month)

            # 3 4 5 6
            sum_co += (emissions.co_3 + emissions.co_4 + emissions.co_5 + emissions.co_6) * profiles.get_other_coffs(code,
                                                                                                                     month)

        print('%s\t%.2f' % (country, sum_co))

    pass
