# Author: Axel Mukwena
# ECG Biometric Authentication using CNN

import argparse
from signals import GetSignals
from features import GetFeatures
from setup import Setup
import CTFN


mit = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
       '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
       '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
       '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
       '222', '223', '228', '230', '231', '232', '233', '234']

# exclude person 74
ecgid = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
         '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
         '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
         '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
         '41', '42', '43', '44', '45', '46', '47', '48', '49', '50',
         '51', '52', '53', '54', '55', '56', '57', '58', '59', '60',
         '61', '62', '63', '64', '65', '66', '67', '68', '69', '70',
         '71', '72', '73', '75', '76', '77', '78', '79', '80',
         '81', '82', '83', '84', '85', '86', '87', '88', '89', '90']

bmd101 = ["1975", "1973"]

AM = [f'AM{i:d}' for i in range(6, 18)]


nsrdb = ['18184','19088','19090','19093','19140','19830','16265','16272','16273','16420','16483','16539','16773','16786','16795','17052','17453','18177']

long = ['AA', 'ABD', 'AC', 'ACA', 'AFS', 'AG', 'AL', 'AR', 'ARA', 'ARF', 'ARL', 'CB', 'CF', 'CSR', 'DB', 'DC', 'DS', 'FM', 'FO', 'FP', 'GF', 'HF', 'IB', 'IC', 'JA', 'JB', 'JC', 'JCA', 'JCC', 'JL', 'JM', 'JN', 'JP', 'JPA', 'JS', 'JSA', 'JV', 'MA', 'MB', 'MBA', 'MC', 'MGA', 'MJR', 'MMJ', 'MP', 'MQ', 'PES', 'PM', 'PMA', 'RA', 'RAA', 'RD', 'RF', 'RL', 'RR', 'RRA', 'SF', 'SR', 'TC', 'TF', 'TV', 'VM', 'VO']
# short = ['AJR', 'ALM', 'AMA', 'ARL', 'ARS', 'AS', 'ASN', 'AV', 'CB', 'CC', 'CGP', 'CIB', 'CO', 'DPC', 'DS', 'DT', 'EP', 'ES', 'FAC', 'FC', 'GD', 'GFN', 'GM', 'HPS', 'IA', 'JC', 'JF', 'JG', 'JL', 'JMDA', 'JMF', 'JN', 'JS', 'JTA', 'JTP', 'JV', 'LCR', 'LDS', 'LGM', 'LR', 'LSM', 'MB', 'MC', 'MLS', 'MNM', 'MR', 'MRN', 'MVA', 'NF', 'NPS', 'PC', 'PLC', 'PLN', 'PME', 'RCN', 'RMAF', 'RN', 'RSB', 'SM', 'SMS', 'SS', 'TCO', 'TMM', 'VRR', 'XZ']
short = ['AJR', 'ALM', 'AMA', 'ARL', 'ARS', 'AS', 'ASN', 'AV', 'CB', 'CC', 'CGP', 'CO', 'DPC', 'DS', 'DT', 'EP', 'ES', 'FAC', 'FC', 'GD', 'GFN', 'GM', 'HPS', 'IA', 'JC', 'JF', 'JG', 'JL', 'JMDA', 'JMF', 'JN', 'JS', 'JTA', 'JTP', 'JV', 'LCR', 'LDS', 'LGM', 'LR', 'LSM', 'MB', 'MC', 'MLS', 'MNM', 'MR', 'MRN', 'MVA', 'NF', 'NPS', 'PC', 'PLC', 'PLN', 'PME', 'RCN', 'RMAF', 'RN', 'RSB', 'SM', 'SMS', 'SS', 'TCO', 'TMM', 'VRR', 'XZ']

def main():
    if arg.signals_mit:
        try:
            gs = GetSignals()
            gs.mit(mit)
        except Exception as e:
            print(e)

    elif arg.signals_ecgid:
        try:
            gs = GetSignals()
            gs.ecgid(ecgid)
        except Exception as e:
            print(e)
    elif arg.signals_bmd:
        try:
            gs = GetSignals()
            gs.bmd(bmd101)
        except Exception as e:
            print(e)
    elif arg.signals_nsrdb:
        try:
            gs = GetSignals()
            gs.nsrdb()
        except Exception as e:
            print(e)

    elif arg.features_mit:
        try:
            feats = GetFeatures()
            feats.features('mit', mit,360)
        except Exception as e:
            print(e)
    elif arg.features_ecgid:
        try:
            feats = GetFeatures()
            feats.features('ecgid', ecgid,500)
        except Exception as e:
            print(e)
    elif arg.features_bmd:
        try:
            feats = GetFeatures()
            feats.features('bmd', bmd101)
        except Exception as e:
            print(e)
    elif arg.features_AM:
        try:
            feats = GetFeatures()
            feats.features('AM', AM)
        except Exception as e:
            print(e)
    elif arg.features_nsrdb:
        try:
            feats = GetFeatures()
            feats.features('nsrdb', nsrdb,128)
        except Exception as e:
            print(e)
    elif arg.features_now:
        try:
            feats = GetFeatures()
            feats.features('now', long,1000)
        except Exception as e:
            print(e)
    elif arg.features_later:
        try:
            feats = GetFeatures()
            feats.features('later', long,1000)
        except Exception as e:
            print(e)
    elif arg.features_85:
        try:
            feats = GetFeatures()
            feats.features('85', short,1000)
        except Exception as e:
            print(e)
    elif arg.features_8B:
        try:
            feats = GetFeatures()
            feats.features('8B', short,1000)
        except Exception as e:
            print(e)
    elif arg.features_lowB:
        try:
            feats = GetFeatures()
            feats.features('lowB', short,1000)
        except Exception as e:
            print(e)
    elif arg.features_low5:
        try:
            feats = GetFeatures()
            feats.features('low5', short,1000)
        except Exception as e:
            print(e)
    elif arg.features_highB:
        try:
            feats = GetFeatures()
            feats.features('highB', short,1000)
        except Exception as e:
            print(e)
    elif arg.features_high5:
        try:
            feats = GetFeatures()
            feats.features('high5', short,1000)
        except Exception as e:
            print(e)
    elif arg.features_CI:
        try:
            feats = GetFeatures()
            feats.features('CI', short,1000)
        except Exception as e:
            print(e)
    elif arg.setup:
        try:
            su = Setup()
            # su.load_signals(2400, "mit_train", mit, 0)
            # su.load_signals(600, "mit_test", mit, 0)
            # su.load_signals(400, "ecgid_train", ecgid, 0)
            # su.load_signals(100, "ecgid_test", ecgid, 0)
            # su.load_signals(40, "AM", AM, 0)
            # su.load_signals(300, "Amit", mit, 0)
            # su.load_signals(100, "Aecgid", ecgid, 0)
            # su.load_signals(300, "Ansrdb", nsrdb, 0)
            su.load_signals(1000, "nsrdb", nsrdb, 0)
            # su.load_signals(300, "now", long, 0)
            # su.load_signals(300, "later", long, 0)
            # su.load_signals(300, "85", short, 0)
            # su.load_signals(300, "8B", short, 0)
            # su.load_signals(300, "highB", short, 0)
            # su.load_signals(300, "CI", short, 0)
        except Exception as e:
            print(e)

    elif arg.snn:
        try:
            snn.main()
        except Exception as e:
            print(e)
    elif arg.cnn:
        try:
            cnn.main()
        except Exception as e:
            print(e)
    elif arg.CTFN:
        try:
            CTFN.main()
        except Exception as e:
            print(e)
    elif arg.AM_train:
        try:
            AM_train.main()
        except Exception as e:
            print(e)      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s-mit', '--signals_mit', nargs='?', const=True, default=False)
    parser.add_argument('-s-ecgid', '--signals_ecgid', nargs='?', const=True, default=False)
    parser.add_argument('-s-bmd', '--signals_bmd', nargs='?', const=True, default=False)
    parser.add_argument('-s-nsrdb', '--signals_nsrdb', nargs='?', const=True, default=False)




    parser.add_argument('-f-mit', '--features_mit', nargs='?', const=True, default=False)
    parser.add_argument('-f-ecgid', '--features_ecgid', nargs='?', const=True, default=False)
    parser.add_argument('-f-bmd', '--features_bmd', nargs='?', const=True, default=False)
    parser.add_argument('-f-AM', '--features_AM', nargs='?', const=True, default=False)
    parser.add_argument('-f-nsrdb', '--features_nsrdb', nargs='?', const=True, default=False)
    parser.add_argument('-f-now', '--features_now', nargs='?', const=True, default=False)
    parser.add_argument('-f-later', '--features_later', nargs='?', const=True, default=False)
    parser.add_argument('-f-85', '--features_85', nargs='?', const=True, default=False)
    parser.add_argument('-f-8B', '--features_8B', nargs='?', const=True, default=False)
    parser.add_argument('-f-lowB', '--features_lowB', nargs='?', const=True, default=False)
    parser.add_argument('-f-low5', '--features_low5', nargs='?', const=True, default=False)
    parser.add_argument('-f-highB', '--features_highB', nargs='?', const=True, default=False)
    parser.add_argument('-f-high5', '--features_high5', nargs='?', const=True, default=False)
    parser.add_argument('-f-CI', '--features_CI', nargs='?', const=True, default=False)



    parser.add_argument('-setup', '--setup', nargs='?', const=True, default=False)

    parser.add_argument('-snn', '--snn', nargs='?', const=True, default=False)
    parser.add_argument('-cnn', '--cnn', nargs='?', const=True, default=False)
    parser.add_argument('-toimage', '--toimage', nargs='?', const=True, default=False)
    parser.add_argument('-transformer', '--transformer', nargs='?', const=True, default=False)
    parser.add_argument('-AM_train', '--AM_train', nargs='?', const=True, default=False)


    arg = parser.parse_args()

    main()

