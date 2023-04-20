import re

all_types = [('ESI-Orbitrap', 180723), ('ESI-qTof', 30244), ('LC-ESI-Orbitrap', 27333), ('N/A-ESI-QFT', 26880),
             ('ESI-Hybrid FT', 19469), ('LC-ESI-qTof', 18514), ('LC-ESI-HCD; Velos', 15141),
             ('LC-ESI-CID; Velos', 15133), ('LC-ESI-Maxis II HD Q-TOF Bruker', 7340), ('ESI-QQQ', 6422),
             ('ESI-LC-ESI-ITFT', 4732), ('ESI-Flow-injection QqQ/MS', 3215), ('N/A-Linear Ion Trap', 3016),
             ('LC-ESI-Ion Trap', 2939), ('ESI-Ion Trap', 2913), ('ESI-LC-ESI-QTOF', 2771), ('ESI-LC-ESI-QQ', 2404),
             ('N/A-ESI-QTOF', 2350), ('ESI-qToF', 2201), ('LC-ESI-ITFT-LC-ESI-ITFT', 1089), ('ESI-LC-ESI-QFT', 1056),
             ('ESI-LC-Q-TOF/MS', 796), ('LC-ESI- impact HD', 733), ('DI-ESI-qTof', 582),
             ('Positive-Quattro_QQQ:40eV', 562), ('Positive-Quattro_QQQ:25eV', 556),
             ('Positive-Quattro_QQQ:10eV', 515), ('LC-ESI-Q-Exactive Plus', 463), ('LC-ESI-qToF', 440),
             ('LC-ESI-CID; Lumos', 431), ('LC-ESI-HCD; Lumos', 430), ('N/A-N/A', 352), ('ESI-LC-APPI-QQ', 248),
             ('ESI-LC-ESI-IT', 207), ('ESI-FAB-EBEB', 163), ('-Q-Exactive Plus Orbitrap Res 70k', 155),
             ('LC-ESI-qTOF', 135), ('-Maxis HD qTOF', 134), ('-Q-Exactive Plus Orbitrap Res 14k', 118),
             ('LC-ESI-Q-Exactive Plus Orbitrap Res 70k', 114), ('LC-ESI-Maxis HD qTOF', 110),
             ('Negative-Quattro_QQQ:40eV', 100), ('DI-ESI-Orbitrap', 99), ('Negative-Quattro_QQQ:25eV', 98),
             ('DI-ESI-Hybrid FT', 96), ('LC-ESI-Hybrid FT', 95), ('LC-ESI-Q-Exactive Plus Orbitrap Res 14k', 78),
             ('Negative-Quattro_QQQ:10eV', 76), ('DI-ESI-Ion Trap', 63), ('ESI-HCD', 49), ('ESI-APCI-ITFT', 42),
             ('LC-ESI-QQQ', 41), ('ESI-Q-TOF', 39), ('ESI-ESI-ITFT', 29), ('ESI-qTOF', 18), ('ESI-LC-ESI-Q', 17),
             ('ESI-LC-ESI-ITTOF', 16), ('LC-APCI-qTof', 15), ('LC-ESI-LTQ-FTICR', 8), ('LC-ESI-QQ-LC-ESI-QQ', 7),
             ('ESI-IT/ion trap', 7), ('LC-ESIMS-qTOF', 5), ('LC-ESI-Hybrid Ft', 5), ('APCI-Ion Trap', 5),
             ('ESI-QqQ', 5), ('ESI-IT-FT/ion trap with FTMS', 4), ('DI-ESI-Q-Exactive Plus', 3), ('APCI-Orbitrap', 3),
             ('ESI or APCI-IT/ion trap', 3), ('ESI-UPLC-ESI-QTOF', 3), ('LC-ESI-QTOF-LC-ESI-QTOF', 3),
             ('LC-ESI-LCQ', 2), ('CI (MeOH)-IT/ion trap', 2), ('In-source CID-API', 2), ('FAB-BEqQ/magnetic and electric sectors with quadrupole', 2), ('APCI-QQQ', 2), ('APCI-qTof', 2), ('ESI-ESI-FTICR', 2), ('ESI-HPLC-ESI-TOF', 2), ('ESI-QIT', 2), ('DIRECT INFUSION NANOESI-ION TRAP-DIRECT INFUSION NANOESI-ION TRAP', 1), ('DI-ESI-Q-Exactive', 1), ('DI-ESI-QQQ', 1), ('EI-QQQ', 1), ('in source ESI-QqQ', 1), ('CI-IT/ion trap', 1), ('LC-ESI-ITTOF-LC-ESI-ITTOF', 1)]


def create_keyword_list(regexp_list, name_list):
    """This function uses the regexp objects defined above and the list of names output by unzip_types_occurrences.
    returns a list of all the matches between the search terms and the instrument type entries found in the dataset"""
    match_list = list(filter(regexp_list.match, name_list))
    return match_list

if __name__ == "__main__":
    all_types_dict = dict(all_types)
    all_types_list = list(all_types_dict.keys())
    orbitrap_regexp_terms = re.compile(".*orbitrap.*|.*hcd.*|.*q-exactive.*|.*lumos.*|.*velos.*", re.IGNORECASE)
    tof_regexp_terms = re.compile(".*tof.*|.*impact*", re.IGNORECASE)
    ft_regexp_terms = re.compile(".*FT.*", re.IGNORECASE)
    quadrupole_regexp_terms = re.compile(".*QQQ.*|.*QQ.*|.*Quadrupole.*", re.IGNORECASE)
    iontrap_regexp_terms = re.compile(".*Ion trap.*|.*IT$", re.IGNORECASE)

    orbitrap_matches = create_keyword_list(orbitrap_regexp_terms, all_types_list)
    tof_matches = create_keyword_list(tof_regexp_terms, all_types_list)
    ft_matches = create_keyword_list(ft_regexp_terms, all_types_list)
    quadrupole_matches = create_keyword_list(quadrupole_regexp_terms, all_types_list)
    iontrap_matches = create_keyword_list(iontrap_regexp_terms, all_types_list)
    quadrupole_matches.append('ESI-LC-ESI-Q')
    iontrap_matches.remove('ESI-IT-FT/ion trap with FTMS')
    ft_matches.append('ESI-IT-FT/ion trap with FTMS')

    not_selected = set(all_types_list).difference(set(orbitrap_matches+tof_matches+ft_matches+quadrupole_matches+iontrap_matches))
    instrument_aliases = {}
    instrument_aliases["Orbitrap"] = orbitrap_matches
    instrument_aliases["TOF"] = tof_matches
    instrument_aliases["Fourier Transform"] = ft_matches
    instrument_aliases["Quadrupole"] = quadrupole_matches
    instrument_aliases["Ion Trap"] = iontrap_matches
    print(instrument_aliases)


    intrument_type_categories = {'Orbitrap': ['ESI-Orbitrap', 'LC-ESI-Orbitrap', 'LC-ESI-HCD; Velos', 'LC-ESI-CID; Velos', 'LC-ESI-Q-Exactive Plus', 'LC-ESI-CID; Lumos', 'LC-ESI-HCD; Lumos', '-Q-Exactive Plus Orbitrap Res 70k', '-Q-Exactive Plus Orbitrap Res 14k', 'LC-ESI-Q-Exactive Plus Orbitrap Res 70k', 'DI-ESI-Orbitrap', 'LC-ESI-Q-Exactive Plus Orbitrap Res 14k', 'ESI-HCD', 'DI-ESI-Q-Exactive Plus', 'APCI-Orbitrap', 'DI-ESI-Q-Exactive'],
              'TOF': ['ESI-qTof', 'LC-ESI-qTof', 'LC-ESI-Maxis II HD Q-TOF Bruker', 'ESI-LC-ESI-QTOF', 'N/A-ESI-QTOF', 'ESI-qToF', 'ESI-LC-Q-TOF/MS', 'LC-ESI- impact HD', 'DI-ESI-qTof', 'LC-ESI-qToF', 'LC-ESI-qTOF', '-Maxis HD qTOF', 'LC-ESI-Maxis HD qTOF', 'ESI-Q-TOF', 'ESI-qTOF', 'ESI-LC-ESI-ITTOF', 'LC-APCI-qTof', 'LC-ESIMS-qTOF', 'ESI-UPLC-ESI-QTOF', 'LC-ESI-QTOF-LC-ESI-QTOF', 'APCI-qTof', 'ESI-HPLC-ESI-TOF', 'LC-ESI-ITTOF-LC-ESI-ITTOF'],
              'Fourier Transform': ['N/A-ESI-QFT', 'ESI-Hybrid FT', 'ESI-LC-ESI-ITFT', 'LC-ESI-ITFT-LC-ESI-ITFT', 'ESI-LC-ESI-QFT', 'DI-ESI-Hybrid FT', 'LC-ESI-Hybrid FT', 'ESI-APCI-ITFT', 'ESI-ESI-ITFT', 'LC-ESI-LTQ-FTICR', 'LC-ESI-Hybrid Ft', 'ESI-IT-FT/ion trap with FTMS', 'ESI-ESI-FTICR', 'ESI-IT-FT/ion trap with FTMS'],
              'Quadrupole': ['ESI-QQQ', 'ESI-Flow-injection QqQ/MS', 'ESI-LC-ESI-QQ', 'Positive-Quattro_QQQ:40eV', 'Positive-Quattro_QQQ:25eV', 'Positive-Quattro_QQQ:10eV', 'ESI-LC-APPI-QQ', 'Negative-Quattro_QQQ:40eV', 'Negative-Quattro_QQQ:25eV', 'Negative-Quattro_QQQ:10eV', 'LC-ESI-QQQ', 'LC-ESI-QQ-LC-ESI-QQ', 'ESI-QqQ', 'FAB-BEqQ/magnetic and electric sectors with quadrupole', 'APCI-QQQ', 'DI-ESI-QQQ', 'EI-QQQ', 'in source ESI-QqQ', 'ESI-LC-ESI-Q'],
              'Ion Trap': ['N/A-Linear Ion Trap', 'LC-ESI-Ion Trap', 'ESI-Ion Trap', 'ESI-LC-ESI-IT', 'DI-ESI-Ion Trap', 'ESI-IT/ion trap', 'APCI-Ion Trap', 'ESI or APCI-IT/ion trap', 'CI (MeOH)-IT/ion trap', 'ESI-QIT', 'DIRECT INFUSION NANOESI-ION TRAP-DIRECT INFUSION NANOESI-ION TRAP', 'CI-IT/ion trap']}
