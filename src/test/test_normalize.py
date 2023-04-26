import csv
import re

import numpy as np
import requests

BASE_URL = "https://rxnav.nlm.nih.gov/REST"


def query_rxnorm_api(drug_name):
    url = f'{BASE_URL}/approximateTerm.json?term={drug_name}&maxEntries=5'
    response = requests.get(url)
    data = response.json()
    try:
        if data['approximateGroup']['candidate'][0]['rxcui']:
            for candidate in data['approximateGroup']['candidate']:
                rxcui = candidate['rxcui']
                generic_name = get_generic_name(rxcui)
                if generic_name:
                    return generic_name
    except (IndexError, KeyError):
        return None


def get_generic_name(rxcui):
    url = f'{BASE_URL}/rxcui/{rxcui}/related.json?tty=IN'
    response = requests.get(url)
    data = response.json()
    try:
        generic_rxcui = data['relatedGroup']['conceptGroup'][0]['conceptProperties'][0]['rxcui']
        generic_name = data['relatedGroup']['conceptGroup'][0]['conceptProperties'][0]['name']
        return generic_name
    except (IndexError, KeyError):
        return None


if __name__ == '__main__':
    input_file = 'drugnames.csv'
    output_file = 'norm_drugs.csv'
    norm_drug_names = []
    with open(input_file, newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            original_drug_name = row[0]
            # normalized_drug_name = re.sub(r'[^a-zA-Z0-9 ]', '', original_drug_name).strip()

            normalized_drug_name = re.sub(r'\([^)]*\)', ' ', original_drug_name)
            normalized_drug_name = re.sub(r'[^\w\s]', ' ', normalized_drug_name)  # remove non-alphanumeric characters
            normalized_drug_name = re.sub(r'(?<!\w)\d+(?!\w)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(?<!\w)\d+[a-zA-Z]+', ' ', normalized_drug_name)
            # normalized_drug_name = re.sub(r'(^|\s|\.)[0-9]+($|\s)', ' ', normalized_drug_name)
            # normalized_drug_name = re.sub(r'(^|\s|\.)[0-9]+($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)[a-zA-Z]{1}($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)PO($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)MG($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)ML($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)TABS($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)SOLN($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)CAPS($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)CAP($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)CAPSULE($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)LIQD($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)BAG($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)IV($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)INJ($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)SW($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)IN($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)EACH($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)VIAL($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)PACKAGE($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)INJECTION($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)SOLR($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)GEL($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)EACH($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)IJ($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)USE($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)MT($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)UNIT($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)TABLET($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)TAB($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)INFUSION($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)USE($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)INH($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)NEB($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)SOLR($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)LIQ($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)SYRINGE($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)FLEX($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)CONT($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)LOCK($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)BASE($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)ACT($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)AERS($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)CONT($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)LVP($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)SOLUTION($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)FOR($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)ER($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)TB12($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)UNITS($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)PF($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)SUSP($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)IM($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)NDC($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)HCL($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)MCG($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)WATER($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)STERILE($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)PLAS($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)FAT($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)CUSTOM($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)NDC($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)EXCEL($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)FLUSH($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)MCG($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)HFA($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)RE($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)EC($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)TBEC($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)GRAM($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)IVPB($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)ORAL($|\s)', ' ', normalized_drug_name)
            normalized_drug_name = re.sub(r'(^|\s)RINSE($|\s)', ' ', normalized_drug_name)

            normalized_drug_name = re.sub(r"\b(?=[MDCLXVI]+\b)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b",
                                          ' ', normalized_drug_name)
            normalized_drug_name = normalized_drug_name.replace('DEXT', 'DEXTROSE')

            # text = text.replace('PO', '-')

            # normalized_drug_name = re.sub(r'(^|\s|\.):($|\s)', ' ', original_drug_name)
            # normalized_drug_name = re.sub(r'(^|\s|\.)-($|\s)', ' ', normalized_drug_name)
            # normalized_drug_name = re.sub(r'[0-9]+-[0-9]+', ' ', normalized_drug_name)
            # normalized_drug_name = normalized_drug_name.replace('-%', ' ')
            normalized_drug_name = re.sub(r'\s+', ' ', normalized_drug_name).strip()
            # normalized_drug_name = original_drug_name
            if generic_name := query_rxnorm_api(normalized_drug_name):
                normalized_drug_name = generic_name
            else:
                generic_name = query_rxnorm_api(original_drug_name)
                if generic_name:
                    normalized_drug_name = generic_name

            result = f"ORIGINAL={original_drug_name}, NORM={normalized_drug_name}"
            print(result)
            norm_drug_names.append(result)
            writer.writerow([original_drug_name, normalized_drug_name])
        norm_drug_names = np.array(norm_drug_names)
        np.expand_dims(np.array(norm_drug_names), axis=1)
        np.savetxt('norm_drug string.txt', norm_drug_names, fmt='%s')
