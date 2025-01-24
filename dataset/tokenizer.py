import re


"""
iupac tokenizer
"""
def smiles_atom_tokenizer (smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def split_tokens(tokens_to_be_splitted, all_tokens):
    if ''.join(tokens_to_be_splitted) in all_tokens:
        indexes = duplicates(all_tokens, ''.join(tokens_to_be_splitted))
        for k, idx in enumerate(indexes):
            all_tokens[idx] = tokens_to_be_splitted[0]
            for i in range(1, len(tokens_to_be_splitted)):
                all_tokens.insert(idx+i, tokens_to_be_splitted[i])
            if k < len(indexes) - 1:
                indexes[k+1:] = list(map(lambda x: x+(len(tokens_to_be_splitted) - 1), indexes[k+1:]))
def add_symbol(pat):
    new_patt = pat[1:].split(',')
    ls_del = []
    if new_patt[0][0] == '0':
        ls_del.append(new_patt[0][0])
        new_patt[0] = new_patt[0][1:]

    while int(new_patt[0]) > int(new_patt[1]):
        ls_del.append(new_patt[0][0])
        new_patt[0] = new_patt[0][1:]
    result = ['.', ''.join(ls_del), '^', new_patt[0], ',', new_patt[1]]
    return result

def check_correct_tokenize(iupac,tokens):
    if ''.join(tokens).replace('^', '') == iupac:
        return True
    else:
        return False

def iupac_tokenizer(iupac):

    pattern = "\.\d+,\d+|[1-9]{2}\-[a-z]\]|[0-9]\-[a-z]\]|[1-9]{2}[a-z]|[1-9]{2}'[a-z]|[0-9]'[a-z]|[0-9][a-z]|\([0-9]\+\)|\([0-9]\-\)|" + \
              "[1-9]{2}|[0-9]|-|\s|\(|\)|S|R|E|Z|N|C|O|'|\"|;|λ|H|,|\.|\[[a-z]{2}\]|\[[a-z]\]|\[|\]|"

    alcane = "methane|methanoyl|methan|ethane|ethanoyl|ethan|propanoyl|propane|propan|propa|butane|butanoyl|butan|buta|pentane|" + \
             "pentanoyl|pentan|hexane|hexanoyl|hexan|heptane|heptanoyl|heptan|octane|octanoyl|octan|nonane|nonanoyl|" + \
             "nonan|decane|decanoyl|decan|icosane|icosan|cosane|cosan|contane|contan|"

    pristavka_name = "hydroxide|hydroxyl|hydroxy|hydrate|hydro|cyclo|spiro|iso|"
    pristavka_digit = "mono|un|bis|bi|dicta|di|tetraza|tetraz|tetra|tetr|pentaza|pentaz|penta|hexaza|" + \
                      "hexa|heptaza|hepta|octaza|octa|nonaza|nona|decaza|deca|kis|"

    prefix_alcane = "methylidene|methyl|ethyl|isopropyl|propyl|isobutyl|sec-butyl|tert-butyl|butyl|pentyl|hexyl|heptyl|octyl|"
    carbon = "meth|eth|prop|but|pent|hex|hept|oct|non|dec|icosa|icos|cosa|cos|icon|conta|cont|con|heni|hene|hen|hecta|hect|"

    prefix_all = "benzhydryl|benzoxaza|benzoxaz|benzoxa|benzox|benzo|benzyl|benz|phenacyl|phenanthro|phenyl|phenoxaza|phenoxaz|phenoxy|phenox|phenol|pheno|phen|acetyl|aceto|acet|" + \
                 "peroxy|oxido|oxino|oxalo|oxolo|oxocyclo|oxol|oxoc|oxon|oxo|oxy|pyrido|pyrimido|imidazo|naphtho|stiboryl|stibolo|"

    prefix_isotope = "protio|deuterio|tritio|"
    suffix_isotope = "protide|"
    prefix_galogen = "fluoro|fluoranyl|fluoridoyl|fluorido|chloro|chloranyl|chloridoyl|chlorido|bromo|bromanyl|bromidoyl|bromido|iodo|iodanyl|iodidoyl|iodanuidyl|iodido|"
    suffix_galogen = "fluoride|chloride|chloridic|perchloric|bromide|iodide|iodane|hypoiodous|hypochlorous|"
    prefix_сhalcogen = "phosphonato|phosphoroso|phosphonia|phosphoryl|phosphanyl|arsono|arsanyl|stiba|"
    suffix_сhalcogen = "phosphanium|phosphate|phosphite|phosphane|phosphanide|phosphonamidic|phosphonous|phosphinous|phosphinite|phosphono|arsonic|stibane|"
    prefix_metal = "alumanyl|gallanyl|stannyl|plumbyl|"
    suffix_metal = "chromium|stannane|gallane|alumane|aluminane|aluminan|"
    prefix_non_metal = "tellanyl|germanyl|germyl|"
    suffix_non_metal = "germane|germa|"

    prefix_sulfur = "sulfanylidene|sulfinamoyl|sulfonimidoyl|sulfinimidoyl|sulfamoyl|sulfonyl|sulfanyl|sulfinyl|sulfinato|sulfenato|" + \
                    "sulfonato|sulfonio|sulfino|sulfono|sulfido|"
    suffix_sulfur = "sulfonamide|sulfinamide|sulfonamido|sulfonic|sulfamic|sulfinic|sulfuric|thial|thione|thiol|" + \
                    "sulfonate|sulfite|sulfate|sulfide|sulfinate|sulfanium|sulfamate|sulfane|sulfo|"

    prefix_nitrogen = "hydrazono|hydrazino|nitroso|nitrous|nitro|formamido|amino|amido|imino|imido|anilino|anilin|thiocyanato|cyanato|cyano|azido|azanidyl|azanyl|" + \
                      "azanide|azanida|azonia|azonio|amidino|nitramido|diazo|"
    suffix_nitrogen = "ammonium|hydrazide|hydrazine|hydrazin|amine|imine|oxamide|nitramide|formamide|cyanamide|amide|imide|amidine|isocyanide|azanium|" + \
                      "thiocyanate|cyanate|cyanic|cyanatidoyl|cyanide|nitrile|nitrite|hydrazonate|"

    suffix_carbon = "carbonitrile|carboxamide|carbamimidothioate|carbodithioate|carbohydrazonate|carbonimidoyl|carboximidoyl|" + \
                    "carbamimidoyl|carbamimidate|carbamimid|carbaldehyde|carbamate|carbothioyl|carboximidothioate|carbonate|" + \
                    "carboximidamide|carboximidate|carbamic|carbonochloridate|carbothialdehyde|carbothioate|carbothioic|carbono|carbon|carbo|" + \
                    "formate|formic|"
    prefix_carbon = "carboxylate|carboxylato|carboxylic|carboxy|halocarbonyl|carbamoyl|carbonyl|carbamo|thioformyl|formyl|"

    silicon = "silanide|silane|silole|silanyl|silyloxy|silylo|silyl|sila|"
    boron = "boranyl|boranuide|boronamidic|boranuida|boranide|borinic|borate|borane|boran|borono|boron|bora|"
    selenium = "selanyl|seleno|"

    suffix_all = "ane|ano|an|ene|enoxy|eno|en|yne|yn|yl|peroxol|peroxo|" + \
                 "terephthalate|terephthalic|phthalic|phthalate|oxide|oate|ol|oic|ic|al|ate|ium|one|"

    carbon_trivial = "naphthalen|naphthal|inden|adamant|fluoren|thiourea|urea|anthracen|acenaphthylen|" + \
                     "carbohydrazide|annulen|aniline|acetaldehyde|benzaldehyde|formaldehyde|phthalaldehyde|acephenanthrylen|" + \
                     "phenanthren|chrysen|carbanid|chloroform|fulleren|cumen|formonitril|fluoranthen|terephthalaldehyde|azulen|picen|" + \
                     "pyren|pleiaden|coronen|tetracen|pentacen|perylen|pentalen|heptalen|cuban|hexacen|oxanthren|ovalen|aceanthrylen|"

    heterocycles = "indolizin|arsindol|indol|furan|furo|piperazin|pyrrolidin|pyrrolizin|thiophen|thiolo|imidazolidin|imidazol|pyrimidin|pyridin|" + \
                    "piperidin|morpholin|pyrazol|pyridazin|oxocinnolin|cinnolin|pyrrol|thiochromen|oxochromen|chromen|quinazolin|phthalazin|quinoxalin|carbazol|xanthen|pyrazin|purin|" + \
                    "indazol|naphthyridin|quinolizin|guanidin|pyranthren|pyran|thianthren|thian|acridin|acrido|yohimban|porphyrin|pteridin|tetramin|pentamin|" + \
                    "borinin|borino|boriran|borolan|borol|borinan|phenanthridin|quinolin|perimidin|corrin|phenanthrolin|phosphinolin|indacen|silonin|borepin|"

    prefix_heterocycles = "thiaz|oxaza|oxaz|oxan|oxa|ox|aza|az|thia|thioc|thion|thio|thi|telluro|phospha|phosph|selen|bor|sil|alum|ars|germ|tellur|imid|"

    suffix_heterocycles = "ir|et|olo|ol|ino|in|ep|oc|on|ec|"
    saturated_unsatured = "idine|idene|idin|ane|an|ine|in|id|e|"
    pristavka_exception = "do|trisodium|tris|triacetyl|triamine|triaza|triaz|tria|trityl|tri|o"

    type_ = "acid|ether|"
    element = "hydrogen|helium|lithium|beryllium|nitrogen|oxygen|fluorine|neon|sodium|magnesium|aluminum|silicon|" + \
              "phosphorus|sulfur|chlorine|argon|potassium|calcium|scandium|titanium|vanadium|chromium|manganese|iron|" + \
              "cobalt|nickel|copper|zinc|gallium|germanium|arsenic|selenium|bromine|krypton|rubidium|yttrium|zirconium|" + \
              "niobium|molybdenum|technetium|ruthenium|rhodium|palladium|silver|cadmium|indium|antimony|tellurium|iodine|" + \
              "xenon|cesium|barium|lanthanum|cerium|praseodymium|neodymium|latinum|promethium|samarium|europium|gadolinium|" + \
              "terbium|dysprosium|holmium|erbium|thulium|ytterbium|lutetium|hafnium|tantalum|tungsten|rhenium|osmium|" + \
              "iridium|platinum|gold|aurum|mercury|thallium|lead|bismuth|polonium|astatine|radon|francium|radium|actinium|" + \
              "thorium|protactinium|uranium|neptunium|plutonium|americium|curium|berkelium|einsteinium|fermium|californium|" + \
              "mendelevium|nobelium|lawrencium|rutherfordium|dubnium|seaborgium|bohrium|hassium|meitnerium|tin|"

    other_ions = "perchlorate|perbromate|periodate|hypofluorite|hypochlorite|hypobromite|hypoiodite|nitrate|silicate|hydride|"

    regex = re.compile(pattern + heterocycles + carbon_trivial + type_ + element + prefix_isotope + other_ions + alcane + pristavka_digit + pristavka_name + prefix_alcane + \
                       carbon + silicon + prefix_nitrogen + prefix_sulfur + prefix_carbon + prefix_metal + prefix_non_metal + prefix_all + prefix_galogen + prefix_сhalcogen + \
                       suffix_carbon + suffix_nitrogen + suffix_sulfur + suffix_galogen + suffix_сhalcogen + suffix_metal + suffix_non_metal + suffix_all + suffix_heterocycles + \
                       suffix_isotope + boron + selenium  + prefix_heterocycles + saturated_unsatured + pristavka_exception)
    tokens = [token for token in regex.findall(iupac)]

    split_tokens(['meth', 'ane'], tokens)
    split_tokens(['meth', 'an'], tokens)
    split_tokens(['eth', 'ane'], tokens)
    split_tokens(['eth', 'an'], tokens)
    split_tokens(['prop', 'ane'], tokens)
    split_tokens(['prop', 'an'], tokens)
    split_tokens(['but', 'ane'], tokens)
    split_tokens(['but', 'an'], tokens)
    split_tokens(['pent', 'ane'], tokens)
    split_tokens(['pent', 'an'], tokens)
    split_tokens(['hex', 'ane'], tokens)
    split_tokens(['hex', 'an'], tokens)
    split_tokens(['hept', 'ane'], tokens)
    split_tokens(['hept', 'an'], tokens)
    split_tokens(['oct', 'ane'], tokens)
    split_tokens(['oct', 'an'], tokens)
    split_tokens(['non', 'ane'], tokens)
    split_tokens(['non', 'an'], tokens)
    split_tokens(['dec', 'ane'], tokens)
    split_tokens(['dec', 'an'], tokens)
    split_tokens(['cos', 'ane'], tokens)
    split_tokens(['cos', 'an'], tokens)
    split_tokens(['cont', 'ane'], tokens)
    split_tokens(['cont', 'an'], tokens)
    split_tokens(['icos', 'ane'], tokens)
    split_tokens(['icos', 'an'], tokens)

    split_tokens(['thi', 'az'], tokens)
    split_tokens(['thi', 'oc'], tokens)
    split_tokens(['thi', 'on'], tokens)
    split_tokens(['benz', 'ox'], tokens)
    split_tokens(['benz', 'oxa'], tokens)
    split_tokens(['benz', 'ox', 'az'], tokens)
    split_tokens(['benz', 'ox', 'aza'], tokens)
    split_tokens(['phen', 'ox'], tokens)
    split_tokens(['phen', 'oxy'], tokens)
    split_tokens(['phen', 'oxa'], tokens)
    split_tokens(['phen', 'ox', 'az'], tokens)
    split_tokens(['phen', 'ox', 'aza'], tokens)
    split_tokens(['phen', 'ol'], tokens)
    split_tokens(['en', 'oxy'], tokens)
    split_tokens(['ox', 'az'], tokens)
    split_tokens(['ox', 'aza'], tokens)
    split_tokens(['tri', 'az'], tokens)
    split_tokens(['tri', 'amine'], tokens)
    split_tokens(['tri', 'acetyl'], tokens)
    split_tokens(['ox', 'ol'], tokens)
    split_tokens(['ox', 'olo'], tokens)
    split_tokens(['ox', 'an'], tokens)
    split_tokens(['ox', 'oc'], tokens)
    split_tokens(['ox', 'on'], tokens)
    split_tokens(['tri', 'az'], tokens)
    split_tokens(['tri', 'aza'], tokens)
    split_tokens(['tri', 'sodium'], tokens)
    split_tokens(['tetr', 'az'], tokens)
    split_tokens(['tetr', 'aza'], tokens)
    split_tokens(['pent', 'az'], tokens)
    split_tokens(['pent', 'aza'], tokens)
    split_tokens(['hex', 'aza'], tokens)
    split_tokens(['hept', 'aza'], tokens)
    split_tokens(['oct', 'aza'], tokens)
    split_tokens(['non', 'aza'], tokens)
    split_tokens(['dec', 'aza'], tokens)
    split_tokens(['oxo', 'chromen'], tokens)
    split_tokens(['oxo', 'cinnolin'], tokens)
    split_tokens(['oxo', 'cyclo'], tokens)
    split_tokens(['thio', 'chromen'], tokens)
    split_tokens(['thio', 'cyanato'], tokens)

    if (len(re.findall(re.compile('[0-9]{2}\-[a-z]\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('[0-9]{2}\-[a-z]\]'), tok):
                tokens[i] = tok[:2]
                tokens.insert(i+1,tok[2])
                tokens.insert(i+2,tok[3])
                tokens.insert(i+3,tok[4])

    if (len(re.findall(re.compile('[0-9]\-[a-z]\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('[0-9]\-[a-z]\]'), tok):
                tokens[i] = tok[:1]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])
                tokens.insert(i+3,tok[3])

    if (len(re.findall(re.compile('\[[a-z]{2}\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('\[[a-z]{2}\]'), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])
                tokens.insert(i+3,tok[3])

    if (len(re.findall(re.compile('\[[a-z]\]'), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile('\[[a-z]\]'), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])

    if (len(re.findall(re.compile("[0-9]{2}'[a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9]{2}'[a-z]"), tok):
                tokens[i] = tok[:2]
                tokens.insert(i+1,tok[2])
                tokens.insert(i+2,tok[3])

    if (len(re.findall(re.compile("[0-9]'[a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9]'[a-z]"), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])
                tokens.insert(i+2,tok[2])

    if (len(re.findall(re.compile("[0-9]{2}[a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9]{2}[a-z]"), tok):
                tokens[i] = tok[:2]
                tokens.insert(i+1,tok[2])

    if (len(re.findall(re.compile("[0-9][a-z]"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("[0-9][a-z]"), tok):
                tokens[i] = tok[0]
                tokens.insert(i+1,tok[1])

    if (len(re.findall(re.compile("\.\d+,\d+"), ''.join(tokens))) > 0):
        for i, tok in enumerate(tokens):
            if re.findall(re.compile("\.\d+,\d+"), tok):
                result = add_symbol(tok)
                tokens[i] = result[0]
                for k, res in enumerate(result[1:], start=1):
                    tokens.insert(i+k,res)

    if check_correct_tokenize(iupac, tokens) == True:
        return tokens
    else:
        return None
    


class Iupac_tokenizer(object):
    def __init__(self):
        self.special_tokens =  ['[PAD]','[CLS]','[EOS]','[unk]',';', '.', '>']
        vocab = {'': 7, ' ': 8, "'": 9, '(': 10, ')': 11, ',': 12, '-': 13, '0': 14, '1': 15, '10': 16, '11': 17, '12': 18, '13': 19, '14': 20, '15': 21, '16': 22, '17': 23, '18': 24, '19': 25, '2': 26, '20': 27, '21': 28, '22': 29, '23': 30, '24': 31, '25': 32, '26': 33, '27': 34, '28': 35, '29': 36, '3': 37, '31': 38, '4': 39, '48': 40, '5': 41, '58': 42, '6': 43, '7': 44, '8': 45, '9': 46, 'C': 47, 'E': 48, 'H': 49, 'N': 50, 'O': 51, 'R': 52, 'S': 53, 'Z': 54, '[': 55, ']': 56, '^': 57, 'a': 58, 'acenaphthylen': 59, 'acet': 60, 'acetaldehyde': 61, 'aceto': 62, 'acetyl': 63, 'acid': 64, 'acridin': 65, 'adamant': 66, 'al': 67, 'amide': 68, 'amido': 69, 'amine': 70, 'amino': 71, 'an': 72, 'ane': 73, 'aniline': 74, 'anilino': 75, 'annulen': 76, 'anthracen': 77, 'ate': 78, 'az': 79, 'aza': 80, 'azanida': 81, 'azanide': 82, 'azanidyl': 83, 'azanium': 84, 'azido': 85, 'azonia': 86, 'azonio': 87, 'azulen': 88, 'b': 89, 'benz': 90, 'benzaldehyde': 91, 'benzhydryl': 92, 'benzo': 93, 'benzyl': 94, 'bi': 95, 'bis': 96, 'bromide': 97, 'bromo': 98, 'but': 99, 'buta': 100, 'butanoyl': 101, 'butyl': 102, 'c': 103, 'carbaldehyde': 104, 'carbamate': 105, 'carbamic': 106, 'carbamimidate': 107, 'carbamimidothioate': 108, 'carbamimidoyl': 109, 'carbamo': 110, 'carbamoyl': 111, 'carbazol': 112, 'carbo': 113, 'carbodithioate': 114, 'carbohydrazide': 115, 'carbohydrazonate': 116, 'carbonate': 117, 'carbonimidoyl': 118, 'carbonitrile': 119, 'carbono': 120, 'carbonochloridate': 121, 'carbonyl': 122, 'carbothialdehyde': 123, 'carbothioate': 124, 'carbothioic': 125, 'carbothioyl': 126, 'carboxamide': 127, 'carboximidamide': 128, 'carboximidate': 129, 'carboximidothioate': 130, 'carboximidoyl': 131, 'carboxy': 132, 'carboxylate': 133, 'carboxylato': 134, 'carboxylic': 135, 'chloride': 136, 'chloridoyl': 137, 'chloro': 138, 'chromen': 139, 'chrysen': 140, 'cinnolin': 141, 'conta': 142, 'cos': 143, 'cosa': 144, 'cuban': 145, 'cyanamide': 146, 'cyanato': 147, 'cyanide': 148, 'cyano': 149, 'cyclo': 150, 'd': 151, 'dec': 152, 'deca': 153, 'di': 154, 'do': 155, 'e': 156, 'ec': 157, 'en': 158, 'ene': 159, 'eno': 160, 'ep': 161, 'et': 162, 'eth': 163, 'ethyl': 164, 'f': 165, 'fluoranthen': 166, 'fluoren': 167, 'fluoride': 168, 'fluoro': 169, 'formamide': 170, 'formamido': 171, 'formate': 172, 'formic': 173, 'formyl': 174, 'furan': 175, 'furo': 176, 'g': 177, 'guanidin': 178, 'h': 179, 'heni': 180, 'hept': 181, 'hepta': 182, 'heptalen': 183, 'heptanoyl': 184, 'heptyl': 185, 'hex': 186, 'hexa': 187, 'hexanoyl': 188, 'hexyl': 189, 'hydrazide': 190, 'hydrazin': 191, 'hydrazine': 192, 'hydrazonate': 193, 'hydrazono': 194, 'hydro': 195, 'hydroxy': 196, 'hydroxyl': 197, 'hypofluorite': 198, 'ic': 199, 'icosa': 200, 'id': 201, 'idene': 202, 'idin': 203, 'idine': 204, 'imid': 205, 'imidazo': 206, 'imidazol': 207, 'imidazolidin': 208, 'imido': 209, 'imine': 210, 'imino': 211, 'in': 212, 'indazol': 213, 'inden': 214, 'indol': 215, 'indolizin': 216, 'ino': 217, 'iodo': 218, 'ir': 219, 'iso': 220, 'ium': 221, 'meth': 222, 'methyl': 223, 'methylidene': 224, 'morpholin': 225, 'naphthalen': 226, 'naphtho': 227, 'naphthyridin': 228, 'nitrate': 229, 'nitrile': 230, 'nitro': 231, 'nitroso': 232, 'nitrous': 233, 'non': 234, 'nona': 235, 'o': 236, 'oate': 237, 'oc': 238, 'oct': 239, 'octa': 240, 'octyl': 241, 'oic': 242, 'ol': 243, 'olo': 244, 'on': 245, 'one': 246, 'ox': 247, 'oxa': 248, 'oxamide': 249, 'oxide': 250, 'oxido': 251, 'oxino': 252, 'oxo': 253, 'oxy': 254, 'pent': 255, 'penta': 256, 'pentalen': 257, 'pentanoyl': 258, 'pentyl': 259, 'perimidin': 260, 'peroxy': 261, 'phen': 262, 'phenacyl': 263, 'phenanthren': 264, 'phenanthridin': 265, 'phenanthro': 266, 'phenanthrolin': 267, 'pheno': 268, 'phenyl': 269, 'phosph': 270, 'phosphane': 271, 'phosphanium': 272, 'phosphanyl': 273, 'phosphate': 274, 'phosphonato': 275, 'phosphonia': 276, 'phosphoryl': 277, 'phthalate': 278, 'phthalazin': 279, 'piperazin': 280, 'piperidin': 281, 'prop': 282, 'propa': 283, 'propanoyl': 284, 'propyl': 285, 'pteridin': 286, 'purin': 287, 'pyran': 288, 'pyrazin': 289, 'pyrazol': 290, 'pyren': 291, 'pyridazin': 292, 'pyridin': 293, 'pyrido': 294, 'pyrimidin': 295, 'pyrimido': 296, 'pyrrol': 297, 'pyrrolidin': 298, 'pyrrolizin': 299, 'quinazolin': 300, 'quinolin': 301, 'quinolizin': 302, 'quinoxalin': 303, 'spiro': 304, 'sulfamate': 305, 'sulfamoyl': 306, 'sulfanium': 307, 'sulfanyl': 308, 'sulfanylidene': 309, 'sulfate': 310, 'sulfido': 311, 'sulfinamide': 312, 'sulfinic': 313, 'sulfino': 314, 'sulfinyl': 315, 'sulfonamide': 316, 'sulfonamido': 317, 'sulfonate': 318, 'sulfonato': 319, 'sulfonimidoyl': 320, 'sulfono': 321, 'sulfonyl': 322, 'terephthalate': 323, 'tert-butyl': 324, 'tetr': 325, 'tetra': 326, 'tetracen': 327, 'thi': 328, 'thia': 329, 'thian': 330, 'thio': 331, 'thiocyanate': 332, 'thiol': 333, 'thiolo': 334, 'thione': 335, 'thiophen': 336, 'thiourea': 337, 'tri': 338, 'tria': 339, 'tris': 340, 'trityl': 341, 'un': 342, 'urea': 343, 'xanthen': 344, 'yl': 345, 'yn': 346, 'yne': 347, '[PAD]': 0, '[CLS]': 1, '[EOS]': 2, '[unk]': 3, ';': 4, '.': 5, '>': 6}
        # if vocab is None:
        #     names = [iupac_tokenizer(name) for name in names]
        #     for i in range(len(names)):
        #         tokens = names[i]
        #         if tokens is not None:
        #             tokens = set(tokens)
        #             tokens -= set(self.special_tokens)
        #             vocab |= tokens
        #     nspec = len(self.special_tokens)
        #     self.vocab = dict(zip(sorted(vocab),
        #                         range(nspec, nspec+len(vocab))))
        # else:
        self.vocab = vocab
        # for i,spec in enumerate(self.special_tokens):
        #     self.vocab[spec] = i
        self.rev_vocab = dict((v,k) for k,v in self.vocab.items())
        self.vocsize = len(self.vocab)        

    def tokenize(self,iupac_name):
        tokens = iupac_tokenizer(iupac_name)
        if tokens is None:
            raise Exception(f"Unable to tokenize IUPAC name: {iupac_name}")
        else:
            return tokens
    
    def convert_tokens_to_ids(self, tokens):
        
        return [self.vocab.get(t, self.vocab['[unk]']) for t in tokens]

    def convert_ids_to_tokens(self,ids):
        
        return [self.rev_vocab.get(i, '[unk]') for i in ids]

"""
selfies tokenizer
"""

import selfies as sf
def selfies_tokenizer(sel):
    word = []
    i = 0
    while i<len(sel):
        word_space1 = []
        if sel[i]=="[":
            word_space1.append(sel[i])
            i=i+1
            while sel[i]!="]":
                word_space1.append(sel[i])
                i=i+1
            word_space1.append(sel[i])
            word_space2=''.join(word_space1)
            word.append(word_space2)
            i=i+1   
        else:
            word_space1.append(sel[i])
            word.append(word_space1)
            i=i+1  
            
    nums_list = [ s for s in word]
    return nums_list

class Selfies_tokenizer(object):
    def __init__(self):
        self.vocab =  {'[PAD]': 0, '[CLS]': 1, '[MASK]': 2, '[EOS]': 3, '[unk]': 4, '[#Branch1]': 5, '[#Branch2]': 6, '[#C]': 7, '[#N+1]': 8, '[#N]': 9, '[-/Ring1]': 10, '[-/Ring2]': 11, '[-\\Ring1]': 12, '[/Br]': 13, '[/C@@H1]': 14, '[/C@@]': 15, '[/C@H1]': 16, '[/C@]': 17, '[/C]': 18, '[/Cl]': 19, '[/F]': 20, '[/N+1]': 21, '[/N-1]': 22, '[/NH1+1]': 23, '[/NH1-1]': 24, '[/NH1]': 25, '[/NH2+1]': 26, '[/N]': 27, '[/O+1]': 28, '[/O-1]': 29, '[/O]': 30, '[/S-1]': 31, '[/S@]': 32, '[/S]': 33, '[=Branch1]': 34, '[=Branch2]': 35, '[=C]': 36, '[=N+1]': 37, '[=N-1]': 38, '[=NH1+1]': 39, '[=NH2+1]': 40, '[=N]': 41, '[=O+1]': 42, '[=OH1+1]': 43, '[=O]': 44, '[=P@@]': 45, '[=P@]': 46, '[=PH2]': 47, '[=P]': 48, '[=Ring1]': 49, '[=Ring2]': 50, '[=S+1]': 51, '[=S@@]': 52, '[=S@]': 53, '[=SH1+1]': 54, '[=S]': 55, '[Br]': 56, '[Branch1]': 57, '[Branch2]': 58, '[C@@H1]': 59, '[C@@]': 60, '[C@H1]': 61, '[C@]': 62, '[CH1-1]': 63, '[CH2-1]': 64, '[C]': 65, '[Cl]': 66, '[F]': 67, '[I]': 68, '[N+1]': 69, '[N-1]': 70, '[NH1+1]': 71, '[NH1-1]': 72, '[NH1]': 73, '[NH2+1]': 74, '[NH3+1]': 75, '[N]': 76, '[O-1]': 77, '[O]': 78, '[P+1]': 79, '[P@@H1]': 80, '[P@@]': 81, '[P@]': 82, '[PH1+1]': 83, '[PH1]': 84, '[P]': 85, '[Ring1]': 86, '[Ring2]': 87, '[S+1]': 88, '[S-1]': 89, '[S@@+1]': 90, '[S@@]': 91, '[S@]': 92, '[S]': 93, '[\\Br]': 94, '[\\C@@H1]': 95, '[\\C@H1]': 96, '[\\C]': 97, '[\\Cl]': 98, '[\\F]': 99, '[\\I]': 100, '[\\N+1]': 101, '[\\N-1]': 102, '[\\NH1+1]': 103, '[\\NH1]': 104, '[\\NH2+1]': 105, '[\\N]': 106, '[\\O-1]': 107, '[\\O]': 108, '[\\S-1]': 109, '[\\S@]': 110, '[\\S]': 111}
        self.rev_vocab = dict((v,k) for k,v in self.vocab.items())
        self.vocsize = len(self.vocab)        

    def tokenize(self,sfs_name):
        tokens = selfies_tokenizer(sfs_name)
        if tokens is None:
            raise Exception(f"Unable to tokenize IUPAC name: {sfs_name}")
        else:
            return tokens
    
    def convert_tokens_to_ids(self, tokens):
        num_list = []
        keys = list(self.vocab.keys())
        for t in tokens:
            if t not in keys:
                num_list.append(4)
            else:
                num_list.append(self.vocab.get(t))
        return num_list
    
        # num_list =  [self.vocab.get(t) if self.vocab.get(t) else 4 for t in tokens]

    def convert_ids_to_tokens(self,ids):
        
        return [self.rev_vocab.get(i, '[unk]') for i in ids]