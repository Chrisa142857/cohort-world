"""
Code -> English description.

Primary source = the authoritative UK Biobank GP-clinical dictionary
(`code_lookup.pkl`, built from UKB resource 592 `all_lkps_maps_v4.xlsx`:
read_v2_lkp / read_ctv3_lkp / icd10_lkp). Covers ~157k Read v2 + ~332k CTV3 + ~14k
ICD-10 codes, so the great majority of GP/HES events render in full English.

If the lookup is missing, or a code is absent, we fall back to the curated tables
below (brain/cognition + common primary-care codes) and finally to the raw code.

describe(code, system) -> str   where system in {"icd10","read2","read3"}.
"""
import os as _os
import pickle as _pickle

_LOOKUP = {"read2": {}, "read3": {}, "icd10": {}}
_lookup_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "code_lookup.pkl")
if _os.path.exists(_lookup_path):
    try:
        with open(_lookup_path, "rb") as _f:
            _LOOKUP = _pickle.load(_f)
    except Exception:
        pass


def _auth(code, system):
    """authoritative dictionary lookup; None if absent."""
    d = _LOOKUP.get(system)
    if not d:
        return None
    if system == "icd10":
        return d.get(code.replace(".", "").replace(" ", "").upper())
    return d.get(code)

# ---------------------------------------------------------------------------
# ICD-10 (3-character) descriptions. Curated, not exhaustive.
# ---------------------------------------------------------------------------
ICD10_3 = {
    # --- brain / cognition (outcome-relevant) ---
    "F00": "Dementia in Alzheimer's disease", "F01": "Vascular dementia",
    "F02": "Dementia in other diseases", "F03": "Unspecified dementia",
    "F04": "Organic amnesic syndrome", "F05": "Delirium",
    "F06": "Other mental disorder due to brain damage", "F07": "Organic personality disorder",
    "F09": "Unspecified organic mental disorder",
    "G30": "Alzheimer's disease", "G31": "Other degenerative disease of nervous system",
    "G20": "Parkinson's disease", "G21": "Secondary parkinsonism", "G22": "Parkinsonism in other diseases",
    "G35": "Multiple sclerosis", "G40": "Epilepsy", "G41": "Status epilepticus",
    "I60": "Subarachnoid haemorrhage", "I61": "Intracerebral haemorrhage",
    "I62": "Other nontraumatic intracranial haemorrhage", "I63": "Cerebral infarction",
    "I64": "Stroke, not specified as haemorrhage or infarction",
    "I65": "Occlusion of precerebral arteries", "I66": "Occlusion of cerebral arteries",
    "I67": "Other cerebrovascular disease", "I69": "Sequelae of cerebrovascular disease",
    "G45": "Transient cerebral ischaemic attack (TIA)",
    "F32": "Depressive episode", "F33": "Recurrent depressive disorder",
    "F34": "Persistent mood disorder", "F41": "Anxiety disorder", "F43": "Reaction to severe stress",
    "S06": "Intracranial (head) injury",
    # --- common comorbidities / risk factors ---
    "E10": "Type 1 diabetes mellitus", "E11": "Type 2 diabetes mellitus",
    "E14": "Unspecified diabetes mellitus", "E66": "Obesity", "E78": "Disorder of lipoprotein metabolism",
    "E03": "Hypothyroidism", "E05": "Hyperthyroidism",
    "I10": "Essential hypertension", "I11": "Hypertensive heart disease",
    "I20": "Angina pectoris", "I21": "Acute myocardial infarction", "I25": "Chronic ischaemic heart disease",
    "I48": "Atrial fibrillation and flutter", "I50": "Heart failure", "I49": "Other cardiac arrhythmia",
    "I70": "Atherosclerosis", "I73": "Peripheral vascular disease", "I80": "Phlebitis/thrombophlebitis",
    "N18": "Chronic kidney disease", "N17": "Acute kidney failure", "N39": "Urinary tract disorder",
    "J44": "Chronic obstructive pulmonary disease (COPD)", "J45": "Asthma",
    "J18": "Pneumonia", "J22": "Lower respiratory infection",
    "K21": "Gastro-oesophageal reflux", "K29": "Gastritis", "K59": "Functional intestinal disorder",
    "M10": "Gout", "M15": "Polyosteoarthritis", "M17": "Osteoarthritis of knee",
    "M19": "Osteoarthritis", "M54": "Dorsalgia (back pain)", "M81": "Osteoporosis",
    "C50": "Malignant neoplasm of breast", "C61": "Malignant neoplasm of prostate",
    "C34": "Malignant neoplasm of lung", "C18": "Malignant neoplasm of colon",
    "Z00": "General examination", "Z01": "Special examination", "Z03": "Observation for suspected disease",
    "Z51": "Medical care (e.g. chemotherapy/follow-up)", "Z86": "Personal history of disease",
    "Z87": "Personal history of other diseases", "R10": "Abdominal/pelvic pain",
    "R07": "Pain in throat and chest", "R55": "Syncope and collapse", "R51": "Headache",
    "R26": "Abnormalities of gait and mobility", "R41": "Cognitive/awareness symptom",
    "F10": "Mental disorder due to alcohol", "F17": "Mental disorder due to tobacco use",
    "G47": "Sleep disorder", "G62": "Polyneuropathy", "G56": "Mononeuropathy of upper limb",
    "H25": "Senile cataract", "H40": "Glaucoma", "H90": "Hearing loss",
}

# ICD-10 chapter fallback by first letter.
ICD10_CHAPTER = {
    "A": "infectious disease", "B": "infectious disease",
    "C": "malignant neoplasm", "D": "neoplasm / blood disorder",
    "E": "endocrine / metabolic disorder", "F": "mental / behavioural disorder",
    "G": "nervous system disease", "H": "eye / ear disorder",
    "I": "circulatory system disease", "J": "respiratory disease",
    "K": "digestive system disease", "L": "skin disorder",
    "M": "musculoskeletal disorder", "N": "genitourinary disease",
    "O": "pregnancy / childbirth", "P": "perinatal condition",
    "Q": "congenital malformation", "R": "symptom / abnormal finding",
    "S": "injury", "T": "injury / poisoning",
    "V": "external cause", "W": "external cause", "X": "external cause", "Y": "external cause",
    "Z": "factor influencing health status / contact with services",
}

# ---------------------------------------------------------------------------
# Read v2 (read_2) descriptions. Curated brain + common primary-care codes.
# ---------------------------------------------------------------------------
READ2 = {
    # dementia / organic (from project's verified list)
    "E00..": "Senile/presenile organic psychotic conditions", "E000.": "Uncomplicated senile dementia",
    "E001.": "Presenile dementia", "E004.": "Arteriosclerotic dementia",
    "Eu00.": "Dementia in Alzheimer's disease", "Eu01.": "Vascular dementia",
    "Eu02.": "Dementia in other diseases", "Eu023": "Dementia in Parkinson's disease",
    "Eu041": "Delirium", "Eu057": "Mild cognitive disorder",
    "F110.": "Alzheimer's disease", "F111.": "Pick's disease", "F112.": "Senile degeneration of brain",
    "F12..": "Parkinson's disease", "F20..": "Multiple sclerosis",
    "F25..": "Epilepsy", "F26..": "Migraine", "F662.": "Restless legs syndrome",
    "G6...": "Cerebrovascular disease", "G66..": "Stroke / cerebrovascular accident",
    "G65..": "Transient cerebral ischaemia (TIA)", "G6X..": "Stroke",
    # depression / mood
    "E2B..": "Depressive disorder", "Eu32.": "Depressive episode", "Eu33.": "Recurrent depression",
    "1B17.": "Depressed mood", "Eu41.": "Anxiety disorder",
    # common chronic disease / risk factors
    "C10..": "Diabetes mellitus", "C108.": "Type 2 diabetes mellitus", "C109.": "Type 1 diabetes mellitus",
    "G20..": "Essential hypertension", "662..": "Hypertension monitoring",
    "G30..": "Ischaemic heart disease", "G3...": "Ischaemic heart disease",
    "G573.": "Atrial fibrillation", "G58..": "Heart failure", "G573": "Atrial fibrillation",
    "C320.": "Hyperlipidaemia", "C32..": "Disorder of lipid metabolism",
    "K05..": "Chronic kidney disease", "H3...": "Chronic lower respiratory disease",
    "H33..": "Asthma", "H3...": "COPD/chronic airway disease",
    "N05..": "Osteoarthritis", "N06..": "Polyarthropathy", "1O1..": "Osteoarthritis",
    "F4...": "Eye disease", "1B...": "General symptom",
    "1372.": "Smoker", "137R.": "Current non-smoker", "137S.": "Ex-smoker", "1371.": "Never smoked",
    "136..": "Alcohol consumption", "22K..": "Body mass index", "246..": "Blood pressure reading",
    "44P..": "Serum cholesterol", "42W..": "Haemoglobin A1c", "451..": "Serum creatinine",
}

# Read v2 chapter fallback (first char). Read v2: alpha = clinical, digit = process/admin.
READ2_CHAPTER = {
    "A": "infectious disease", "B": "neoplasm", "C": "endocrine / metabolic / immunity disorder",
    "D": "blood disorder", "E": "mental disorder", "F": "nervous system / sense organ disease",
    "G": "circulatory system disease", "H": "respiratory disease", "J": "digestive system disease",
    "K": "genitourinary disease", "L": "pregnancy / childbirth", "M": "skin disease",
    "N": "musculoskeletal disease", "P": "congenital anomaly", "Q": "perinatal condition",
    "R": "symptom / ill-defined condition", "S": "injury / poisoning",
    "T": "external cause of injury",
    "Eu": "mental/behavioural disorder (ICD-10 mapped)",
}


def _norm_icd(code: str) -> str:
    return code.replace(".", "").replace(" ", "").upper()


def describe_icd10(code: str) -> str:
    c = _norm_icd(code)
    if not c:
        return "(unknown)"
    if c[:3] in ICD10_3:
        return ICD10_3[c[:3]]
    # try 3-char of an ICD-9 numeric? skip; chapter fallback
    chap = ICD10_CHAPTER.get(c[0])
    return f"{chap} [{code}]" if chap else code


def describe_read2(code: str) -> str:
    if not code:
        return "(unknown)"
    if code in READ2:
        return READ2[code]
    # prefix match against curated keys (longest first)
    for k in sorted(READ2, key=len, reverse=True):
        if code.startswith(k.rstrip(".")):
            return READ2[k]
    if code[:2] == "Eu":
        return f"mental/behavioural disorder [{code}]"
    chap = READ2_CHAPTER.get(code[0])
    return f"{chap} [{code}]" if chap else code


def describe_read3(code: str) -> str:
    """CTV3 (read_3). No CTV3 dictionary ships locally, but CTV3 retains many legacy
    Read v2 codes (same string, same chapter semantics). For those we reuse the Read v2
    map / chapter fallback. CTV3-native codes (X.., Xa.., Y.., Ub..) have no structural
    chapter meaning here, so they stay raw (documented route-2 long tail)."""
    if not code:
        return "(unknown)"
    if code[:1] in ("X", "Y") or code[:2] in ("Ub", "Yb"):
        return describe_read2(code) if code in READ2 else code
    return describe_read2(code)


def describe(code: str, system: str) -> str:
    if not code:
        return "(unknown)"
    # 1) authoritative UKB dictionary (covers the vast majority of GP/HES codes)
    a = _auth(code, system)
    if a:
        return a
    # 2) curated fallbacks
    if system == "icd10":
        return describe_icd10(code)
    if system == "read2":
        return describe_read2(code)
    if system == "read3":
        return describe_read3(code)
    return code


if __name__ == "__main__":
    print(f"lookup sizes: " + ", ".join(f"{k}={len(v)}" for k, v in _LOOKUP.items()))
    for c, s in [("F001", "icd10"), ("I639", "icd10"), ("22A..", "read2"), ("42P..", "read2"),
                 ("9N11.", "read2"), ("XE2eD", "read3"), ("XM0lt", "read3"), ("XaF6J", "read3"),
                 ("8B314", "read2"), ("Ua0TB", "read3")]:
        print(f"{s:6} {c:8} -> {describe(c, s)}")
