import tarfile
import os
import shutil
import re
from zipfile import ZipFile

datasets = ["DeEnGoldAlignment",
    "English-French.test",
    "ep-ensv-alignref.v2015-10-12",
    "kftt-alignments",
    "Romanian-English.test",
    "TsinghuaAlignmentEvalSet"]

ALIGN6 = True

def main():
    if os.path.exists(f'./data/pre_processed_data/accAlign'):
        shutil.rmtree(f'./data/pre_processed_data/accAlign')
    
    if os.path.exists(f'./data/pre_processed_data/awesomeAlign'):
        shutil.rmtree(f'./data/pre_processed_data/awesomeAlign')
    
    if os.path.exists(f'./data/datasets'):
        shutil.rmtree(f'./data/datasets')  

    os.mkdir(f'./data/pre_processed_data/accAlign')
    os.mkdir(f'./data/pre_processed_data/awesomeAlign')
    os.mkdir(f'./data/datasets')

    open(f'./data/pre_processed_data/accAlign/.gitkeep', 'a').close()
    open(f'./data/pre_processed_data/awesomeAlign/.gitkeep', 'a').close()
    open(f'./data/datasets/.gitkeep', 'a').close()

    # First extract all tar files
    for dataset in datasets:
        # open file
        file = tarfile.open(f'./data/zipped_datasets/{dataset}.tar.gz')
        
        # extracting file
        file.extractall(f'./data/datasets/')
        
        file.close()

    if ALIGN6:
        zf = ZipFile('./data/zipped_datasets/ALIGN6.zip', 'r')
        zf.extractall('./data/datasets/')
        zf.close()

        zf = ZipFile('./data/datasets/ALIGN6/dev_data.zip', 'r')
        zf.extractall('./data/datasets/ALIGN6/')
        zf.close()

        zf = ZipFile('./data/datasets/ALIGN6/train_data.zip', 'r')
        zf.extractall('./data/datasets/ALIGN6/')
        zf.close()
            
    if "DeEnGoldAlignment" in datasets:
        # Set up directories
        os.mkdir(f'./data/pre_processed_data/accAlign/DeEn')
        os.mkdir(f'./data/pre_processed_data/awesomeAlign/DeEn')

        enAccAlign = open(f'./data/pre_processed_data/accAlign/DeEn/en', 'w') 
        deAccAlign = open(f'./data/pre_processed_data/accAlign/DeEn/de', 'w')

        endeAwesomeAlign = open(f'./data/pre_processed_data/awesomeAlign/DeEn/deen.src-tgt', 'w') 


        en = open(f'./data/datasets/DeEn/en', 'r', encoding='ISO-8859-1')
        enLines = en.readlines()
        de = open(f'./data/datasets/DeEn/de', 'r', encoding='ISO-8859-1')
        deLines = de.readlines()

        # Generate target and source files
        for enLine, deLine in zip(enLines, deLines):
            if enLine.strip() != "" and deLine.strip() != "":
                enAccAlign.write(enLine)
                deAccAlign.write(deLine)
                endeAwesomeAlign.write(f'{deLine.strip()} ||| {enLine.strip()}\n')


        # Generate gold alignment files
        deenGoldAccAlign = open(f'./data/pre_processed_data/accAlign/DeEn/alignmentDeEn.talp', 'w') 
        deenGoldAwesomeAlign = open(f'./data/pre_processed_data/awesomeAlign/DeEn/deen.gold', 'w') 

        goldAlignments = open('./data/datasets/DeEn/alignmentDeEn.talp', 'r')
        goldAlignmentLines = goldAlignments.readlines()

        for line in goldAlignmentLines:
            if line.strip() != "":
                deenGoldAccAlign.write(line.strip() + "\n")
                deenGoldAwesomeAlign.write(line.strip() + "\n")

        deenGoldAccAlign.close()
        deenGoldAwesomeAlign.close()
        goldAlignments.close()
        
        enAccAlign.close()
        deAccAlign.close()
        endeAwesomeAlign.close()
        en.close()
        de.close()
    
    if "English-French.test" in datasets:
        # Set up directories
        os.mkdir(f'./data/pre_processed_data/accAlign/EnFr')
        os.mkdir(f'./data/pre_processed_data/awesomeAlign/EnFr')

        enAccAlign = open(f'./data/pre_processed_data/accAlign/EnFr/en', 'w') 
        frAccAlign = open(f'./data/pre_processed_data/accAlign/EnFr/fr', 'w')

        enfrAwesomeAlign = open(f'./data/pre_processed_data/awesomeAlign/EnFr/enfr.src-tgt', 'w') 


        en = open(f'./data/datasets/English-French/test/test.e', 'r', encoding='ISO-8859-1')
        enLines = en.readlines()
        fr = open(f'./data/datasets/English-French/test/test.f', 'r', encoding='ISO-8859-1')
        frLines = fr.readlines()

        # Generate target and source files
        for enLine, frLine in zip(enLines, frLines):
            enMatch = re.search('<.*>(.*)<\/s>', enLine, re.IGNORECASE)
            frMatch = re.search('<.*>(.*)<\/s>', frLine, re.IGNORECASE)

            if enMatch and frMatch:
                enAccAlign.write(f'{enMatch.group(1)}\n')
                frAccAlign.write(f'{frMatch.group(1)}\n')
                enfrAwesomeAlign.write(f'{enMatch.group(1)} ||| {frMatch.group(1)}\n')

        enAccAlign.close()
        frAccAlign.close()
        enfrAwesomeAlign.close()
        en.close()
        fr.close()

        # Generate gold alignment files
        enfrGoldAccAlign = open(f'./data/pre_processed_data/accAlign/EnFr/alignmentEnFr.talp', 'w') 
        enfrGoldAwesomeAlign = open(f'./data/pre_processed_data/awesomeAlign/EnFr/enfr.gold', 'w') 

        goldAlignments = open('./data/datasets/English-French/answers/test.wa.nonullalign', 'r')
        goldAlignmentLines = goldAlignments.readlines()
        
        curr = "0001"
        temp = ""
        for line in goldAlignmentLines:
            line = line.strip().split()
            if line[0] == curr:
                if line[3] == "S":
                    temp += f"{line[1]}-{line[2]} "
                elif line[3] == "P":
                    temp += f"{line[1]}p{line[2]} "
            else:
                temp += "\n"
                enfrGoldAccAlign.write(temp)
                enfrGoldAwesomeAlign.write(temp)
                if line[3] == "S":
                    temp = f"{line[1]}-{line[2]} "
                elif line[3] == "P":
                    temp = f"{line[1]}p{line[2]} "
                curr = line[0]
        temp += "\n"
        enfrGoldAccAlign.write(temp)
        enfrGoldAwesomeAlign.write(temp)

        enfrGoldAccAlign.close()
        enfrGoldAwesomeAlign.close()
        goldAlignments.close()

    if "ep-ensv-alignref.v2015-10-12" in datasets:
        # Set up directories
        os.mkdir(f'./data/pre_processed_data/accAlign/EnSv')
        os.mkdir(f'./data/pre_processed_data/awesomeAlign/EnSv')

        enAccAlign = open(f'./data/pre_processed_data/accAlign/EnSv/en', 'w') 
        svAccAlign = open(f'./data/pre_processed_data/accAlign/EnSv/sv', 'w')

        ensvAwesomeAlign = open(f'./data/pre_processed_data/awesomeAlign/EnSv/ensv.src-tgt', 'w') 

        en = open(f'./data/datasets/ep-ensv-alignref.v2015-10-12/test/test.en.naacl', 'r')
        enLines = en.readlines()
        sv = open(f'./data/datasets/ep-ensv-alignref.v2015-10-12/test/test.sv.naacl', 'r')
        svLines = sv.readlines()

        # Generate target and source files
        for enLine, svLine in zip(enLines, svLines):
            enMatch = re.search('<.*>(.*)<\/s>', enLine, re.IGNORECASE)
            svMatch = re.search('<.*>(.*)<\/s>', svLine, re.IGNORECASE)

            if enMatch and svMatch:
                enAccAlign.write(f'{enMatch.group(1)}\n')
                svAccAlign.write(f'{svMatch.group(1)}\n')
                ensvAwesomeAlign.write(f'{enMatch.group(1)} ||| {svMatch.group(1)}\n')

        enAccAlign.close()
        svAccAlign.close()
        ensvAwesomeAlign.close()
        en.close()
        sv.close()

        # Generate gold alignment files
        ensvGoldAccAlign = open(f'./data/pre_processed_data/accAlign/EnSv/alignmentEnSv.talp', 'w') 
        ensvGoldAwesomeAlign = open(f'./data/pre_processed_data/awesomeAlign/EnSv/ensv.gold', 'w') 

        goldAlignments = open('./data/datasets/ep-ensv-alignref.v2015-10-12/test/test.ensv.naacl', 'r')
        goldAlignmentLines = goldAlignments.readlines()
        
        curr = "1"
        temp = ""
        for line in goldAlignmentLines:
            line = line.strip().split()
            if line[0] == curr:
                if line[1] == "0" or line[2] == "0":
                    continue
                elif len(line) == 4 and line[3] == "P":
                    temp += f"{int(line[1])}p{int(line[2])} "
                else:
                    temp += f"{int(line[1])}-{int(line[2])} "
            else:
                temp += "\n"
                ensvGoldAccAlign.write(temp)
                ensvGoldAwesomeAlign.write(temp)
                if line[1] == "0" or line[2] == "0":
                    temp = ""
                elif len(line) == 4 and line[3] == "P":
                    temp = f"{int(line[1])}p{int(line[2])} "
                else:
                    temp = f"{int(line[1])}-{int(line[2])} "
                curr = line[0]
        temp += "\n"
        ensvGoldAccAlign.write(temp)
        ensvGoldAwesomeAlign.write(temp)

        ensvGoldAccAlign.close()
        ensvGoldAwesomeAlign.close()
        goldAlignments.close()

    if "kftt-alignments" in datasets:
        # Set up directories
        os.mkdir(f'./data/pre_processed_data/accAlign/JaEn')
        os.mkdir(f'./data/pre_processed_data/awesomeAlign/JaEn')

        # Copy over text files for accAlign
        shutil.copyfile('./data/datasets/kftt-alignments/data/english-test.txt', './data/pre_processed_data/accAlign/JaEn/en')
        shutil.copyfile('./data/datasets/kftt-alignments/data/japanese-test.txt', './data/pre_processed_data/accAlign/JaEn/ja')

        jaenAwesomeAlign = open(f'./data/pre_processed_data/awesomeAlign/JaEn/jaen.src-tgt', 'w') 

        en = open(f'./data/datasets/kftt-alignments/data/english-test.txt', 'r')
        enLines = en.readlines()
        ja = open(f'./data/datasets/kftt-alignments/data/japanese-test.txt', 'r')
        jaLines = ja.readlines()

        # Generate target and source files
        for enLine, jaLine in zip(enLines, jaLines):
            if enLine.strip() != "" and jaLine.strip() != "":
                jaenAwesomeAlign.write(f'{jaLine.strip()} ||| {enLine.strip()}\n')

        jaenAwesomeAlign.close()
        en.close()
        ja.close()

        # Copy over gold alignment file
        shutil.copyfile('./awesome-align/examples/jaen.gold', './data/pre_processed_data/accAlign/JaEn/alignmentJaEn.talp')
        shutil.copyfile('./awesome-align/examples/jaen.gold', './data/pre_processed_data/awesomeAlign/JaEn/jaen.gold')

    if "Romanian-English.test" in datasets:
        # Set up directories
        os.mkdir(f'./data/pre_processed_data/accAlign/RoEn')
        os.mkdir(f'./data/pre_processed_data/awesomeAlign/RoEn')

        roAccAlign = open(f'./data/pre_processed_data/accAlign/RoEn/ro', 'w') 
        enAccAlign = open(f'./data/pre_processed_data/accAlign/RoEn/en', 'w')

        shutil.copyfile('./awesome-align/examples/roen.src-tgt', './data/pre_processed_data/awesomeAlign/RoEn/roen.src-tgt')

        sentence_file = open(f'./awesome-align/examples/roen.src-tgt', 'r')
        sentences = sentence_file.readlines()

        for sentence in sentences:
            sentence = sentence.strip().split("|||")
            roAccAlign.write(f'{sentence[0].strip()}\n')
            enAccAlign.write(f'{sentence[1].strip()}\n')

        enAccAlign.close()
        roAccAlign.close()

        # Generate gold alignment files
        shutil.copyfile('./awesome-align/examples/roen.gold', './data/pre_processed_data/accAlign/RoEn/alignmentRoEn.talp')
        shutil.copyfile('./awesome-align/examples/roen.gold', './data/pre_processed_data/awesomeAlign/RoEn/roen.gold')

    if "TsinghuaAlignmentEvalSet" in datasets:
        # Set up directories
        os.mkdir(f'./data/pre_processed_data/accAlign/ZhEn')
        os.mkdir(f'./data/pre_processed_data/awesomeAlign/ZhEn')

        enAccAlign = open(f'./data/pre_processed_data/accAlign/ZhEn/en', 'w') 
        zhAccAlign = open(f'./data/pre_processed_data/accAlign/ZhEn/zh', 'w')

        zhenAwesomeAlign = open(f'./data/pre_processed_data/awesomeAlign/ZhEn/zhen.src-tgt', 'w') 

        zh = open(f'./data/datasets/TsinghuaAlignmentEvalSet/v1/tstset/tst.f', 'r', encoding='UTF-8')
        zhLines = zh.readlines()
        en = open(f'./data/datasets/TsinghuaAlignmentEvalSet/v1/tstset/tst.e', 'r', encoding='Latin-1')
        enLines = en.readlines()

        # Generate target and source files
        for zhLine, enLine in zip(zhLines, enLines):
            enAccAlign.write(enLine)
            zhAccAlign.write(zhLine)
            if enLine.strip() != "" and zhLine.strip() != "":
                zhenAwesomeAlign.write(f'{zhLine.strip()} ||| {enLine.strip()}\n')
        
        enAccAlign.close()
        zhAccAlign.close()
        zhenAwesomeAlign.close()
        zh.close()
        en.close()

        # Generate gold alignment files
        zhenGoldAccAlign = open(f'./data/pre_processed_data/accAlign/ZhEn/alignmentZhEn.talp', 'w') 
        zhenGoldAwesomeAlign = open(f'./data/pre_processed_data/awesomeAlign/ZhEn/zhen.gold', 'w') 

        goldAlignments = open('./data/datasets/TsinghuaAlignmentEvalSet/v1/tstset/tst.wa', 'r')
        goldAlignmentLines = goldAlignments.readlines()
        
        for line in goldAlignmentLines:
            temp = ""
            for align in line.strip().split():
                alignMatch = re.search('(.*):(.*)\/', align, re.IGNORECASE)
                if alignMatch:
                    if "/1" in align:
                        temp += f"{alignMatch.group(1)}-{alignMatch.group(2)} "
                    else:
                        temp += f"{alignMatch.group(1)}p{alignMatch.group(2)} "
            temp += "\n"
            zhenGoldAccAlign.write(temp)
            zhenGoldAwesomeAlign.write(temp)

        zhenGoldAccAlign.close()
        zhenGoldAwesomeAlign.close()
        goldAlignments.close()
        
    if ALIGN6:
        # Set up directories
        os.mkdir(f'./data/pre_processed_data/accAlign/ALIGN6')
        os.mkdir(f'./data/pre_processed_data/awesomeAlign/ALIGN6')
        os.mkdir(f'./data/pre_processed_data/awesomeAlign/ALIGN6/dev_data')
        os.mkdir(f'./data/pre_processed_data/awesomeAlign/ALIGN6/train_data')

        shutil.copytree('./data/datasets/ALIGN6/dev_data', './data/pre_processed_data/accAlign/ALIGN6/dev_data')
        shutil.copytree('./data/datasets/ALIGN6/train_data', './data/pre_processed_data/accAlign/ALIGN6/train_data')

        ALIGN6_dev = open(f'./data/pre_processed_data/awesomeAlign/ALIGN6/dev_data/dev.src-tgt', 'w') 

        dev_src = open(f'./data/datasets/ALIGN6/dev_data/dev.src', 'r', encoding='UTF-8')
        src_lines = dev_src.readlines()
        dev_tgt = open(f'./data/datasets/ALIGN6/dev_data/dev.tgt', 'r', encoding='UTF-8')
        tgt_lines = dev_tgt.readlines()
        shutil.copyfile('./data/datasets/ALIGN6/dev_data/dev.talp', './data/pre_processed_data/awesomeAlign/ALIGN6/dev_data/dev.gold')

        # Generate target and source files
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            if src_line.strip() != "" and tgt_line.strip() != "":
                ALIGN6_dev.write(f'{src_line.strip()} ||| {tgt_line.strip()}\n')
        ALIGN6_dev.close()
        dev_src.close()
        dev_tgt.close()

        ALIGN6_test = open(f'./data/pre_processed_data/awesomeAlign/ALIGN6/train_data/train.src-tgt', 'w') 

        train_src = open(f'./data/datasets/ALIGN6/train_data/train.src', 'r', encoding='UTF-8')
        src_lines = train_src.readlines()
        train_tgt = open(f'./data/datasets/ALIGN6/train_data/train.tgt', 'r', encoding='UTF-8')
        tgt_lines = train_tgt.readlines()
        shutil.copyfile('./data/datasets/ALIGN6/train_data/train.talp', './data/pre_processed_data/awesomeAlign/ALIGN6/train_data/train.gold')

        # Generate target and source files
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            if src_line.strip() != "" and tgt_line.strip() != "":
                ALIGN6_test.write(f'{src_line.strip()} ||| {tgt_line.strip()}\n')
        ALIGN6_test.close()
        train_src.close()
        train_tgt.close()

if __name__ == "__main__":
    main()