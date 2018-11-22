import os
import lxml.etree as etree


def generate_about_xml(path, outputName):
    output = os.path.join(path, outputName)
    with open(output, mode='w') as out:
        for file in os.listdir(path):
            if file.endswith('.xml'):
                out.write(file.split('.')[0] + '\n')

def generate_about_txt(path, outputName):
    output = os.path.join(path, outputName)
    with open(output, mode='w') as out:
        for file in os.listdir(path):
            if file.endswith('org.txt'):
                out.write(file[0:-7] + '\n')


def changeXML(path, default='MOTFromWinDataSet'):
    for file in os.listdir(path):
        if file.endswith('.xml'):
            file = os.path.join(path, file)
            with open(file,mode='r') as o:
                x=o.readlines()
                line=[]
                for i in x:
                    if i.find('folder')>=0:
                        line+=['    <folder>11111</folder>\n']
                    else:
                        line+=i
            with open(file,mode='w') as o:
                o.writelines(line)
            #tree = etree.parse(file)
            #folder = tree.xpath('//folder')
            #folder[0].text = default
            #tree.write(file, xml_declaration=False)
    pass


if __name__ == '__main__':
    PATH = 'F:\\PostGraduate\\DataSet\\Others\\CarDataSetForHe\\CarDataSetForHe'
    generate_about_txt(path=PATH, outputName='train.txt')
