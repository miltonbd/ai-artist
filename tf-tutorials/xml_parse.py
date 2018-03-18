import xml.etree.ElementTree as ET 

xml=ET.parse("test_xml.xml")

root=xml.getroot()

for child in root:
    print child.attrib